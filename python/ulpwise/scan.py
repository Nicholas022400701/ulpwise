"""``ulpwise scan``: read a repository for the floating point patterns that produced the bugs in the
corpus, and rank its functions by how much elementary math they do.

The scan is static. It parses every Python file with :mod:`ast`, walks each function and reports
the expressions below with the file, the line and the function. It does not import or run the
code, so it works on a repository whose dependencies are not installed. The report is a reading
list, not a verdict: every finding names the pattern, why it loses digits or overflows, and the
usual replacement.

``--run`` adds a dynamic half: the module level functions with the most elementary math are
imported and called with the same grid in float32 and float64, and the float32 result is measured
in ulps against the float64 one. That imports and runs the repository's code.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

EXP = {"exp", "exp2"}
LOG = {"log", "log2", "log10"}
SQRT = {"sqrt"}
SIN = {"sin", "cos", "tan"}
INVERSE_TRIG = {"acos", "arccos", "asin", "arcsin", "atan", "arctan"}
ELEMENTARY = (
    EXP | LOG | SQRT | SIN | INVERSE_TRIG
    | {"expm1", "log1p", "atan2", "arctan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "pow", "rsqrt",
       "erf", "erfc", "erfinv", "lgamma", "digamma", "polygamma", "gamma", "sigmoid", "softplus", "logsumexp",
       "hypot", "norm", "cross", "det", "inverse", "inv", "solve", "cholesky", "svd", "eig", "eigh", "qr"}
)

SKIP_DIRS = {".git", "__pycache__", "build", "dist", "docs", "doc", "examples", "benchmarks", "benchmark",
             "site-packages", "node_modules", ".venv", "venv", ".tox", ".eggs"}
TEST_DIRS = {"test", "tests", "testing", "conftest"}

# id: (severity, what it does, the replacement, an upstream example or "")
RULES = {
    "exp-of-square": (
        "high",
        "exp of a squared value: the rounding error of x * x is multiplied by x * x / 2 ulps in the result",
        "split the square with an fma or compute exp(x * x) as exp(hi) * exp(lo) with hi = round(x) ** 2",
        "pytorch #198664, erfcx off by 157 float64 ulps at x = -23.25",
    ),
    "sin-of-pi-times": (
        "high",
        "trigonometric function of pi * x: the product is rounded before the argument reduction, and the error grows with x",
        "reduce x to [-0.5, 0.5] first (x - round(x)) and use the symmetry of the function, or use sinpi / cospi",
        "pytorch #198663, trigamma reflection loses every float32 digit for large negative x",
    ),
    "softplus-by-hand": (
        "high",
        "log(1 + exp(x)): exp overflows to inf for moderate x and log(1 + exp(x)) loses digits for negative x",
        "softplus(x) or logaddexp(x, 0)",
        "",
    ),
    "logsumexp-by-hand": (
        "high",
        "log of a sum of exps: any large term overflows the sum to inf",
        "logsumexp, or subtract the maximum before exp",
        "",
    ),
    "hypot-by-hand": (
        "high",
        "sqrt(a * a + b * b): the squares overflow or underflow long before the result would",
        "hypot(a, b) or norm",
        "",
    ),
    "sqrt-of-difference": (
        "medium",
        "sqrt of a difference: cancellation in a - b costs digits that the sqrt then halves in relative terms, and rounding can push the argument below zero",
        "rewrite as a product where possible (1 - c * c as (1 - c) * (1 + c)), or clamp the argument at 0 with a comment",
        "",
    ),
    "one-minus-cos": (
        "medium",
        "1 - cos(x): all digits cancel for small x",
        "2 * sin(x / 2) ** 2",
        "kornia #4897, the So3 Jacobians for small angles",
    ),
    "log1p-by-hand": (
        "medium",
        "log(1 + x): 1 + x rounds x away for small x",
        "log1p(x)",
        "",
    ),
    "expm1-by-hand": (
        "medium",
        "exp(x) - 1: cancellation for small x",
        "expm1(x)",
        "",
    ),
    "atan-of-quotient": (
        "medium",
        "atan(y / x): the quotient overflows or loses the quadrant",
        "atan2(y, x)",
        "",
    ),
    "small-angle-division": (
        "medium",
        "division by an angle or by sin of it in a function that also takes sin or cos of that angle, with no visible guard",
        "a series for small angles, or a where / clamp with an explicit threshold",
        "kornia #4838 and #4897, So3.log and the Jacobians",
    ),
    "acos-for-angle": (
        "info",
        "acos or asin to recover an angle: the derivative is infinite at the ends, so the angle near 0 or pi loses half its digits",
        "atan2 of the cross product norm and the dot product",
        "",
    ),
}
SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}


@dataclass
class Finding:
    rule: str
    path: str
    line: int
    function: str
    snippet: str

    @property
    def severity(self) -> str:
        return RULES[self.rule][0]


@dataclass
class HotSpot:
    path: str
    line: int
    function: str
    calls: int
    divisions: int
    names: List[str]


def _callee(node: ast.AST) -> Tuple[Optional[str], Optional[ast.AST]]:
    """Name of the function called and its first argument, for ``f(x)``, ``mod.f(x)`` and the method
    form ``x.f()``. Returns (None, None) when ``node`` is not such a call."""
    if not isinstance(node, ast.Call):
        return None, None
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id, node.args[0] if node.args else None
    if isinstance(fn, ast.Attribute):
        if node.args:
            return fn.attr, node.args[0]
        return fn.attr, fn.value  # method form, the receiver is the argument
    return None, None


def _strip(node: ast.AST) -> ast.AST:
    """Drop unary minus and parentheses-like wrappers so ``exp(-(x * x))`` matches ``exp(x * x)``."""
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return node


def _same(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a) == ast.dump(b)


def _is_const(node: ast.AST, value: float) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value == value


def _is_pi(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "pi"
    if isinstance(node, ast.Name):
        return node.id in {"pi", "PI"}
    return isinstance(node, ast.Constant) and isinstance(node.value, float) and abs(node.value - math.pi) < 1e-3


def _is_square(node: ast.AST) -> bool:
    node = _strip(node)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult) and _same(node.left, node.right):
            return True
        if isinstance(node.op, ast.Pow) and _is_const(node.right, 2):
            return True
    name, _ = _callee(node)
    if name == "square":
        return True
    if name in {"pow", "power"} and isinstance(node, ast.Call):
        if len(node.args) == 2 and _is_const(node.args[1], 2):  # pow(x, 2), torch.pow(x, 2)
            return True
        if len(node.args) == 1 and isinstance(node.func, ast.Attribute) and _is_const(node.args[0], 2):  # x.pow(2)
            return True
    return False


def _contains_call(node: ast.AST, names: set) -> bool:
    for sub in ast.walk(node):
        n, _ = _callee(sub)
        if n in names:
            return True
    return False


def _angle_names(body: ast.AST) -> set:
    """Names that appear as the argument of sin, cos or tan inside ``body``."""
    out = set()
    for sub in ast.walk(body):
        n, arg = _callee(sub)
        if n in SIN and isinstance(arg, ast.Name):
            out.add(arg.id)
    return out


_GUARD = re.compile(r"\bwhere\b|\bclamp|\bclip\b|\beps\b|\bfinfo\b|\btaylor\b|\bseries\b|small.angle|\bmasked", re.I)


def _findings_for_function(fn: ast.AST, qualname: str, path: str, source_lines: Sequence[str]) -> Tuple[List[Finding], HotSpot]:
    findings: List[Finding] = []
    calls = 0
    divisions = 0
    names: List[str] = []
    angles = _angle_names(fn)
    fn_source = "\n".join(source_lines[fn.lineno - 1 : getattr(fn, "end_lineno", fn.lineno)])
    guarded = bool(_GUARD.search(fn_source))

    def add(rule: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", fn.lineno)
        findings.append(Finding(rule, path, line, qualname, source_lines[line - 1].strip()[:160]))

    for node in ast.walk(fn):
        name, arg = _callee(node)
        if name in ELEMENTARY:
            calls += 1
            names.append(name)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            divisions += 1
            den = _strip(node.right)
            if isinstance(den, ast.BinOp) and isinstance(den.op, ast.Pow):
                den = den.left  # theta ** 2 divides like theta
            den_name = den.id if isinstance(den, ast.Name) else None
            if not guarded and angles and (den_name in angles or _contains_call(den, SIN)):
                add("small-angle-division", node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            left, right = _strip(node.left), _strip(node.right)
            rn, _ = _callee(right)
            if _is_const(left, 1) and rn == "cos":
                add("one-minus-cos", node)
            ln, _ = _callee(left)
            if ln in EXP and _is_const(right, 1):
                add("expm1-by-hand", node)
        if name is None or arg is None:
            continue
        inner = _strip(arg)
        if name in EXP and _is_square(inner):
            add("exp-of-square", node)
        elif name in SIN:
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Mult) and (_is_pi(inner.left) or _is_pi(inner.right)):
                add("sin-of-pi-times", node)
        elif name in LOG:
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add):
                a, b = _strip(inner.left), _strip(inner.right)
                an, _ = _callee(a)
                bn, _ = _callee(b)
                if (_is_const(a, 1) and bn in EXP) or (_is_const(b, 1) and an in EXP):
                    add("softplus-by-hand", node)
                elif an in EXP and bn in EXP:
                    add("logsumexp-by-hand", node)
                elif _is_const(a, 1) or _is_const(b, 1):
                    add("log1p-by-hand", node)
            else:
                sn, sarg = _callee(inner)
                if sn == "sum" and sarg is not None and _contains_call(sarg, EXP):
                    add("logsumexp-by-hand", node)
        elif name in SQRT:
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add) and _is_square(inner.left) and _is_square(inner.right):
                add("hypot-by-hand", node)
            elif isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub):
                add("sqrt-of-difference", node)
        elif name in {"atan", "arctan"}:
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div):
                add("atan-of-quotient", node)
        elif name in {"acos", "arccos", "asin", "arcsin"}:
            add("acos-for-angle", node)
    seen = set()
    findings = [f for f in findings if not ((f.rule, f.line) in seen or seen.add((f.rule, f.line)))]
    return findings, HotSpot(path, fn.lineno, qualname, calls, divisions, sorted(set(names)))


def _functions(tree: ast.Module) -> Iterable[Tuple[ast.AST, str]]:
    def walk(node: ast.AST, prefix: str) -> Iterable[Tuple[ast.AST, str]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child, prefix + child.name
                yield from walk(child, prefix + child.name + ".")
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, prefix + child.name + ".")
            else:
                yield from walk(child, prefix)

    return walk(tree, "")


def python_files(root: str, include_tests: bool = False) -> List[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".") and (include_tests or d.lower() not in TEST_DIRS)
        )
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            if not include_tests and (f.startswith("test_") or f.endswith("_test.py") or f == "conftest.py"):
                continue
            out.append(os.path.join(dirpath, f))
    return out


def scan_source(source: str, path: str = "<string>") -> Tuple[List[Finding], List[HotSpot]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []
    lines = source.splitlines()
    findings: List[Finding] = []
    spots: List[HotSpot] = []
    for fn, qualname in _functions(tree):
        f, spot = _findings_for_function(fn, qualname, path, lines)
        findings.extend(f)
        if spot.calls:
            spots.append(spot)
    return findings, spots


def scan_tree(root: str, include_tests: bool = False) -> Tuple[List[Finding], List[HotSpot], int]:
    findings: List[Finding] = []
    spots: List[HotSpot] = []
    files = python_files(root, include_tests)
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            continue
        f, s = scan_source(source, os.path.relpath(path, root))
        findings.extend(f)
        spots.extend(s)
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.rule, f.path, f.line))
    spots.sort(key=lambda s: (-s.calls, -s.divisions, s.path, s.line))
    return findings, spots, len(files)


_REPO = re.compile(r"^(?:https?://github\.com/|git@github\.com:)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


def checkout(target: str, workdir: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Return a local directory for ``target`` and the commit it is at. A directory is used as is; a
    GitHub URL or ``owner/repo`` is cloned with depth 1."""
    if os.path.isdir(target):
        try:
            sha = subprocess.run(["git", "-C", target, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            sha = None
        return target, sha
    m = _REPO.match(target)
    if not m:
        raise ValueError(f"{target!r} is neither a directory nor a GitHub repository")
    owner, repo = m.groups()
    dest = os.path.join(workdir or tempfile.mkdtemp(prefix="ulpwise-scan-"), repo)
    if not os.path.isdir(dest):
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", f"https://github.com/{owner}/{repo}", dest], check=True)
    sha = subprocess.run(["git", "-C", dest, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return dest, sha


def render(findings: Sequence[Finding], spots: Sequence[HotSpot], target: str, sha: Optional[str], nfiles: int, top: int = 20, markdown: bool = False) -> str:
    out: List[str] = []
    where = f"{target} @ {sha}" if sha else target
    counts = {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER}
    head = f"ulpwise scan of {where}: {nfiles} files, {len(findings)} findings ({counts['high']} high, {counts['medium']} medium, {counts['info']} info)"
    out.append(f"# {head}" if markdown else head)
    out.append("")
    rule_ids = sorted({f.rule for f in findings}, key=lambda r: (SEVERITY_ORDER[RULES[r][0]], r))
    for rule in rule_ids:
        sev, what, fix, example = RULES[rule]
        group = [f for f in findings if f.rule == rule]
        title = f"{rule} [{sev}] x{len(group)}: {what}. Fix: {fix}." + (f" Example: {example}." if example else "")
        out.append(f"## {title}" if markdown else title)
        if markdown:
            out.append("")
        for f in group:
            loc = f"{f.path}:{f.line} {f.function}"
            out.append(f"- `{loc}` `{f.snippet}`" if markdown else f"  {loc}\n      {f.snippet}")
        out.append("")
    out.append("## Functions with the most elementary math, for reading" if markdown else "Functions with the most elementary math, for reading:")
    if markdown:
        out.append("")
        out.append("| calls | divisions | function | uses |")
        out.append("|---:|---:|---|---|")
    for s in list(spots)[:top]:
        loc = f"{s.path}:{s.line} {s.function}"
        if markdown:
            out.append(f"| {s.calls} | {s.divisions} | `{loc}` | {' '.join(s.names)} |")
        else:
            out.append(f"  {s.calls:>4} calls {s.divisions:>3} div  {loc}  [{' '.join(s.names)}]")
    return "\n".join(out) + "\n"


@dataclass
class RunResult:
    spot: HotSpot
    status: str  # "ran" or the reason it did not
    worst: Optional[int] = None  # elementwise float32 ulps against the float64 result, None for a NaN mismatch
    at_scale: Optional[float] = None  # largest absolute error in ulps of the largest output magnitude
    at: Optional[float] = None  # the input where the elementwise worst happened, when outputs map to inputs
    backend: str = ""


def _compare(out32: Sequence[float], ref32: Sequence[float]) -> Tuple[Optional[int], int, Optional[float]]:
    """Elementwise worst ulp distance with its index, and the largest absolute error measured in ulps
    of the largest finite reference value. The second number is the one to read for outputs that
    contain mathematically zero entries (rotation matrices, differences): there the elementwise
    count compares rounding noise with rounding noise and says nothing."""
    import ulpwise

    worst, i = ulpwise.max_ulp(out32, ref32, "f32")
    pairs = [(x, y) for x, y in zip(out32, ref32) if math.isfinite(x) and math.isfinite(y)]
    for x, y in zip(out32, ref32):
        if math.isfinite(x) != math.isfinite(y) or (not math.isfinite(x) and x != y and not (x != x and y != y)):
            return worst, i, None  # finite on one side only, or different infinities: no scale to speak of
    scale = max((abs(y) for _, y in pairs), default=0.0)
    err = max((abs(x - y) for x, y in pairs), default=0.0)
    if scale == 0.0:
        return worst, i, 0.0 if err == 0.0 else math.inf
    return worst, i, err / ulpwise.spacing(scale, "f32")


def _module_name(rel_path: str) -> str:
    parts = rel_path[:-3].split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    while len(parts) > 1 and parts[0] in ("src", "python", "lib"):
        parts = parts[1:]
    return ".".join(parts)


def _grid():
    """Both signs of a log grid from 1e-8 to 1e3 plus zero: small angles, moderate values and the
    start of overflow territory, 91 points in float64 as Python floats."""
    mags = [10.0 ** (-8 + 11 * i / 44) for i in range(45)]
    return [-m for m in reversed(mags)] + [0.0] + mags


def _call(fn, n_args: int, values: Sequence[float], backend: str):
    if backend == "torch":
        import torch

        x32 = torch.tensor(values, dtype=torch.float32)
        x64 = x32.to(torch.float64)
        out32, out64 = fn(*[x32] * n_args), fn(*[x64] * n_args)
        ok = torch.is_tensor(out32) and torch.is_tensor(out64) and out32.is_floating_point() and out32.shape == out64.shape
        if not ok:
            return None
        return out32.detach().reshape(-1).tolist(), out64.detach().to(torch.float32).reshape(-1).tolist()
    import numpy as np

    x32 = np.asarray(values, dtype=np.float32)
    x64 = x32.astype(np.float64)
    out32, out64 = fn(*[x32] * n_args), fn(*[x64] * n_args)
    out32, out64 = np.asarray(out32), np.asarray(out64)
    if out32.dtype != np.float32 or out32.shape != out64.shape or out64.dtype.kind != "f":
        return None
    return out32.reshape(-1).tolist(), out64.astype(np.float32).reshape(-1).tolist()


def _resolve(module, qualname: str):
    """Walk ``Class.method`` style names. Returns (callable, reason) with one of them None."""
    obj = module
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None, "nested function or not importable by name"
    if not callable(obj):
        return None, "not callable"
    if inspect.isclass(obj):
        return None, "a class"
    try:
        params = list(inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        return None, "no signature"
    if params and params[0].name in ("self", "cls") and "." in qualname:
        return None, "instance method"
    return obj, None


def run_functions(root: str, spots: Sequence[HotSpot], limit: int = 50, installed: bool = False) -> List[RunResult]:
    """Import the module level functions among ``spots``, call each with the same float32 and
    float64 grid for every required argument (torch first, then numpy) and measure the float32
    result against the float64 one rounded to float32, in float32 ulps.

    This imports and runs the repository's code, from the scanned tree unless ``installed`` asks
    for the environment's copy of the package (useful when the tree has unbuilt extensions).
    Static methods run; instance methods, anything that fails to import, needs other arguments
    or returns something that is not a float array is reported with the reason, not guessed at.
    """
    if not installed:
        for base in (root, os.path.join(root, "src"), os.path.join(root, "python")):
            if os.path.isdir(base) and base not in sys.path:
                sys.path.insert(0, base)
    values = _grid()
    results: List[RunResult] = []
    for spot in list(spots)[:limit]:
        try:
            module = importlib.import_module(_module_name(spot.path))
        except BaseException as e:  # noqa: BLE001, repository code can raise anything on import
            results.append(RunResult(spot, f"import failed: {type(e).__name__}"))
            continue
        fn, reason = _resolve(module, spot.function)
        if fn is None:
            results.append(RunResult(spot, reason))
            continue
        params = list(inspect.signature(fn).parameters.values())
        required = [
            p for p in params
            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if not required or len(required) > 2:
            results.append(RunResult(spot, f"needs {len(required)} positional arguments"))
            continue
        outcome, reasons = None, []
        for backend in ("torch", "numpy"):
            try:
                outcome = _call(fn, len(required), values, backend)
            except BaseException as e:  # noqa: BLE001
                reasons.append(f"call failed: {type(e).__name__}")
                continue
            if outcome is not None:
                break
            reasons.insert(0, "did not return a float array")
        if outcome is None:
            results.append(RunResult(spot, reasons[0]))
            continue
        out32, ref32 = outcome
        worst, i, at_scale = _compare(out32, ref32)
        at = values[i] if len(out32) == len(values) else None
        results.append(RunResult(spot, "ran", worst, at_scale, at, backend))
    return results


def render_run(results: Sequence[RunResult], markdown: bool = False) -> str:
    def key(r: RunResult):
        scale = -1.0 if r.at_scale is None else r.at_scale
        return (-scale, -(r.worst if r.worst is not None else 2**40))

    ran = sorted((r for r in results if r.status == "ran"), key=key)
    skipped = [r for r in results if r.status != "ran"]
    title = "float32 against float64, same function, log grid 1e-8 to 1e3 both signs and zero"
    out = [f"## {title}" if markdown else f"{title}:"]
    out.append(
        f"{len(results)} functions tried, {len(ran)} ran, {len(skipped)} skipped. "
        "'at scale' is the largest absolute error in ulps of the largest output, the number to read when the output "
        "has entries that should be zero; 'elementwise' is the worst per element ulp distance and its input."
    )
    if markdown:
        out.append("")
        out.append("| at scale | elementwise | at | function | backend |")
        out.append("|---:|---:|---:|---|---|")
    for r in ran:
        scale = "NaN/inf" if r.at_scale is None else f"{r.at_scale:.3g}"
        ulps = "NaN mismatch" if r.worst is None else str(r.worst)
        at = "" if r.at is None else f"{r.at:.3g}"
        loc = f"{r.spot.path}:{r.spot.line} {r.spot.function}"
        if markdown:
            out.append(f"| {scale} | {ulps} | {at} | `{loc}` | {r.backend} |")
        else:
            out.append(f"  {scale:>10} {ulps:>12} {at:>10}  {loc}  [{r.backend}]")
    if skipped:
        reasons = {}
        for r in skipped:
            reasons[r.status] = reasons.get(r.status, 0) + 1
        out.append("")
        out.append("skipped: " + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
    return "\n".join(out) + "\n"


def main(args) -> int:
    try:
        root, sha = checkout(args.target, args.workdir)
    except (ValueError, subprocess.CalledProcessError) as e:
        print(f"ulpwise scan: {e}", file=sys.stderr)
        return 2
    findings, spots, nfiles = scan_tree(root, args.include_tests)
    if args.rules:
        wanted = set(args.rules.split(","))
        unknown = wanted - set(RULES)
        if unknown:
            print(f"ulpwise scan: unknown rule(s) {sorted(unknown)}; known: {sorted(RULES)}", file=sys.stderr)
            return 2
        findings = [f for f in findings if f.rule in wanted]
    text = render(findings, spots, args.target, sha, nfiles, args.top, markdown=bool(args.report))
    if getattr(args, "run", False):
        results = run_functions(root, spots, args.run_limit, getattr(args, "run_installed", False))
        text += "\n" + render_run(results, markdown=bool(args.report))
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        counts = {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER}
        print(f"ulpwise scan: {len(findings)} findings ({counts['high']} high, {counts['medium']} medium, {counts['info']} info) in {nfiles} files, report in {args.report}")
    else:
        sys.stdout.write(text)
    if args.fail_on:
        worst = SEVERITY_ORDER[args.fail_on]
        if any(SEVERITY_ORDER[f.severity] <= worst for f in findings):
            return 1
    return 0
