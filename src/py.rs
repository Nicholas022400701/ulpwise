use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::edge;
use crate::exact::{self, MidpointReport, Unary};
use crate::knife;
use crate::ulp;

fn is_f32(dtype: &str) -> PyResult<bool> {
    match dtype {
        "f32" | "float32" | "float" => Ok(true),
        "f64" | "float64" | "double" => Ok(false),
        _ => Err(PyValueError::new_err(format!(
            "unsupported dtype {dtype:?}: use 'f32' or 'f64'"
        ))),
    }
}

fn parse_op(op: &str) -> PyResult<Unary> {
    Unary::parse(op).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown op {op:?}: one of {:?} or 'div'",
            Unary::NAMES
        ))
    })
}

type Report = (f64, f64, bool, bool);
/// (x, rounded, distance_ulp, exact_above)
type Hit = (f64, f64, f64, bool);
type ScanResult = (Vec<Hit>, usize, bool);

fn tuple(r: MidpointReport) -> Report {
    (r.rounded, r.distance_ulp, r.exact_above, r.exact)
}

/// Number of representable floats between a and b, None if either is NaN.
#[pyfunction]
#[pyo3(signature = (a, b, dtype = "f64"))]
fn ulp_distance(a: f64, b: f64, dtype: &str) -> PyResult<Option<u64>> {
    Ok(if is_f32(dtype)? {
        ulp::ulp_distance_f32(a as f32, b as f32)
    } else {
        ulp::ulp_distance_f64(a, b)
    })
}

/// Elementwise ulp_distance over two equally long sequences.
#[pyfunction]
#[pyo3(signature = (a, b, dtype = "f64"))]
fn ulp_distances(a: Vec<f64>, b: Vec<f64>, dtype: &str) -> PyResult<Vec<Option<u64>>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let f32_ = is_f32(dtype)?;
    Ok(a.iter()
        .zip(&b)
        .map(|(&x, &y)| {
            if f32_ {
                ulp::ulp_distance_f32(x as f32, y as f32)
            } else {
                ulp::ulp_distance_f64(x, y)
            }
        })
        .collect())
}

/// Monotone integer view of x (-0.0 and +0.0 both map to 0).
#[pyfunction]
#[pyo3(signature = (x, dtype = "f64"))]
fn ordered(x: f64, dtype: &str) -> PyResult<i64> {
    Ok(if is_f32(dtype)? {
        ulp::ordered_f32(x as f32) as i64
    } else {
        ulp::ordered_f64(x)
    })
}

#[pyfunction]
#[pyo3(signature = (x, dtype = "f64"))]
fn next_up(x: f64, dtype: &str) -> PyResult<f64> {
    Ok(if is_f32(dtype)? {
        (x as f32).next_up() as f64
    } else {
        x.next_up()
    })
}

#[pyfunction]
#[pyo3(signature = (x, dtype = "f64"))]
fn next_down(x: f64, dtype: &str) -> PyResult<f64> {
    Ok(if is_f32(dtype)? {
        (x as f32).next_down() as f64
    } else {
        x.next_down()
    })
}

/// Spacing between |x| and the next larger float of the dtype.
#[pyfunction]
#[pyo3(signature = (x, dtype = "f64"))]
fn spacing(x: f64, dtype: &str) -> PyResult<f64> {
    Ok(if is_f32(dtype)? {
        ulp::spacing_f32(x as f32) as f64
    } else {
        ulp::spacing_f64(x)
    })
}

/// Named edge values of the dtype as (name, value) pairs.
#[pyfunction]
#[pyo3(signature = (dtype = "f32"))]
fn special(dtype: &str) -> PyResult<Vec<(String, f64)>> {
    Ok(if is_f32(dtype)? {
        edge::special_f32()
            .into_iter()
            .map(|(n, v)| (n.to_string(), v as f64))
            .collect()
    } else {
        edge::special_f64()
            .into_iter()
            .map(|(n, v)| (n.to_string(), v))
            .collect()
    })
}

/// The k floats below x, x, and the k floats above it (finite only, ascending).
#[pyfunction]
#[pyo3(signature = (x, k = 1, dtype = "f64"))]
fn neighbours(x: f64, k: u32, dtype: &str) -> PyResult<Vec<f64>> {
    Ok(if is_f32(dtype)? {
        edge::neighbours_f32(x as f32, k)
            .into_iter()
            .map(|v| v as f64)
            .collect()
    } else {
        edge::neighbours_f64(x, k)
    })
}

/// For each exponent e in [min_exp, max_exp]: the float just below 2**e and 2**e.
#[pyfunction]
#[pyo3(signature = (min_exp, max_exp, dtype = "f64"))]
fn binade_edges(min_exp: i32, max_exp: i32, dtype: &str) -> PyResult<Vec<f64>> {
    Ok(if is_f32(dtype)? {
        edge::binade_edges_f32(min_exp, max_exp)
            .into_iter()
            .map(|v| v as f64)
            .collect()
    } else {
        edge::binade_edges_f64(min_exp, max_exp)
    })
}

/// Every float of the dtype in [lo, hi], ascending. Errors if there are more than `limit`.
#[pyfunction]
#[pyo3(signature = (lo, hi, dtype = "f32", limit = 1_000_000))]
fn all_floats(lo: f64, hi: f64, dtype: &str, limit: usize) -> PyResult<Vec<f64>> {
    let out: Vec<f64> = if is_f32(dtype)? {
        edge::all_f32_in(lo as f32, hi as f32)
            .take(limit + 1)
            .map(|v| v as f64)
            .collect()
    } else {
        edge::all_f64_in(lo, hi).take(limit + 1).collect()
    };
    if out.len() > limit {
        return Err(PyValueError::new_err(format!(
            "more than {limit} floats in [{lo}, {hi}], raise limit or narrow the range"
        )));
    }
    Ok(out)
}

/// Where the exact result of `op(x)` (or `x / y` for op="div") sits relative to the rounding
/// midpoint: (rounded, distance_ulp, exact_above, exact). None when the result is not finite.
#[pyfunction]
#[pyo3(signature = (op, x, dtype = "f32", y = None))]
fn midpoint(op: &str, x: f64, dtype: &str, y: Option<f64>) -> PyResult<Option<Report>> {
    let f32_ = is_f32(dtype)?;
    if op == "div" {
        let y = y.ok_or_else(|| PyValueError::new_err("op='div' needs y"))?;
        let rep = if f32_ {
            exact::div_midpoint(x as f32, y as f32)
        } else {
            exact::div_midpoint(x, y)
        };
        return Ok(rep.map(tuple));
    }
    let u = parse_op(op)?;
    if f32_ {
        return Ok(exact::unary_midpoint_f32(u, x as f32).map(tuple));
    }
    match u {
        Unary::Sqrt => Ok(exact::sqrt_midpoint(x).map(tuple)),
        Unary::Recip => {
            if x.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                return Err(PyValueError::new_err("recip for f64 takes a positive x"));
            }
            Ok(exact::div_midpoint(1.0f64, x).map(tuple))
        }
        _ => Err(PyValueError::new_err(format!(
            "{op:?} has no exact f64 oracle yet, only sqrt, recip and div do; use dtype='f32'"
        ))),
    }
}

/// Up to `limit` inputs in [lo, hi] whose exact `op` result is within `tol_ulp` of a rounding
/// midpoint, as (x, rounded, distance_ulp, exact_above), plus the number of inputs evaluated and
/// whether the whole range was covered. Scans every `stride`-th float, at most `max_evals`.
#[pyfunction]
#[pyo3(signature = (op, lo, hi, tol_ulp = 1e-3, dtype = "f32", limit = 100, stride = 1, max_evals = 100_000_000))]
#[allow(clippy::too_many_arguments)]
fn knife_edges_raw(
    op: &str,
    lo: f64,
    hi: f64,
    tol_ulp: f64,
    dtype: &str,
    limit: usize,
    stride: usize,
    max_evals: usize,
) -> PyResult<ScanResult> {
    let u = parse_op(op)?;
    if is_f32(dtype)? {
        let s = knife::scan_unary_f32(u, lo as f32, hi as f32, tol_ulp, limit, stride, max_evals);
        let hits = s
            .hits
            .into_iter()
            .map(|(x, r)| (x as f64, r.rounded, r.distance_ulp, r.exact_above))
            .collect();
        Ok((hits, s.evaluated, s.exhausted))
    } else if u == Unary::Sqrt {
        let s = knife::scan_sqrt_f64(lo, hi, tol_ulp, limit, stride, max_evals);
        let hits = s
            .hits
            .into_iter()
            .map(|(x, r)| (x, r.rounded, r.distance_ulp, r.exact_above))
            .collect();
        Ok((hits, s.evaluated, s.exhausted))
    } else {
        Err(PyValueError::new_err(
            "f64 scans are available for op='sqrt' only",
        ))
    }
}

/// Correctly rounded square root of the dtype, independent of the platform sqrt.
#[pyfunction]
#[pyo3(signature = (x, dtype = "f64"))]
fn sqrt_cr(x: f64, dtype: &str) -> PyResult<f64> {
    Ok(if is_f32(dtype)? {
        exact::sqrt_cr_f32(x as f32) as f64
    } else {
        exact::sqrt_cr_f64(x)
    })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ulp_distance, m)?)?;
    m.add_function(wrap_pyfunction!(ulp_distances, m)?)?;
    m.add_function(wrap_pyfunction!(ordered, m)?)?;
    m.add_function(wrap_pyfunction!(next_up, m)?)?;
    m.add_function(wrap_pyfunction!(next_down, m)?)?;
    m.add_function(wrap_pyfunction!(spacing, m)?)?;
    m.add_function(wrap_pyfunction!(special, m)?)?;
    m.add_function(wrap_pyfunction!(neighbours, m)?)?;
    m.add_function(wrap_pyfunction!(binade_edges, m)?)?;
    m.add_function(wrap_pyfunction!(all_floats, m)?)?;
    m.add_function(wrap_pyfunction!(midpoint, m)?)?;
    m.add_function(wrap_pyfunction!(knife_edges_raw, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt_cr, m)?)?;
    m.add("UNARY_OPS", Unary::NAMES.to_vec())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
