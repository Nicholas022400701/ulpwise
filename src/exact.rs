//! Exact rounding oracles.
//!
//! For `sqrt` and division the exact result is compared against candidate floats with integer
//! arithmetic on significands (`u128`), so the reported rounding direction and the distance to the
//! rounding midpoint do not depend on any libm. The correctly rounded result is found by starting
//! from the hardware result and stepping toward the exact value until the midpoint test passes,
//! so [`sqrt_cr_f64`] and friends are correct even on a platform whose `sqrt` is not.
//!
//! For transcendental `f32` functions the reference is the `f64` libm, which is enough to place an
//! `f32` result relative to its rounding midpoint down to about `1e-8` ulp.

use core::cmp::Ordering;

/// Where the exact result of an operation sits relative to the floats around it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MidpointReport {
    /// The correctly rounded result (as `f64`; exact for `f32` results too).
    pub rounded: f64,
    /// The exact result is representable: `distance_ulp` is then reported as 0.5.
    pub exact: bool,
    /// The exact result lies above `rounded`. An implementation that errs upward returns
    /// `next_up(rounded)`, one that errs downward returns `next_down(rounded)`.
    pub exact_above: bool,
    /// Distance from the exact result to the rounding midpoint on that side, in units of the
    /// spacing on that side. 0.5 means the exact result equals `rounded`; values near 0 are the
    /// knife edges where a 1 ulp libm error flips the rounded result.
    pub distance_ulp: f64,
}

impl MidpointReport {
    fn exact(rounded: f64) -> Self {
        MidpointReport {
            rounded,
            exact: true,
            exact_above: false,
            distance_ulp: 0.5,
        }
    }
}

/// Format parameters of a float type.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct Fmt {
    /// Significand width including the implicit bit.
    pub p: u32,
    /// Exponent of the subnormal quantum.
    pub emin: i32,
}

/// A float format the exact machinery knows how to take apart.
pub trait Float: Copy + PartialOrd {
    #[doc(hidden)]
    const FMT: Fmt;
    /// `(m, e)` with value `m * 2^e`, the implicit bit included, zero giving `(0, emin)`.
    fn decompose(self) -> (u128, i32);
    fn next_up_(self) -> Self;
    fn next_down_(self) -> Self;
    fn as_f64(self) -> f64;
    fn is_finite_(self) -> bool;
    fn zero() -> Self;
    fn sqrt_(self) -> Self;
    fn div_(self, other: Self) -> Self;
}

impl Float for f64 {
    const FMT: Fmt = Fmt { p: 53, emin: -1074 };
    fn decompose(self) -> (u128, i32) {
        let bits = self.to_bits();
        let exp = ((bits >> 52) & 0x7FF) as i32;
        let frac = bits & ((1u64 << 52) - 1);
        if exp == 0 {
            (frac as u128, -1074)
        } else {
            ((frac | (1u64 << 52)) as u128, exp - 1075)
        }
    }
    fn next_up_(self) -> Self {
        self.next_up()
    }
    fn next_down_(self) -> Self {
        self.next_down()
    }
    fn as_f64(self) -> f64 {
        self
    }
    fn is_finite_(self) -> bool {
        self.is_finite()
    }
    fn zero() -> Self {
        0.0
    }
    fn sqrt_(self) -> Self {
        self.sqrt()
    }
    fn div_(self, other: Self) -> Self {
        self / other
    }
}

impl Float for f32 {
    const FMT: Fmt = Fmt { p: 24, emin: -149 };
    fn decompose(self) -> (u128, i32) {
        let bits = self.to_bits();
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7F_FFFF;
        if exp == 0 {
            (frac as u128, -149)
        } else {
            ((frac | (1u32 << 23)) as u128, exp - 150)
        }
    }
    fn next_up_(self) -> Self {
        self.next_up()
    }
    fn next_down_(self) -> Self {
        self.next_down()
    }
    fn as_f64(self) -> f64 {
        self as f64
    }
    fn is_finite_(self) -> bool {
        self.is_finite()
    }
    fn zero() -> Self {
        0.0
    }
    fn sqrt_(self) -> Self {
        // 53 >= 2 * 24 + 2, so rounding the f64 square root to f32 is the correctly rounded f32
        // square root (double rounding is innocuous for + - * / sqrt at that width ratio).
        (self as f64).sqrt() as f32
    }
    fn div_(self, other: Self) -> Self {
        self / other
    }
}

/// Shift left by a signed amount, shifting right when negative. Panics if bits would be lost,
/// which cannot happen while the candidate is within one ulp of the exact result.
fn scale(v: u128, s: i32) -> u128 {
    if s >= 0 {
        let r = v.checked_shl(s as u32).unwrap_or(0);
        assert!(s < 128 && (r >> s) == v, "exact residual overflowed u128");
        r
    } else {
        let r = v >> (-s).min(127);
        assert!((r << (-s).min(127)) == v, "exact residual lost bits");
        r
    }
}

/// Compare `(mm * 2^em)^2` with `x = mx * 2^ex`. Returns the ordering of the square against `x`,
/// `|x - square|` and the exponent of that difference.
fn residual_sq(mx: u128, ex: i32, mm: u128, em: i32) -> (Ordering, u128, i32) {
    let m2 = mm * mm;
    let e2 = 2 * em;
    let s = ex.min(e2);
    let a = scale(mx, ex - s);
    let b = scale(m2, e2 - s);
    (b.cmp(&a), a.abs_diff(b), s)
}

/// Compare `(mm * 2^em) * (mb * 2^eb)` with `a = ma * 2^ea`.
fn residual_mul(ma: u128, ea: i32, mm: u128, em: i32, mb: u128, eb: i32) -> (Ordering, u128, i32) {
    let prod = mm * mb;
    let ep = em + eb;
    let s = ea.min(ep);
    let a = scale(ma, ea - s);
    let b = scale(prod, ep - s);
    (b.cmp(&a), a.abs_diff(b), s)
}

/// The rounding midpoint next to `r = mr * 2^er` on the given side, as `(M, eM)`, and the
/// exponent of the spacing on that side.
fn midpoint(mr: u128, er: i32, f: Fmt, above: bool) -> (u128, i32, i32) {
    if above {
        (2 * mr + 1, er - 1, er)
    } else if mr == (1u128 << (f.p - 1)) && er > f.emin {
        // Bottom of a binade: the spacing below is half the spacing above.
        (4 * mr - 1, er - 2, er - 1)
    } else {
        (2 * mr - 1, er - 1, er)
    }
}

/// `diff * 2^s / (den * 2^eden)` as f64.
fn ratio(diff: u128, s: i32, den: u128, eden: i32) -> f64 {
    let t = eden - s;
    if t >= 0 {
        diff as f64 / scale(den, t) as f64
    } else {
        scale(diff, -t) as f64 / den as f64
    }
}

/// Correctly rounded square root with an exact midpoint report. `None` for negative, infinite or
/// NaN input.
pub fn sqrt_midpoint<T: Float>(x: T) -> Option<MidpointReport> {
    if !x.is_finite_() || x < T::zero() {
        return None;
    }
    if x == T::zero() {
        return Some(MidpointReport::exact(x.sqrt_().as_f64()));
    }
    let (mx, ex) = x.decompose();
    let mut r = x.sqrt_();
    // Step toward the exact value until r is the correctly rounded result.
    for _ in 0..8 {
        let (mr, er) = r.decompose();
        let (ord, _, _) = residual_sq(mx, ex, mr, er);
        if ord == Ordering::Equal {
            return Some(MidpointReport::exact(r.as_f64()));
        }
        let above = ord == Ordering::Less;
        let (mm, em, eu) = midpoint(mr, er, T::FMT, above);
        let (mord, diff, s) = residual_sq(mx, ex, mm, em);
        // The exact result must lie between r and the midpoint: m^2 > x when above, m^2 < x below.
        let past_midpoint = if above {
            mord != Ordering::Greater
        } else {
            mord != Ordering::Less
        };
        if past_midpoint {
            r = if above { r.next_up_() } else { r.next_down_() };
            continue;
        }
        // |sqrt(x) - m| = |x - m^2| / (sqrt(x) + m). The integer ratio below uses 2 m for the
        // denominator; the f64 factor 2 m / (sqrt(x) + m) corrects that to within 2^-52.
        let approx = ratio(diff, s, mm, em + eu + 1);
        let m_f64 = (mm as f64) * 2f64.powi(em);
        let distance_ulp = approx * 2.0 / (1.0 + x.as_f64().sqrt() / m_f64);
        return Some(MidpointReport {
            rounded: r.as_f64(),
            exact: false,
            exact_above: above,
            distance_ulp,
        });
    }
    None
}

/// Correctly rounded quotient `a / b` with an exact midpoint report, for finite positive `a`, `b`
/// whose quotient is finite. `None` otherwise.
pub fn div_midpoint<T: Float>(a: T, b: T) -> Option<MidpointReport> {
    let positive = |v: T| v.is_finite_() && v.partial_cmp(&T::zero()) == Some(Ordering::Greater);
    if !positive(a) || !positive(b) {
        return None;
    }
    let (ma, ea) = a.decompose();
    let (mb, eb) = b.decompose();
    let mut r = a.div_(b);
    if !r.is_finite_() {
        return None;
    }
    for _ in 0..8 {
        let (mr, er) = r.decompose();
        let (ord, _, _) = residual_mul(ma, ea, mr, er, mb, eb);
        if ord == Ordering::Equal {
            return Some(MidpointReport::exact(r.as_f64()));
        }
        let above = ord == Ordering::Less;
        let (mm, em, eu) = midpoint(mr, er, T::FMT, above);
        let (mord, diff, s) = residual_mul(ma, ea, mm, em, mb, eb);
        let past_midpoint = if above {
            mord != Ordering::Greater
        } else {
            mord != Ordering::Less
        };
        if past_midpoint {
            r = if above { r.next_up_() } else { r.next_down_() };
            if !r.is_finite_() {
                return None;
            }
            continue;
        }
        // |a/b - m| = |a - m b| / b, exactly.
        let distance_ulp = ratio(diff, s, mb, eb + eu);
        return Some(MidpointReport {
            rounded: r.as_f64(),
            exact: false,
            exact_above: above,
            distance_ulp,
        });
    }
    None
}

/// Correctly rounded `f64` square root, independent of the platform `sqrt`.
pub fn sqrt_cr_f64(x: f64) -> f64 {
    match sqrt_midpoint(x) {
        Some(r) => r.rounded,
        None => x.sqrt(),
    }
}

/// Correctly rounded `f32` square root, independent of the platform `sqrt`.
pub fn sqrt_cr_f32(x: f32) -> f32 {
    match sqrt_midpoint(x) {
        Some(r) => r.rounded as f32,
        None => x.sqrt(),
    }
}

/// Unary functions with a knife-edge oracle for `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
    Sqrt,
    Rsqrt,
    Recip,
    Exp,
    Exp2,
    Expm1,
    Ln,
    Log2,
    Log10,
    Ln1p,
    Sin,
    Cos,
    Tan,
    Atan,
    Tanh,
    Sigmoid,
    Softplus,
}

impl Unary {
    pub const NAMES: [&'static str; 17] = [
        "sqrt", "rsqrt", "recip", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sin",
        "cos", "tan", "atan", "tanh", "sigmoid", "softplus",
    ];

    pub fn parse(name: &str) -> Option<Unary> {
        Some(match name {
            "sqrt" => Unary::Sqrt,
            "rsqrt" => Unary::Rsqrt,
            "recip" | "reciprocal" => Unary::Recip,
            "exp" => Unary::Exp,
            "exp2" => Unary::Exp2,
            "expm1" => Unary::Expm1,
            "log" | "ln" => Unary::Ln,
            "log2" => Unary::Log2,
            "log10" => Unary::Log10,
            "log1p" => Unary::Ln1p,
            "sin" => Unary::Sin,
            "cos" => Unary::Cos,
            "tan" => Unary::Tan,
            "atan" => Unary::Atan,
            "tanh" => Unary::Tanh,
            "sigmoid" => Unary::Sigmoid,
            "softplus" => Unary::Softplus,
            _ => return None,
        })
    }

    /// Whether the oracle for this function is exact (integer arithmetic) rather than f64 libm.
    pub fn is_exact(self) -> bool {
        matches!(self, Unary::Sqrt | Unary::Recip)
    }

    /// The function evaluated in `f64`.
    pub fn eval_f64(self, x: f64) -> f64 {
        match self {
            Unary::Sqrt => x.sqrt(),
            Unary::Rsqrt => 1.0 / x.sqrt(),
            Unary::Recip => 1.0 / x,
            Unary::Exp => x.exp(),
            Unary::Exp2 => x.exp2(),
            Unary::Expm1 => x.exp_m1(),
            Unary::Ln => x.ln(),
            Unary::Log2 => x.log2(),
            Unary::Log10 => x.log10(),
            Unary::Ln1p => x.ln_1p(),
            Unary::Sin => x.sin(),
            Unary::Cos => x.cos(),
            Unary::Tan => x.tan(),
            Unary::Atan => x.atan(),
            Unary::Tanh => x.tanh(),
            Unary::Sigmoid => {
                if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                }
            }
            Unary::Softplus => {
                if x > 0.0 {
                    x + (-x).exp().ln_1p()
                } else {
                    x.exp().ln_1p()
                }
            }
        }
    }
}

/// Midpoint report for an `f32` unary function. `sqrt` and `recip` use the exact integer oracle,
/// the rest use the `f64` libm as reference (trust `distance_ulp` down to about `1e-8`).
pub fn unary_midpoint_f32(op: Unary, x: f32) -> Option<MidpointReport> {
    match op {
        Unary::Sqrt => return sqrt_midpoint(x),
        Unary::Recip => {
            if !(x != 0.0) {
                return None;
            }
            let rep = div_midpoint(1.0f32, x.abs())?;
            return Some(MidpointReport {
                rounded: rep.rounded.copysign(x as f64),
                ..rep
            });
        }
        _ => {}
    }
    let y = op.eval_f64(x as f64);
    if !y.is_finite() {
        return None;
    }
    let r = y as f32;
    if !r.is_finite() {
        return None;
    }
    let rd = r as f64;
    if rd == y {
        return Some(MidpointReport::exact(rd));
    }
    let above = y > rd;
    let (up, down) = (r.next_up(), r.next_down());
    let u = if above && up.is_finite() {
        up as f64 - rd
    } else if !above && down.is_finite() {
        rd - down as f64
    } else {
        // At +-MAX the spacing beyond is the spacing inside the last binade.
        (r.abs() as f64) - (r.abs().next_down() as f64)
    };
    let m = if above { rd + u / 2.0 } else { rd - u / 2.0 };
    Some(MidpointReport {
        rounded: rd,
        exact: false,
        exact_above: above,
        distance_ulp: (y - m).abs() / u,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_squares_are_exact() {
        for x in [
            1.0f64,
            4.0,
            0.25,
            2.25,
            1e10 * 1e10,
            f64::MIN_POSITIVE * 4.0,
        ] {
            let r = sqrt_midpoint(x).unwrap();
            assert!(r.exact, "{x}");
            assert_eq!(r.rounded, x.sqrt());
        }
        let r = sqrt_midpoint(0.5625f32).unwrap();
        assert!(r.exact && r.rounded == 0.75);
    }

    #[test]
    fn the_kornia_knife_edge() {
        // sqrt(0.8528626561164856f32) sits 0.0004 ulp below the midpoint: exact rounding gives
        // 0.9235056, a sqrt that errs by one ulp upward gives 0.92350566.
        let x = 0.8528626561164856f32;
        let r = sqrt_midpoint(x).unwrap();
        assert!(!r.exact && r.exact_above, "{r:?}");
        assert!(r.distance_ulp < 5e-4 && r.distance_ulp > 3e-4, "{r:?}");
        assert_eq!(r.rounded as f32, 0.9235056f32);
        assert_eq!(sqrt_cr_f32(x), 0.9235056f32);
    }

    #[test]
    fn distances_are_within_half_ulp_and_agree_with_hardware() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = f64::from_bits((state >> 12) | 0x3FF0_0000_0000_0000)
                * 2f64.powi((state % 200) as i32 - 100);
            let r = sqrt_midpoint(x).unwrap();
            assert!(r.distance_ulp > 0.0 && r.distance_ulp <= 0.5, "{x} {r:?}");
            assert_eq!(r.rounded, x.sqrt(), "hardware sqrt is correctly rounded");
            let xf = x as f32;
            if xf.is_finite() && xf > 0.0 {
                let rf = sqrt_midpoint(xf).unwrap();
                assert_eq!(rf.rounded as f32, xf.sqrt(), "{xf}");
                assert_eq!(sqrt_cr_f32(xf), xf.sqrt());
            }
        }
    }

    #[test]
    fn subnormal_and_extreme_sqrt() {
        for x in [
            f64::from_bits(1),
            f64::from_bits(3),
            f64::MIN_POSITIVE.next_down(),
            f64::MAX,
            f64::MAX.next_down(),
        ] {
            let r = sqrt_midpoint(x).unwrap();
            assert_eq!(r.rounded, x.sqrt());
            assert!(r.distance_ulp <= 0.5);
        }
        assert_eq!(
            sqrt_midpoint(f32::from_bits(1)).unwrap().rounded as f32,
            f32::from_bits(1).sqrt()
        );
        assert!(sqrt_midpoint(-1.0f64).is_none());
        assert!(sqrt_midpoint(f64::INFINITY).is_none());
        assert!(sqrt_midpoint(0.0f64).unwrap().exact);
    }

    #[test]
    fn division() {
        let r = div_midpoint(1.0f64, 3.0).unwrap();
        assert!(!r.exact && r.rounded == 1.0 / 3.0);
        // 1/3 = 0x1.5555...p-2: the rounded value ends in ...5555 (down), the tail 0x5555.. is a third
        // of the way to the next float, so the distance to the midpoint is 1/2 - 1/3 = 1/6.
        assert!((r.distance_ulp - 1.0 / 6.0).abs() < 1e-12, "{r:?}");
        assert!(r.exact_above);
        assert!(div_midpoint(1.0f64, 4.0).unwrap().exact);
        assert!(div_midpoint(1.0f64, 0.0).is_none());
        assert!(div_midpoint(f64::MAX, 0.5).is_none());
        // Underflow to zero: 1 / MAX lies above 0.
        let r = div_midpoint(f64::from_bits(1), 4.0).unwrap();
        assert_eq!(r.rounded, 0.0);
        assert!(r.exact_above);
        let r = div_midpoint(2.0f32, 3.0).unwrap();
        assert_eq!(r.rounded as f32, 2.0f32 / 3.0);
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let a = f64::from_bits((state >> 12) | 0x3FF0_0000_0000_0000);
            let b = f64::from_bits((state.rotate_left(29) >> 12) | 0x3FF0_0000_0000_0000);
            let r = div_midpoint(a, b).unwrap();
            assert_eq!(r.rounded, a / b);
            assert!(r.distance_ulp <= 0.5);
            let (af, bf) = (a as f32, b as f32);
            assert_eq!(div_midpoint(af, bf).unwrap().rounded as f32, af / bf);
        }
    }

    #[test]
    fn unary_reference() {
        let r = unary_midpoint_f32(Unary::Exp, 1.0).unwrap();
        assert_eq!(r.rounded as f32, core::f64::consts::E as f32);
        assert!(r.distance_ulp <= 0.5);
        assert!(unary_midpoint_f32(Unary::Ln, -1.0).is_none());
        assert!(unary_midpoint_f32(Unary::Exp, 100.0).is_none());
        assert!(unary_midpoint_f32(Unary::Exp2, 3.0).unwrap().exact);
        let r = unary_midpoint_f32(Unary::Recip, -3.0).unwrap();
        assert_eq!(r.rounded as f32, -1.0f32 / 3.0);
        assert_eq!(Unary::parse("log1p"), Some(Unary::Ln1p));
        for n in Unary::NAMES {
            assert!(Unary::parse(n).is_some(), "{n}");
        }
    }
}
