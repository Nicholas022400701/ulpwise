//! Edge values: the inputs where floating point code goes wrong first.
//!
//! Everything here is computed from the format, not typed in by hand, so the lists stay right
//! for both widths and the thresholds (where `x * x` becomes zero, subnormal or infinite, where
//! `1 / x` overflows) are exact for the target dtype.

use crate::ulp::*;

/// Largest `x` in `[lo, hi]` with `pred(x) == true`, for a predicate that is true on a prefix.
fn largest_where_f32(pred: impl Fn(f32) -> bool, lo: f32, hi: f32) -> f32 {
    let (mut lo, mut hi) = (ordered_f32(lo), ordered_f32(hi));
    debug_assert!(pred(from_ordered_f32(lo)) && !pred(from_ordered_f32(hi)));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if pred(from_ordered_f32(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    from_ordered_f32(lo)
}

fn largest_where_f64(pred: impl Fn(f64) -> bool, lo: f64, hi: f64) -> f64 {
    let (mut lo, mut hi) = (ordered_f64(lo), ordered_f64(hi));
    debug_assert!(pred(from_ordered_f64(lo)) && !pred(from_ordered_f64(hi)));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if pred(from_ordered_f64(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    from_ordered_f64(lo)
}

/// Named `f32` values that a numerical test should see at least once.
pub fn special_f32() -> Vec<(&'static str, f32)> {
    let min_sub = f32::from_bits(1);
    let square_zero = largest_where_f32(|x| x * x == 0.0, min_sub, 1.0);
    let square_sub = largest_where_f32(|x| x * x < f32::MIN_POSITIVE, square_zero.next_up(), 1.0);
    let square_finite = largest_where_f32(|x| (x * x).is_finite(), 1.0, f32::MAX);
    let recip_inf = largest_where_f32(|x| (1.0 / x).is_infinite(), min_sub, 1.0);
    vec![
        ("zero", 0.0),
        ("neg_zero", -0.0),
        ("min_subnormal", min_sub),
        ("neg_min_subnormal", -min_sub),
        ("max_subnormal", f32::MIN_POSITIVE.next_down()),
        ("min_normal", f32::MIN_POSITIVE),
        ("neg_min_normal", -f32::MIN_POSITIVE),
        ("reciprocal_overflows", recip_inf),
        ("square_underflows_to_zero", square_zero),
        ("square_is_subnormal", square_sub),
        ("tenth", 0.1),
        ("third", 1.0 / 3.0),
        ("half", 0.5),
        ("one_minus_ulp", 1.0f32.next_down()),
        ("one", 1.0),
        ("one_plus_ulp", 1.0f32.next_up()),
        ("neg_one", -1.0),
        ("two", 2.0),
        ("e", core::f32::consts::E),
        ("pi", core::f32::consts::PI),
        ("integer_limit", 16_777_216.0),
        ("sqrt_max", f32::MAX.sqrt()),
        ("square_just_finite", square_finite),
        ("square_overflows", square_finite.next_up()),
        ("max", f32::MAX),
        ("neg_max", -f32::MAX),
        ("inf", f32::INFINITY),
        ("neg_inf", f32::NEG_INFINITY),
        ("nan", f32::NAN),
    ]
}

/// Named `f64` values that a numerical test should see at least once.
pub fn special_f64() -> Vec<(&'static str, f64)> {
    let min_sub = f64::from_bits(1);
    let square_zero = largest_where_f64(|x| x * x == 0.0, min_sub, 1.0);
    let square_sub = largest_where_f64(|x| x * x < f64::MIN_POSITIVE, square_zero.next_up(), 1.0);
    let square_finite = largest_where_f64(|x| (x * x).is_finite(), 1.0, f64::MAX);
    let recip_inf = largest_where_f64(|x| (1.0 / x).is_infinite(), min_sub, 1.0);
    vec![
        ("zero", 0.0),
        ("neg_zero", -0.0),
        ("min_subnormal", min_sub),
        ("neg_min_subnormal", -min_sub),
        ("max_subnormal", f64::MIN_POSITIVE.next_down()),
        ("min_normal", f64::MIN_POSITIVE),
        ("neg_min_normal", -f64::MIN_POSITIVE),
        ("reciprocal_overflows", recip_inf),
        ("square_underflows_to_zero", square_zero),
        ("square_is_subnormal", square_sub),
        ("tenth", 0.1),
        ("third", 1.0 / 3.0),
        ("half", 0.5),
        ("one_minus_ulp", 1.0f64.next_down()),
        ("one", 1.0),
        ("one_plus_ulp", 1.0f64.next_up()),
        ("neg_one", -1.0),
        ("two", 2.0),
        ("e", core::f64::consts::E),
        ("pi", core::f64::consts::PI),
        ("integer_limit", 9_007_199_254_740_992.0),
        ("sqrt_max", f64::MAX.sqrt()),
        ("square_just_finite", square_finite),
        ("square_overflows", square_finite.next_up()),
        ("max", f64::MAX),
        ("neg_max", -f64::MAX),
        ("inf", f64::INFINITY),
        ("neg_inf", f64::NEG_INFINITY),
        ("nan", f64::NAN),
    ]
}

/// The `k` floats below `x`, `x` itself and the `k` floats above it, finite ones only, ascending.
pub fn neighbours_f32(x: f32, k: u32) -> Vec<f32> {
    if !x.is_finite() {
        return vec![x];
    }
    let o = ordered_f32(x) as i64;
    ((o - k as i64)..=(o + k as i64))
        .filter_map(|v| i32::try_from(v).ok())
        .map(from_ordered_f32)
        .filter(|v| v.is_finite())
        .collect()
}

/// The `k` floats below `x`, `x` itself and the `k` floats above it, finite ones only, ascending.
pub fn neighbours_f64(x: f64, k: u32) -> Vec<f64> {
    if !x.is_finite() {
        return vec![x];
    }
    let o = ordered_f64(x) as i128;
    ((o - k as i128)..=(o + k as i128))
        .filter_map(|v| i64::try_from(v).ok())
        .map(from_ordered_f64)
        .filter(|v| v.is_finite())
        .collect()
}

/// For every exponent `e` in `min_exp..=max_exp`: the float just below `2^e` and `2^e` itself.
/// These are the points where the ulp doubles, so where rounding behaviour changes.
pub fn binade_edges_f32(min_exp: i32, max_exp: i32) -> Vec<f32> {
    let mut out = Vec::new();
    for e in min_exp.max(-126)..=max_exp.min(127) {
        let p = f32::from_bits(((e + 127) as u32) << 23);
        out.push(p.next_down());
        out.push(p);
    }
    out
}

/// For every exponent `e` in `min_exp..=max_exp`: the float just below `2^e` and `2^e` itself.
pub fn binade_edges_f64(min_exp: i32, max_exp: i32) -> Vec<f64> {
    let mut out = Vec::new();
    for e in min_exp.max(-1022)..=max_exp.min(1023) {
        let p = f64::from_bits(((e + 1023) as u64) << 52);
        out.push(p.next_down());
        out.push(p);
    }
    out
}

/// Every `f32` in `[lo, hi]` in ascending order (`-0.0` is skipped, NaN bounds give nothing).
pub fn all_f32_in(lo: f32, hi: f32) -> impl Iterator<Item = f32> {
    every_f32_in(lo, hi, 1)
}

/// Every `stride`-th `f32` in `[lo, hi]`, starting at `lo`. Stepping is done on the ordered
/// integer view, so large strides over large ranges cost nothing.
pub fn every_f32_in(lo: f32, hi: f32, stride: usize) -> impl Iterator<Item = f32> {
    let (a, b) = if lo.is_nan() || hi.is_nan() || lo > hi {
        (1, 0)
    } else {
        (ordered_f32(lo), ordered_f32(hi))
    };
    (a..=b).step_by(stride.max(1)).map(from_ordered_f32)
}

/// Every `f64` in `[lo, hi]` in ascending order (`-0.0` is skipped, NaN bounds give nothing).
pub fn all_f64_in(lo: f64, hi: f64) -> impl Iterator<Item = f64> {
    every_f64_in(lo, hi, 1)
}

/// Every `stride`-th `f64` in `[lo, hi]`, starting at `lo`.
pub fn every_f64_in(lo: f64, hi: f64, stride: usize) -> impl Iterator<Item = f64> {
    let (a, b) = if lo.is_nan() || hi.is_nan() || lo > hi {
        (1, 0)
    } else {
        (ordered_f64(lo), ordered_f64(hi))
    };
    (a..=b).step_by(stride.max(1)).map(from_ordered_f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get<T: Copy>(v: &[(&str, T)], name: &str) -> T {
        v.iter().find(|(n, _)| *n == name).map(|(_, x)| *x).unwrap()
    }

    #[test]
    fn f32_thresholds_are_exact() {
        let s = special_f32();
        let z = get(&s, "square_underflows_to_zero");
        assert_eq!(z * z, 0.0);
        assert!(z.next_up() * z.next_up() > 0.0);
        let sub = get(&s, "square_is_subnormal");
        assert!(sub * sub > 0.0 && sub * sub < f32::MIN_POSITIVE);
        assert!(sub.next_up() * sub.next_up() >= f32::MIN_POSITIVE);
        let fin = get(&s, "square_just_finite");
        assert!((fin * fin).is_finite());
        assert!((get(&s, "square_overflows") * get(&s, "square_overflows")).is_infinite());
        let r = get(&s, "reciprocal_overflows");
        assert!((1.0 / r).is_infinite() && (1.0 / r.next_up()).is_finite());
        assert_eq!(get(&s, "integer_limit") + 1.0, get(&s, "integer_limit"));
        let names: std::collections::HashSet<_> = s.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), s.len());
    }

    #[test]
    fn f64_thresholds_are_exact() {
        let s = special_f64();
        let z = get(&s, "square_underflows_to_zero");
        assert_eq!(z * z, 0.0);
        assert!(z.next_up() * z.next_up() > 0.0);
        let fin = get(&s, "square_just_finite");
        assert!((fin * fin).is_finite());
        assert!((get(&s, "square_overflows") * get(&s, "square_overflows")).is_infinite());
        assert_eq!(get(&s, "integer_limit") + 1.0, get(&s, "integer_limit"));
        assert_eq!(s.len(), special_f32().len());
    }

    #[test]
    fn neighbours_and_binades() {
        assert_eq!(
            neighbours_f32(1.0, 1),
            vec![1.0f32.next_down(), 1.0, 1.0f32.next_up()]
        );
        assert_eq!(
            neighbours_f64(f64::MAX, 1),
            vec![f64::MAX.next_down(), f64::MAX]
        );
        assert_eq!(neighbours_f32(f32::NAN, 3).len(), 1);
        let b = binade_edges_f32(0, 1);
        assert_eq!(b, vec![1.0f32.next_down(), 1.0, 2.0f32.next_down(), 2.0]);
        assert_eq!(binade_edges_f64(-1022, -1022)[1], f64::MIN_POSITIVE);
        assert_eq!(all_f32_in(1.0, 1.0f32.next_up().next_up()).count(), 3);
        assert_eq!(all_f32_in(-f32::from_bits(1), f32::from_bits(1)).count(), 3);
        assert_eq!(all_f64_in(2.0, 1.0).count(), 0);
        assert_eq!(every_f64_in(0.5, 1.0, 1 << 50).count(), 5);
        assert_eq!(
            every_f32_in(1.0, 2.0, 1 << 22).collect::<Vec<_>>(),
            vec![1.0, 1.5, 2.0]
        );
    }
}
