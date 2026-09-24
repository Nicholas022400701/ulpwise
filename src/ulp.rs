//! Units in the last place: ordered integer views, neighbours, spacings and distances.
//!
//! The integer view follows numpy's `assert_array_max_ulp`: `+0.0` and `-0.0` both map to 0,
//! consecutive finite floats map to consecutive integers, and the largest finite value is one
//! step away from infinity. NaN is never one step away from anything, so distances involving a
//! NaN are reported as `None`.

/// Monotone integer view of an `f64`. `-0.0` and `+0.0` both map to 0.
#[inline]
pub fn ordered_f64(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    if b < 0 {
        i64::MIN - b
    } else {
        b
    }
}

/// Monotone integer view of an `f32`. `-0.0` and `+0.0` both map to 0.
#[inline]
pub fn ordered_f32(x: f32) -> i32 {
    let b = x.to_bits() as i32;
    if b < 0 {
        i32::MIN - b
    } else {
        b
    }
}

/// Inverse of [`ordered_f64`]. 0 maps back to `+0.0`.
#[inline]
pub fn from_ordered_f64(o: i64) -> f64 {
    let b = if o < 0 {
        (i64::MIN - o) as u64
    } else {
        o as u64
    };
    f64::from_bits(b)
}

/// Inverse of [`ordered_f32`]. 0 maps back to `+0.0`.
#[inline]
pub fn from_ordered_f32(o: i32) -> f32 {
    let b = if o < 0 {
        (i32::MIN - o) as u32
    } else {
        o as u32
    };
    f32::from_bits(b)
}

/// Number of representable `f64` values between `a` and `b`. `None` if either is NaN.
#[inline]
pub fn ulp_distance_f64(a: f64, b: f64) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    Some((ordered_f64(a) as i128 - ordered_f64(b) as i128).unsigned_abs() as u64)
}

/// Number of representable `f32` values between `a` and `b`. `None` if either is NaN.
#[inline]
pub fn ulp_distance_f32(a: f32, b: f32) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    Some((ordered_f32(a) as i64 - ordered_f32(b) as i64).unsigned_abs())
}

/// Spacing between `|x|` and the next larger `f64` (numpy's `spacing`). NaN for non-finite input.
#[inline]
pub fn spacing_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let a = x.abs();
    if a == f64::MAX {
        a - a.next_down()
    } else {
        a.next_up() - a
    }
}

/// Spacing between `|x|` and the next larger `f32` (numpy's `spacing`). NaN for non-finite input.
#[inline]
pub fn spacing_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let a = x.abs();
    if a == f32::MAX {
        a - a.next_down()
    } else {
        a.next_up() - a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_is_monotone_and_invertible() {
        let xs = [
            -f64::MAX,
            -1.0,
            -f64::MIN_POSITIVE,
            -f64::from_bits(1),
            0.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            1.0,
            f64::MAX,
            f64::INFINITY,
        ];
        for w in xs.windows(2) {
            assert!(ordered_f64(w[0]) < ordered_f64(w[1]), "{:?}", w);
        }
        for &x in &xs {
            assert_eq!(from_ordered_f64(ordered_f64(x)).to_bits(), x.to_bits());
        }
        assert_eq!(ordered_f64(-0.0), 0);
        assert_eq!(ordered_f32(-0.0), 0);
        assert_eq!(from_ordered_f32(ordered_f32(-1.5f32)), -1.5f32);
    }

    #[test]
    fn distances() {
        assert_eq!(ulp_distance_f64(1.0, 1.0f64.next_up()), Some(1));
        assert_eq!(ulp_distance_f32(1.0, 1.0f32.next_down()), Some(1));
        assert_eq!(ulp_distance_f32(-0.0, 0.0), Some(0));
        assert_eq!(
            ulp_distance_f32(-f32::from_bits(1), f32::from_bits(1)),
            Some(2)
        );
        assert_eq!(ulp_distance_f64(f64::MAX, f64::INFINITY), Some(1));
        assert_eq!(ulp_distance_f64(f64::NAN, 1.0), None);
        assert_eq!(spacing_f32(1.0), f32::EPSILON);
        assert_eq!(spacing_f64(-2.0), 2.0 * f64::EPSILON);
        assert!(spacing_f32(f32::INFINITY).is_nan());
    }
}
