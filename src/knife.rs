//! Knife-edge scans: inputs whose exact result sits so close to a rounding midpoint that two
//! implementations differing by one ulp return different floats.

use crate::edge::{every_f32_in, every_f64_in};
use crate::exact::{sqrt_midpoint, unary_midpoint_f32, MidpointReport, Unary};

/// Result of a scan: the hits, how many inputs were evaluated and whether the range was
/// covered (false when `max_evals` or `limit` stopped the scan first).
#[derive(Clone, Debug, PartialEq)]
pub struct Scan<T> {
    pub hits: Vec<(T, MidpointReport)>,
    pub evaluated: usize,
    pub exhausted: bool,
}

fn scan<T: Copy>(
    inputs: impl Iterator<Item = T>,
    report: impl Fn(T) -> Option<MidpointReport>,
    tol_ulp: f64,
    limit: usize,
    max_evals: usize,
) -> Scan<T> {
    let mut hits = Vec::new();
    let mut evaluated = 0usize;
    let mut exhausted = true;
    for x in inputs {
        if evaluated >= max_evals || hits.len() >= limit {
            exhausted = false;
            break;
        }
        evaluated += 1;
        if let Some(rep) = report(x) {
            if !rep.exact && rep.distance_ulp < tol_ulp {
                hits.push((x, rep));
            }
        }
    }
    Scan {
        hits,
        evaluated,
        exhausted,
    }
}

/// Scan every `stride`-th `f32` in `[lo, hi]` (at most `max_evals` of them) for up to `limit`
/// inputs whose exact result is within `tol_ulp` of a rounding midpoint.
pub fn scan_unary_f32(
    op: Unary,
    lo: f32,
    hi: f32,
    tol_ulp: f64,
    limit: usize,
    stride: usize,
    max_evals: usize,
) -> Scan<f32> {
    scan(
        every_f32_in(lo, hi, stride),
        |x| unary_midpoint_f32(op, x),
        tol_ulp,
        limit,
        max_evals,
    )
}

/// Same as [`scan_unary_f32`] for the exact `f64` square root.
pub fn scan_sqrt_f64(
    lo: f64,
    hi: f64,
    tol_ulp: f64,
    limit: usize,
    stride: usize,
    max_evals: usize,
) -> Scan<f64> {
    scan(
        every_f64_in(lo, hi, stride),
        sqrt_midpoint,
        tol_ulp,
        limit,
        max_evals,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_kornia_value() {
        let scan = scan_unary_f32(Unary::Sqrt, 0.85, 0.86, 1e-3, 10_000, 1, usize::MAX);
        assert!(scan.exhausted);
        let hits = scan.hits;
        assert!(hits.iter().any(|(x, _)| *x == 0.8528626561164856f32));
        assert!(hits.iter().all(|(_, r)| r.distance_ulp < 1e-3));
        // Midpoint distances are close to uniform on (0, 0.5], so about 2 * tol of the inputs hit.
        let n = every_f32_in(0.85, 0.86, 1).count() as f64;
        assert!(
            (hits.len() as f64) > 0.5 * 2e-3 * n && (hits.len() as f64) < 2.0 * 2e-3 * n,
            "{}",
            hits.len()
        );
    }

    #[test]
    fn f64_scan_respects_limits() {
        let scan = scan_sqrt_f64(2.0, 3.0, 1e-2, 5, 1_000_003, 1_000_000);
        assert!(scan.hits.len() <= 5 && !scan.exhausted && scan.evaluated <= 1_000_000);
        // A huge stride over the whole binade must be cheap: about 2^52 / 2^40 = 4096 evaluations.
        let sparse = scan_sqrt_f64(0.5, 1.0, 0.5, usize::MAX, 1 << 40, usize::MAX);
        assert!(
            sparse.exhausted && sparse.evaluated == 4097,
            "{}",
            sparse.evaluated
        );
        assert!(scan.hits.iter().all(|(_, r)| r.distance_ulp < 1e-2));
        let full = scan_sqrt_f64(2.0, 2.0f64.next_up().next_up(), 1.0, 100, 1, 100);
        assert!(full.exhausted && full.evaluated == 3);
    }
}
