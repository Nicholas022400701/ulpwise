//! ulpwise: numerical conformance testing for ML code.
//!
//! * [`ulp`]: ordered integer views of floats, neighbours, spacings and ulp distances.
//! * [`edge`]: the named edge values every numerical test should see, computed from the format.
//! * [`exact`]: exact rounding oracles for `sqrt` and division, correctly rounded results that
//!   do not depend on the platform libm, and `f64` referenced midpoint reports for `f32`
//!   transcendental functions.
//! * [`knife`]: scans for knife-edge inputs, where two implementations that differ by one ulp
//!   return different floats.

pub mod edge;
pub mod exact;
pub mod knife;
pub mod ulp;

#[cfg(feature = "python")]
mod py;

pub use exact::{
    div_midpoint, sqrt_cr_f32, sqrt_cr_f64, sqrt_midpoint, unary_midpoint_f32, MidpointReport,
    Unary,
};
