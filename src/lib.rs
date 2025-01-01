#![allow(
    async_fn_in_trait,
    clippy::reversed_empty_ranges,
    non_snake_case,
    dead_code,
    clippy::new_without_default,
    clippy::op_ref
)]

use ndarray::{prelude::*, Data, DataMut};

pub mod advect;
pub mod bc;
pub mod body;
#[cfg(feature = "wgpu")]
pub mod gpu;
pub mod project;
pub mod system;

mod calc;

#[doc(inline)]
pub use crate::{
    advect::{Advect, Interpolate, LinearInterpolate, LinearTrace, Trace},
    bc::{BoundaryCondition, ContinuityBC, NoSlipBC},
    body::{BodyForce, ConstantBodyForce, InitialBodyForce},
    project::ProjectGaussSeidel,
    system::{System, SystemBuilder, SystemOp},
};

#[cfg(all(feature = "f64", feature = "wgpu"))]
compile_error!("Double precision and GPU support are mutually exclusive");

#[cfg(feature = "f64")]
pub type Float = f64;

#[cfg(not(feature = "f64"))]
pub type Float = f32;

// Marker traits to make handling generic functions over arrays easier.
pub trait DataF: Data<Elem = Float> {}
impl<T> DataF for T where T: Data<Elem = Float> {}
pub trait DataMutF: DataMut<Elem = Float> {}
impl<T> DataMutF for T where T: DataMut<Elem = Float> {}

pub type ArrayBaseF<S, D> = ArrayBase<S, D>;

pub type ArrayBase1F<S> = ArrayBaseF<S, Ix1>;
pub type ArrayBase2F<S> = ArrayBaseF<S, Ix2>;
pub type ArrayBase3F<S> = ArrayBaseF<S, Ix3>;

pub type Array1F = Array1<Float>;
pub type Array2F = Array2<Float>;
pub type Array3F = Array3<Float>;

#[cfg(test)]
mod tests {
    use std::any::type_name;

    use super::*;

    #[test]
    fn test_float_type() {
        #[cfg(feature = "f64")]
        assert_eq!(type_name::<Float>(), type_name::<f64>());

        #[cfg(not(feature = "f64"))]
        assert_eq!(type_name::<Float>(), type_name::<f32>());
    }
}
