#![allow(
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

#[cfg(feature = "f64")]
pub type Float = f64;

#[cfg(not(feature = "f64"))]
pub type Float = f64;

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
