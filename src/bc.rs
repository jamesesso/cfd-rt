//! Boundary conditions.

use std::{fmt::Debug, marker::PhantomData};

use ndarray::prelude::*;

use crate::{ArrayBase2F, ArrayBase3F, DataMutF, Float};

/// Trait representing an generic boundary condition.
///
/// Note that an [ArrayViewMut] has to be used here, due to
pub trait BoundaryCondition<D: Dimension>: Debug {
    /// Applies the boundary condition to an array.
    fn apply(&self, a: &mut ArrayViewMut<Float, D>);
    /// Name for the boundary condition.
    fn name(&self) -> &'static str;
}

/// [BoundaryCondition] for continuity.
///
/// This means that a_0i = a_1j and a_i0 = a_j1.
#[derive(Debug, Clone)]
pub struct ContinuityBC<D> {
    _marker: PhantomData<D>,
}

impl<D: Dimension> ContinuityBC<D> {
    pub fn new() -> Self {
        ContinuityBC {
            _marker: PhantomData,
        }
    }
}

impl ContinuityBC<Ix2> {
    /// Deals with the corners which need special handling to average neighbour
    /// horizontally and vertically.
    fn fix_corners<S: DataMutF>(&self, a: &mut ArrayBase2F<S>) {
        let axx = 0.5 * (&a.slice(s![0, 1]) + &a.slice(s![1, 0]));
        a.slice_mut(s![0, 0]).assign(&axx);

        let axx = 0.5 * (&a.slice(s![0, -2]) + &a.slice(s![1, -1]));
        a.slice_mut(s![0, -1]).assign(&axx);

        let axx = 0.5 * (&a.slice(s![-2, 0]) + &a.slice(s![-1, 1]));
        a.slice_mut(s![-1, 0]).assign(&axx);

        let axx = 0.5 * (&a.slice(s![-1, -2]) + &a.slice(s![-2, -1]));
        a.slice_mut(s![-1, -1]).assign(&axx);
    }
}

impl BoundaryCondition<Ix2> for ContinuityBC<Ix2> {
    fn apply(&self, a: &mut ArrayViewMut<Float, Ix2>) {
        let s = a.slice(s![1, ..]).to_owned();
        a.slice_mut(s![0, ..]).assign(&s);

        let s = a.slice(s![-2, ..]).to_owned();
        a.slice_mut(s![-1, ..]).assign(&s);

        let s = a.slice(s![.., 1]).to_owned();
        a.slice_mut(s![.., 0]).assign(&s);

        let s = a.slice(s![.., -2]).to_owned();
        a.slice_mut(s![.., -1]).assign(&s);

        self.fix_corners(a);
    }

    fn name(&self) -> &'static str {
        "continuity_bc"
    }
}

/// No slip [BoundaryCondition].
///
/// At the grid edges u = 0. As the wall lies at the midpoint between two grid
/// points, we set a_0i = -a_1j so the interpolated value is 0.
#[derive(Debug, Clone)]
pub struct NoSlipBC<D> {
    _marker: PhantomData<D>,
}

impl NoSlipBC<Ix3> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
    /// Deals with the corners which need special handling to average neighbour
    /// horizontally and vertically.
    fn fix_corners<S: DataMutF>(&self, a: &mut ArrayBase3F<S>) {
        let axx = 0.5 * (&a.slice(s![0, 1, ..]) + &a.slice(s![1, 0, ..]));
        a.slice_mut(s![0, 0, ..]).assign(&axx);

        let axx = 0.5 * (&a.slice(s![0, -2, ..]) + &a.slice(s![1, -1, ..]));
        a.slice_mut(s![0, -1, ..]).assign(&axx);

        let axx = 0.5 * (&a.slice(s![-2, 0, ..]) + &a.slice(s![-1, 1, ..]));
        a.slice_mut(s![-1, 0, ..]).assign(&axx);

        let axx = 0.5 * (&a.slice(s![-1, -2, ..]) + &a.slice(s![-2, -1, ..]));
        a.slice_mut(s![-1, -1, ..]).assign(&axx);
    }
}

impl BoundaryCondition<Ix3> for NoSlipBC<Ix3> {
    fn apply(&self, a: &mut ArrayViewMut<Float, Ix3>) {
        let s = -a.slice(s![1, .., ..]).into_owned();
        a.slice_mut(s![0, .., ..]).assign(&s);
        let s = -a.slice(s![-2, .., ..]).into_owned();
        a.slice_mut(s![-1, .., ..]).assign(&s);
        let s = -a.slice(s![.., 1, ..]).into_owned();
        a.slice_mut(s![.., 0, ..]).assign(&s);
        let s = -a.slice(s![.., -2, ..]).into_owned();
        a.slice_mut(s![.., -1, ..]).assign(&s);

        self.fix_corners(a);
    }

    fn name(&self) -> &'static str {
        "no_slip_bc"
    }
}

#[cfg(test)]
mod tests {
    use approx::*;

    use super::*;

    #[test]
    fn test_continuity_bc() {
        let mut a = Array2::zeros([6, 6]);
        let bc = ContinuityBC::new();

        // BCs should still be zero.
        a.slice_mut(s![1..-1, 1..-1]).fill(3.0);
        bc.apply(&mut a.view_mut());

        let mut a_expect = a.clone();
        a_expect.fill(3.0);
        assert_relative_eq!(&a_expect, &a);

        // Test the corner cases as these are handled differently.
        a.slice_mut(s![1, 1]).fill(5.0);
        a.slice_mut(s![1, -2]).fill(8.0);
        a.slice_mut(s![-2, 1]).fill(10.0);
        a.slice_mut(s![-2, -2]).fill(16.0);

        bc.apply(&mut a.view_mut());

        assert_relative_eq!(*a.slice(s![0, 0]).into_scalar(), 5.0);
        assert_relative_eq!(*a.slice(s![0, -1]).into_scalar(), 8.0);
        assert_relative_eq!(*a.slice(s![-1, 0]).into_scalar(), 10.0);
        assert_relative_eq!(*a.slice(s![-1, -1]).into_scalar(), 16.0);
    }

    #[test]
    fn test_no_slip_bc() {
        let mut a = Array3::from_elem([6, 6, 2], 3.0);
        let bc = NoSlipBC::new();

        // Lets make the array elements different enough to eliminate any indexing
        // weirdness.
        a.slice_mut(s![1, 1..-1, ..]).assign(&array![0.3, 0.5]); // Top row
        a.slice_mut(s![-2, 1..-1, ..]).assign(&array![0.7, 0.8]); // Bottom row
        a.slice_mut(s![1..-1, 1, ..]).assign(&array![0.9, 0.4]); // First column
        a.slice_mut(s![1..-1, -2, ..]).assign(&array![1.5, 8.7]); // Last column

        bc.apply(&mut a.view_mut());

        assert_relative_eq!(a.slice(s![0, 1..-1, ..]), -&a.slice(s![1, 1..-1, ..]));
        assert_relative_eq!(a.slice(s![-1, 1..-1, ..]), -&a.slice(s![-2, 1..-1, ..]));
        assert_relative_eq!(a.slice(s![1..-1, 0, ..]), -&a.slice(s![1..-1, 1, ..]));
        assert_relative_eq!(a.slice(s![1..-1, -1, ..]), -&a.slice(s![1..-1, -2, ..]));
    }
}
