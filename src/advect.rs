//! Advection types and traits.

use downcast_rs::Downcast;
use ndarray::prelude::*;

use crate::{
    system::{Data, Grid, SystemOp},
    Array1F,
    Array2F,
    Array3F,
    ArrayBase1F,
    ArrayBase2F,
    ArrayBase3F,
    DataF,
    DataMutF,
    Float,
};

/// [SystemOp] that performs Semi-Lagrangian advection.
#[derive(Debug, Clone)]
pub struct Advect<T, I> {
    trace: T,
    interp: I,
}

/// Generic trait for Semi-Lagrangian particle tracer.
///
/// Represents some kind of particle tracer (linear, Runge-Kutta, etc.) which
/// takes a particle position at time t and returns a particle at time t + dt.
pub trait Trace {
    /// and returns the position at a time t + dt.
    fn trace<SX: DataF, SU: DataF>(
        &self,
        dt: Float,
        x: &ArrayBase3F<SX>,
        u: &ArrayBase3F<SU>,
    ) -> Array3F;
}

/// [Trace] that performs a linear trace.
///
/// A linear interpolator, which assumes the velocity is constant between x(t)
/// and x(t + dt). This means that x(t + dt) = x + u * dt.
#[derive(Debug, Clone)]
pub struct LinearTrace();

impl Trace for LinearTrace {
    fn trace<SX: DataF, SU: DataF>(
        &self,
        dt: Float,
        x: &ArrayBase3F<SX>,
        u: &ArrayBase3F<SU>,
    ) -> Array3F {
        x + u * dt
    }
}

/// Generic trait for 2D interpolators on a grid for [Advect].
///
/// Given a [Grid] and values of a function on this grid f(x_grid), calculates
/// the value of f(x) for off grid values of x.
pub trait Interpolate {
    fn interp<SX: DataF, SF: DataF>(
        &self,
        x_arr: &ArrayBase2F<SX>,
        f: &ArrayBase3F<SF>,
        grid: &Grid,
    ) -> Array2F;
}

/// [Interpolate] that performs Bilinear Interpolation.
#[derive(Debug, Clone)]
pub struct LinearInterpolate();

impl Interpolate for LinearInterpolate {
    fn interp<SX: DataF, SF: DataF>(
        &self,
        x_arr: &ArrayBase2F<SX>,
        f: &ArrayBase3F<SF>,
        grid: &Grid,
    ) -> Array2F {
        // TODO: This could do with some cleanup...
        // TODO: Grid should have info about number of points.
        // TODO: The ndarray matrix indices need swapping around.
        let x_arr = x_arr.clamp(grid.xlo + grid.dx, grid.xhi - grid.dx);
        let N = ((grid.xhi - grid.xlo) / grid.dx).round() as usize;

        let mut out = Array2::zeros(x_arr.raw_dim());
        for i in 0..x_arr.shape()[0] {
            let q11 = self.find_q11(x_arr.slice(s![i, ..]), grid);
            let q11i = q11[0] as usize;
            let q11j = q11[1] as usize;
            let q11i = q11i.clamp(1, N - 2);
            let q11j = q11j.clamp(1, N - 2);

            let x = *x_arr.slice(s![i, 0]).into_scalar();
            let y = *x_arr.slice(s![i, 1]).into_scalar();

            let x1 = q11i as Float * grid.dx + grid.xlo;
            let x2 = x1 + grid.dx;

            let y1 = q11j as Float * grid.dx + grid.xlo;
            let y2 = y1 + grid.dx;

            let f11 = f.slice(s![q11i, q11j, ..]);
            let f12 = f.slice(s![q11i, q11j + 1, ..]);
            let f21 = f.slice(s![q11i + 1, q11j, ..]);
            let f22 = f.slice(s![q11i + 1, q11j + 1, ..]);

            let mut fxy = &f11 * (x2 - x) * (y2 - y)
                + &f21 * (x - x1) * (y2 - y)
                + &f12 * (x2 - x) * (y - y1)
                + &f22 * (x - x1) * (y - y1);
            fxy /= (x2 - x1) * (y2 - y1);

            out.slice_mut(s![i, ..]).assign(&fxy);
        }
        out
    }
}

impl LinearInterpolate {
    fn find_q11<S: DataF>(&self, x: ArrayBase1F<S>, grid: &Grid) -> Array1F {
        let q11 = (&x - grid.xlo) / grid.dx;

        q11.floor()
    }
}

impl<T: Trace + Downcast, I: Interpolate + Downcast> SystemOp for Advect<T, I> {
    fn exec(&mut self, data: &mut Data) -> anyhow::Result<()> {
        let mut new_u = data.u.clone();
        self.advect(&mut new_u, &data.x, &data.u, data.dt);
        data.u = new_u;

        Ok(())
    }

    fn op_type(&self) -> &'static str {
        "advect"
    }
}

impl<T: Trace, I: Interpolate> Advect<T, I> {
    /// Builds a new Advect from a chosen [Trace] and [Interpolate].
    pub fn new(trace: T, interp: I) -> Self {
        Self { trace, interp }
    }

    /// Advects an array a at time t to a time t + dt, using a grid defined by x
    /// and a velocity field u.
    pub fn advect<SA: DataMutF, SX: DataF, SU: DataF>(
        &self,
        a: &mut ArrayBase3F<SA>,
        x: &ArrayBase3F<SX>,
        u: &ArrayBase3F<SU>,
        dt: Float,
    ) {
        let x_dt = self.trace.trace(-dt, x, u);
        let trim = u.shape();

        // TODO: This can probably be removed.
        let s = (trim[0] - 2) * (trim[1] - 2);
        let x_dt_trim = x_dt
            .slice(s![1..-1, 1..-1, ..])
            .to_owned()
            .into_shape_with_order([s, 2])
            .unwrap();

        // TODO: Fix this.
        let xlo = *x.slice(s![0, 0, 0]).into_scalar();
        let xhi = *x.slice(s![-1, 0, 0]).into_scalar();
        let dx = x.slice(s![1, 0, 0]).into_scalar() - xlo;

        let grid = Grid { dx, xlo, xhi };
        let interp_arr = self.interp.interp(&x_dt_trim, &*a, &grid);

        let shape = u.shape();
        a.slice_mut(s![1..-1, 1..-1, ..]).assign(
            &interp_arr
                .into_shape_with_order([shape[0] - 2, shape[1] - 2, 2])
                .unwrap(),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::ops::AddAssign;

    use approx::*;

    use super::*;
    use crate::system::make_grid;

    // Fairly simple test, but I think it does the job in checking the simple trace.
    // Not exactly mutch to test other than x + u * t.
    #[test]
    fn test_linear_trace() {
        let x = make_grid(1.0, 5);
        let dims = x.raw_dim();

        // Apply a constant velocity in the x direction.
        let mut u = Array::zeros(dims);
        u.slice_mut(s![.., .., 0]).fill(0.1);

        let trace = LinearTrace();
        let traced = trace.trace(0.1, &x, &u);

        let mut x_traced = x;
        x_traced.slice_mut(s![.., .., 0]).add_assign(0.01);
        assert_eq!(traced, x_traced);
    }

    fn linear_test_func(x: &Array2F) -> Array2F {
        let x0 = x.slice(s![.., 0]);
        let x1 = x.slice(s![.., 1]);

        let y0 = 3.0 * &x0 + 2.0 * &x1;
        let y1 = 2.0 * &x0 + 7.0 * &x1;

        let mut y = 0.0 * &x.clone();

        y.slice_mut(s![.., 0]).assign(&y0);
        y.slice_mut(s![.., 1]).assign(&y1);
        y
    }

    // Simple test with linear function so that the interpolation is exact to
    // numerical precisision.
    #[test]
    fn test_linear_interpolate() {
        let x = make_grid(1.0, 10);

        let x_2d = x.to_owned().into_shape_with_order([10 * 10, 2]).unwrap();
        let y_2d = linear_test_func(&x_2d);
        let y = y_2d.to_owned().into_shape_with_order([10, 10, 2]).unwrap();

        let interp = LinearInterpolate();
        let x_test = Array2::from(vec![[1.7, 1.9], [3.2, 1.8], [3.2, 7.1], [5.6, 9.2]]);

        let grid = Grid {
            dx: 1.0,
            xlo: 0.5,
            xhi: 10.5,
        };

        let y_interp = interp.interp(&x_test, &y, &grid);
        let y_actual = linear_test_func(&x_test);

        azip!((yi in &y_interp, ya in &y_actual) assert_abs_diff_eq!(yi, ya , epsilon=1E-5));
    }

    // This runs through the interpolation manually and with advect to check that
    // they are consistent.
    #[test]
    fn test_advect() {
        let dt = 0.05;
        let x = make_grid(1.0, 10);

        let mut u: Array3F = 0.0 * &x;

        u.slice_mut(s![.., .., 0])
            .assign(&(0.01 * x.slice(s![.., .., 0]).powf(2.0)));
        u.slice_mut(s![.., .., 1])
            .assign(&(0.04 * x.slice(s![.., .., 0]).powf(2.0)));

        let advect = Advect::new(LinearTrace(), LinearInterpolate());

        let mut u_advect = u.clone();
        advect.advect(&mut u_advect, &x, &u, dt);

        let trace = LinearTrace();
        let interp = LinearInterpolate();

        let x_interp = trace.trace(-0.05, &x, &u);
        let grid = Grid {
            dx: 1.0,
            xlo: 0.5,
            xhi: 10.5,
        };

        let x2d = x_interp
            .to_owned()
            .into_shape_with_order([10 * 10, 2])
            .unwrap();
        let u_interp2d = interp.interp(&x2d, &u, &grid);
        let u_interp = u_interp2d
            .to_owned()
            .into_shape_with_order([10, 10, 2])
            .unwrap();

        assert_abs_diff_eq!(
            u_advect.slice(s![1..-1, 1..-1, ..]),
            u_interp.slice(s![1..-1, 1..-1, ..]),
            epsilon = 1E-5
        );
    }
}
