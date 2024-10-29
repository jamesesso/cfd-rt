//! Module for 'Projection' for ensuring incompressibility.

use ndarray::{prelude::*, stack};

use crate::{
    bc::*,
    calc::laplacian,
    system::{Data, SystemOp},
    Array2F,
    Array3F,
    ArrayBase2F,
    ArrayBase3F,
    DataF,
    DataMutF,
    Float,
};

/// [SystemOp] that performs projection with Gauss-Seidel iteration.
///
/// This is performed by solving the following: laplacian(q) = div(u^n) and
/// u^(n+1) = u - grad(q). The Poisson Equation is solved with Gauss-Seidel
/// iteration.
#[derive(Debug)]
pub struct ProjectGaussSeidel {
    // TODO: Can some of this be extracted out into some general solver code?
    Ncycles: usize,
    Ncheck: usize,
    tolerance: Float,
    div_bc: Box<dyn BoundaryCondition<Ix2>>,
    u_bc: Box<dyn BoundaryCondition<Ix3>>,
}

/// Result for convergence.
///
/// An Ok result contains the number of cycles and the RMS error and indicates
/// that convergence did not fail. An Err result indicates that the convergence
/// failed and just contains the RMS error.
type ConvergeResult = Result<(usize, Float), Float>;

impl SystemOp for ProjectGaussSeidel {
    fn exec(&mut self, data: &mut Data) -> anyhow::Result<()> {
        // TODO: Repalce with Grid?
        let dx = data.x[(1, 0, 0)] - data.x[(0, 0, 0)];

        self.project(&mut data.u, dx);
        Ok(())
    }

    fn op_type(&self) -> &'static str {
        "project_gauss_seidel"
    }
}

impl ProjectGaussSeidel {
    pub fn new(
        Ncycles: usize,
        Ncheck: usize,
        tolerance: Float,
        div_bc: impl BoundaryCondition<Ix2> + 'static,
        u_bc: impl BoundaryCondition<Ix3> + 'static,
    ) -> Self {
        let div_bc = Box::new(div_bc);
        let u_bc = Box::new(u_bc);
        ProjectGaussSeidel {
            Ncycles,
            Ncheck,
            tolerance,
            div_bc,
            u_bc,
        }
    }

    /// Performs the whole projection operation on a particular array.
    ///
    /// [ProjectGaussSeidel::exec()] essentially forwards to this.
    fn project<S: DataMutF>(&self, a: &mut ArrayBase3F<S>, dx: Float) {
        // TODO: Investigate project code. Under some circumstances does not converge.
        let div = self.div_u(a, dx);
        let mut p = Array2::zeros(div.raw_dim());

        // Currently I ignore if the system does not converge as it only degrades
        // visuals unlike a 'real' simulation.
        let _ = self.converge(&mut p, &div, dx);

        self.div_bc.apply(&mut p.view_mut());

        let p_grad = self.grad(&p, dx);

        *a -= &p_grad;
        self.u_bc.apply(&mut a.view_mut());
    }

    /// Calculates the divergence of a 3D array
    ///
    /// When running also applies the 'div_bc' [BoundaryCondition].
    fn div_u<S: DataF>(&self, u: &ArrayBase3F<S>, dx: Float) -> Array2F {
        let width = u.shape();

        let dudx = &u.slice(s![2.., 1..-1, 0]) - &u.slice(s![..(width[1] - 2), 1..-1, 0]);
        let dudy = &u.slice(s![1..-1, 2.., 1]) - &u.slice(s![1..-1, ..(width[1] - 2), 1]);

        let divu = 0.5 * (&dudx + &dudy) / dx;

        let out_dim: [usize; 2] = u.shape()[0..2].try_into().unwrap();
        let mut out = Array2::zeros(out_dim);
        out.slice_mut(s![1..-1, 1..-1]).assign(&divu);

        self.div_bc.apply(&mut out.view_mut());

        out
    }

    /// Performs a single Gauss-Seidel step on an array u.
    fn step<SU: DataMutF, SD: DataF>(
        &self,
        u: &mut ArrayBase2F<SU>,
        div: &ArrayBase2F<SD>,
        dx: Float,
    ) {
        let N = u.shape()[0];
        for i in 1..(N - 1) {
            for j in 1..(N - 1) {
                let u_n1j = *u.slice(s![i - 1, j]).into_scalar();
                let u_p1j = *u.slice(s![i + 1, j]).into_scalar();
                let u_1nj = *u.slice(s![i, j - 1]).into_scalar();
                let u_1pj = *u.slice(s![i, j + 1]).into_scalar();

                let u_ij = u.slice_mut(s![i, j]);

                *u_ij.into_scalar() = 0.25
                    * (u_n1j + u_p1j + u_1nj + u_1pj
                        // - 4.0 * uij_old
                        - &div.slice(s![i, j]) * dx.powi(2))
                    .into_scalar();
            }
        }
        self.div_bc.apply(&mut u.view_mut());
    }

    /// Converges the Poisson Equation with Gauss-Seidel iteration.
    fn converge<SP: DataMutF, SD: DataF>(
        &self,
        p: &mut ArrayBase2F<SP>,
        div: &ArrayBase2F<SD>,
        dx: Float,
    ) -> ConvergeResult {
        for i in 0..self.Ncycles {
            self.step(p, div, dx);

            if i % self.Ncheck == 0 {
                let lap_p = laplacian(p, dx);
                let norm_err = (&lap_p - div).powi(2).sum().sqrt();
                println!("error: {}", norm_err);
                if norm_err < self.tolerance {
                    return Ok((i, norm_err));
                }
            }
        }

        // Do a final check before exiting loop.
        let lap_p = laplacian(p, dx);
        let norm_err = (&lap_p - div).powi(2).sum().sqrt();
        if norm_err < self.tolerance {
            return Ok((self.Ncycles, norm_err));
        }

        Err(norm_err)
    }

    /// Calculates grad(q).
    fn grad<S: DataF>(&self, q: &ArrayBase2F<S>, dx: Float) -> Array3F {
        let width = q.shape();
        let dudx = &q.slice(s![2.., 1..-1]) - &q.slice(s![..(width[1] - 2), 1..-1]);
        let dudy = &q.slice(s![1..-1, 2..]) - &q.slice(s![1..-1, ..(width[1] - 2)]);

        let mut out = Array3::zeros([width[0], width[1], 2]);
        out.slice_mut(s![1..-1, 1..-1, ..]).assign(&stack![
            Axis(2),
            0.5 * dudx / dx,
            0.5 * dudy / dx
        ]);

        out
    }
}

#[cfg(test)]
mod tests {
    use approx::*;

    use super::*;
    use crate::system::make_grid;

    #[test]
    fn test_div() {
        let N = 10;
        let x = make_grid(1.0, N);
        let x0 = x.slice(s![.., .., 0]);
        let x1 = x.slice(s![.., .., 1]);

        // This is a simple test function that div * f can be found analytically.
        let mut fx = x.clone();
        fx.slice_mut(s![.., .., 0])
            .assign(&(2.0 * &x0.powi(2) + 5.0 * &x1));
        fx.slice_mut(s![.., .., 1])
            .assign(&(3.0 * &x0.powi(2) + 7.0 * &x1));

        let project = ProjectGaussSeidel::new(10, 1, 1E-6, ContinuityBC::new(), NoSlipBC::new());
        let div_fx_anal = 4.0 * &x0 + 7.0;
        let div_u_fd = project.div_u(&fx, 1.0);
        assert_abs_diff_eq!(
            div_fx_anal.slice(s![1..-1, 1..-1]),
            div_u_fd.slice(s![1..-1, 1..-1]),
            epsilon = 1E-5
        )
    }

    #[test]
    fn test_grad() {
        let N = 10;
        let x = make_grid(1.0, N);
        let x0 = x.slice(s![.., .., 0]);
        let x1 = x.slice(s![.., .., 1]);

        // This is a simple test function that div * f can be found analytically.
        let mut fx = x.clone();
        fx.slice_mut(s![.., .., 0])
            .assign(&(2.0 * &x0.powi(2) + 5.0 * &x1));
        fx.slice_mut(s![.., .., 1])
            .assign(&(3.0 * &x0.powi(2) + 7.0 * &x1));

        let project = ProjectGaussSeidel::new(10, 1, 1E-6, ContinuityBC::new(), NoSlipBC::new());
        let div_fx_anal = 4.0 * &x0 + 7.0;
        let div_u_fd = project.div_u(&fx, 1.0);
        assert_abs_diff_eq!(
            div_fx_anal.slice(s![1..-1, 1..-1]),
            div_u_fd.slice(s![1..-1, 1..-1]),
            epsilon = 1E-5
        )
    }

    #[test]
    #[ignore = "Convergence is slow on debug build."]
    fn test_gauss_seidel_converge() {
        let div_bc = ContinuityBC::new();
        let u_bc = NoSlipBC::new();
        let project = ProjectGaussSeidel::new(5000, 5, 1E-6, div_bc.clone(), u_bc.clone());
        let N = 20;

        // I think there is something pathologic about the test function going right to
        // the edges.
        let x = make_grid(1.0, N);
        let s0 = s![3..-3, 3..-3, 0];
        let s1 = s![3..-3, 3..-3, 1];
        let x0 = x.slice(s0);
        let x1 = x.slice(s1);
        let y0 = (4.0 - (&x0 - 11.0).powi(3).abs().sqrt()) + (3.0 - (&x1 - 7.0).powi(2).sqrt());
        let y1 =
            (6.0 - (&x0 - 10.0).powi(3).abs().sqrt()) + (17.0 - (&x1 - 10.0).powi(2).sqrt() + 4.0);

        let mut fx = Array3::zeros([N, N, 2]);
        fx.slice_mut(s0).assign(&y0);
        fx.slice_mut(s1).assign(&y1);

        u_bc.apply(&mut fx.view_mut());
        let mut div_fx = project.div_u(&fx, 1.0);
        div_bc.apply(&mut div_fx.view_mut());

        let mut p = Array2::zeros(div_fx.raw_dim());
        let res = project.converge(&mut p, &div_fx, 1.0);

        match res {
            Ok((_, norm)) => {
                assert!(norm < 1E-6, "Norm too large for converged run.");
                let norm_actual = (laplacian(&p, 1.0) - div_fx).powi(2).sum().sqrt();
                assert_relative_eq!(norm, norm_actual);
            }
            Err(_) => {
                panic!("Convergence did not finish.");
            }
        }
    }
}
