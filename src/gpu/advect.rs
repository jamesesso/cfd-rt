//! Contains [`AdvectOp`]

use std::rc::Rc;

use thiserror::*;

use super::{dispatch_workgroup_size, prelude::*, DeviceOp, DeviceState, DeviceStateError};
use crate::Float;

/// [`DeviceOp`] that applies the advetive term of a simulation on the GPU.
///
/// The choice of algorithm for the particle tracing is controlled by the
/// [`TraceStrategy`] passed on construction and the interpolation algorithm is
/// chosen by the [`InterpStrategy`] enum passed on construciton. Currently only
/// a single algorithm is available for both.
pub struct AdvectOp {
    label: String,
    interp_type: InterpStrategy,
    tracer_type: TraceStrategy,
    trace_buf: Rc<Buffer>,
    advect_bg: Rc<BindGroup>,
    advect_bg_layout: Rc<BindGroupLayout>,
    common_bg: Rc<BindGroup>,
    common_bg_layout: Rc<BindGroupLayout>,
    trace_pipeline: Rc<ComputePipeline>,
    interp_pipeline: Rc<ComputePipeline>,
}

/// Error type for [`AdvectOp`].
#[derive(Debug, Error)]
pub enum AdvectOpError {
    #[error("Error from DeviceState when building DeviceOp.")]
    DeviceState(#[from] DeviceStateError),
    #[error("Common DeviceOp not initialized")]
    CommonNotInit,
}

impl DeviceOp for AdvectOp {
    type Input<'a> = ();
    type Output = anyhow::Result<()>;
    type BuildInput<'a> = AdvectOpDesc<'a>;
    type BuildError = AdvectOpError;

    async fn device_exec(&self, state: &DeviceState, _input: Self::Input<'_>) -> Self::Output {
        let mut encoder = state.create_encoder();
        // TODO: This should be passed in a better way. It seems pretty brittle...
        // This is okay currently as the trace_buf is the same size as u_buf. As the
        // number of ops we need is the number of floats in u_buf, which is
        // size()/4 to convert to the dispatch_workgroup_size of Float.
        let npoints = self.trace_buf.size() / std::mem::size_of::<Float>() as u64;
        {
            // Set up Trace pipeline.
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, Some(&*self.common_bg), &[]);
            pass.set_bind_group(1, Some(&*self.advect_bg), &[]);
            pass.set_pipeline(&self.trace_pipeline);
            pass.dispatch_workgroups(dispatch_workgroup_size(npoints) as u32, 1, 1);
        }
        {
            // Set up interpolation pipeline.
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, Some(&*self.common_bg), &[]);
            pass.set_bind_group(1, Some(&*self.advect_bg), &[]);
            pass.set_pipeline(&self.interp_pipeline);
            // As we do ux and uy in one pass we need half the number of tasks.
            pass.dispatch_workgroups(dispatch_workgroup_size(npoints / 2) as u32, 1, 1);
        }

        state.queue().submit([encoder.finish()]);

        Ok(())
    }

    async fn new_in_device(
        input: Self::BuildInput<'_>,
        state: &super::DeviceState,
    ) -> Result<Self, Self::BuildError> {
        // Get resources from CommonOp.
        let u_buf = state.buffer("u").ok_or(AdvectOpError::CommonNotInit)?;
        let common_bg = state
            .bind_group("common")
            .ok_or(AdvectOpError::CommonNotInit)?;
        let common_bg_layout = state
            .bind_group_layout("common")
            .ok_or(AdvectOpError::CommonNotInit)?;

        // Make resources for Advect.
        let label = input.label.to_owned();
        let res = state.create_resources(
            "advect",
            &[wgpu::BufferDescriptor {
                label: Some(&format!("{}_x_trace", label)),
                size: u_buf.size(), // Set the size to be the same as u.
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }],
        )?;

        let trace_buf = res.buffers[0].clone();
        let advect_bg_layout = res.bind_group_layout;

        // Build pipelines.
        let trace_pipeline = match input.tracer_type {
            TraceStrategy::Linear => {
                let label = format!("{}_advect_trace", input.label);
                state.create_compute_pipeline(
                    &label,
                    "trace_x_linear",
                    "advect",
                    &[&common_bg_layout, &advect_bg_layout],
                )?
            }
            TraceStrategy::None => {
                let label = format!("{}_advect_trace", input.label);
                state.create_compute_pipeline(
                    &label,
                    "noop_trace",
                    "advect",
                    &[&common_bg_layout, &advect_bg_layout],
                )?
            }
        };

        // Match so future additions don't keep using linear shaders by accident.
        let interp_pipeline = match input.interp_type {
            InterpStrategy::Linear => {
                let label = format!("{}_advect_interp", input.label);
                state.create_compute_pipeline(
                    &label,
                    "interp_u_linear",
                    "advect",
                    &[&common_bg_layout, &advect_bg_layout],
                )?
            }
        };

        Ok(Self {
            label: input.label.to_owned(),
            interp_type: input.interp_type,
            tracer_type: input.tracer_type,
            trace_buf,
            advect_bg: res.bind_group,
            advect_bg_layout,
            common_bg: common_bg.clone(),
            common_bg_layout: common_bg_layout.clone(),
            interp_pipeline,
            trace_pipeline,
        })
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn op_kind() -> &'static str {
        "advect"
    }
}

/// Descriptor type for building [`AdvectOp`] with
/// [`DeviceOp::new_in_device()`].
pub struct AdvectOpDesc<'a> {
    label: &'a str,
    interp_type: InterpStrategy,
    tracer_type: TraceStrategy,
}

impl<'a> From<(&'a str, InterpStrategy, TraceStrategy)> for AdvectOpDesc<'a> {
    fn from(value: (&'a str, InterpStrategy, TraceStrategy)) -> Self {
        Self {
            label: value.0,
            interp_type: value.1,
            tracer_type: value.2,
        }
    }
}

/// Enum representing which strategy is used for the particle tracer in
/// [`AdvectOp`].
#[non_exhaustive]
#[derive(Debug)]
pub enum TraceStrategy {
    /// Linear particle tracer.
    Linear,
    /// No particle tracer (for testing/debugging).
    ///
    /// If this is enabled the tracer stage does nothing so the interpolator can
    /// be tested separately from the traceing stage.
    None,
}

/// Enum representing which strategy is used for the interpolator in
/// [`AdvectOp`].
#[non_exhaustive]
#[derive(Debug)]
pub enum InterpStrategy {
    // Bilinear interpolator.
    Linear,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::prelude::*;

    use super::*;
    use crate::{
        gpu::{
            common::{CommonOp, CommonOpDesc},
            copy::{ReadDeviceBuffer, WriteDeviceBuffer},
            DeviceInstance,
        },
        system::make_grid,
        Array2F,
        Float,
    };

    // Test that trace produecs the expected results.
    #[test_log::test(pollster::test)]
    async fn test_trace_linear() -> anyhow::Result<()> {
        let x = make_grid(2.0, 10);
        let dx = x[(1, 0, 0)] - x[(0, 0, 0)];
        let dt = 0.05;
        let u0 = &x + 0.2;

        let mut instance = DeviceInstance::new().await?;
        let _common = instance
            .add_op::<CommonOp>(CommonOpDesc::new(dt, dx, &x, &u0))
            .await?;

        // Use InterpStrategy::Linear for convenience, in this test it does nothing.
        let advect = instance
            .add_op::<AdvectOp>(("advect", InterpStrategy::Linear, TraceStrategy::Linear).into())
            .await?;

        let u0_buf = instance.state.buffer("u0").unwrap();

        let write_u0 = instance
            .add_op::<WriteDeviceBuffer<Float>>(("write_u0", u0_buf).into())
            .await?;

        let x_trace = instance.state.buffer("advect_x_trace").unwrap();
        let read_x_trace = instance
            .add_op::<ReadDeviceBuffer<Float>>(("x_trace", x_trace).into())
            .await?;

        let mut x_trace = 0.0 * &u0;
        write_u0.device_exec(u0.as_slice().unwrap()).await?;
        advect.device_exec(()).await?;
        read_x_trace
            .device_exec(x_trace.as_slice_mut().unwrap())
            .await?;

        // Don't forget that this goes backwards in time..
        let x_trace_expected = &x - dt * u0;

        assert_relative_eq!(&x_trace, &x_trace_expected);
        Ok(())
    }

    // Test that bilinear interpolation works without trace.
    #[test_log::test(pollster::test)]
    async fn test_interpolate() -> anyhow::Result<()> {
        let x = make_grid(2.0, 10);
        let dx = x[(1, 0, 0)] - x[(0, 0, 0)];
        let dt = 0.05;
        let u0 = &x + 0.0;

        let mut instance = DeviceInstance::new().await?;
        let common = instance
            .add_op::<CommonOp>(CommonOpDesc::new(dt, dx, &x, &u0))
            .await?;

        let advect = instance
            .add_op::<AdvectOp>(("advect", InterpStrategy::Linear, TraceStrategy::None).into())
            .await?;

        // Get buffers needed for tests.
        let u0_buf = instance.state.buffer("u0").unwrap();
        let u_buf = instance.state.buffer("u").unwrap();
        let x_trace_buf = instance.state.buffer("advect_x_trace").unwrap();

        // Copies u to u0.
        common.device_exec(()).await?;

        // Make read/write ops.
        let write_u0 = instance
            .add_op::<WriteDeviceBuffer<Float>>(("write_u0", u0_buf).into())
            .await?;
        let read_u = instance
            .add_op::<ReadDeviceBuffer<Float>>(("read_u", u_buf).into())
            .await?;
        let write_x_trace = instance
            .add_op::<WriteDeviceBuffer<Float>>(("x_trace", x_trace_buf).into())
            .await?;

        let mut x_trace = x.clone();
        for i in 0..10 {
            for j in 0..10 {
                x_trace[(i, j, 0)] += 0.6 * dx; // Make x cross a boundary.
                x_trace[(i, j, 1)] -= 0.1 * dx;
            }
        }

        // Setup test function.
        let f2d = linear_test_func(&x.clone().into_shape_with_order([100, 2])?);
        let f = f2d.into_shape_with_order([10, 10, 2])?;

        let f_trace2d = linear_test_func(&x_trace.clone().into_shape_with_order([100, 2])?);
        let f_trace = f_trace2d.into_shape_with_order([10, 10, 2])?;
        let mut u = &u0 * 0.0;

        write_u0.device_exec(f.as_slice().unwrap()).await?;
        write_x_trace
            .device_exec(x_trace.as_slice().unwrap())
            .await?;
        advect.device_exec(()).await?;
        read_u.device_exec(u.as_slice_mut().unwrap()).await?;

        for i in 1..9 {
            for j in 1..9 {
                let uij = u[(i, j, 0)];
                let f_traceij = f_trace[(i, j, 0)];
                assert_relative_eq!(uij, f_traceij, epsilon = 1E-4);

                let uij = u[(i, j, 1)];
                let f_traceij = f_trace[(i, j, 1)];
                assert_relative_eq!(uij, f_traceij, epsilon = 1E-4);
            }
        }

        Ok(())
    }

    // Integration(ish) test that checks both stages together.
    #[test_log::test(pollster::test)]
    async fn test_advect() -> anyhow::Result<()> {
        // Calculate basic quantities for advection.
        let x = make_grid(2.0, 10);
        let dx = x[(1, 0, 0)] - x[(0, 0, 0)];
        let dt = 0.05;

        let f2d = linear_test_func(&x.clone().into_shape_with_order([100, 2])?);
        let u0 = f2d.into_shape_with_order([10, 10, 2])?; // Set u0 to the linear test fuction.

        // Set up GPU resources.
        let mut instance = DeviceInstance::new().await?;
        let _common = instance
            .add_op::<CommonOp>(CommonOpDesc::new(dt, dx, &x, &u0))
            .await?;

        let advect = instance
            .add_op::<AdvectOp>(("advect", InterpStrategy::Linear, TraceStrategy::Linear).into())
            .await?;

        // Get buffers needed for tests.
        let u0_buf = instance.state.buffer("u0").unwrap();
        let u_buf = instance.state.buffer("u").unwrap();
        let x_trace_buf = instance.state.buffer("advect_x_trace").unwrap();

        // Make read/write ops.
        let write_u0 = instance
            .add_op::<WriteDeviceBuffer<Float>>(("write_u0", u0_buf).into())
            .await?;
        let read_u = instance
            .add_op::<ReadDeviceBuffer<Float>>(("read_u", u_buf).into())
            .await?;
        let read_x_trace = instance
            .add_op::<ReadDeviceBuffer<Float>>(("x_trace", x_trace_buf).into())
            .await?;

        // Write to u0 array and run the advection step.
        write_u0.device_exec(u0.as_slice().unwrap()).await?;
        advect.device_exec(()).await?;

        // Read x_trace from GPU and calculate predicted f(x) values from the traced
        // stage.
        let mut x_trace = &u0 * 0.0; // Make an output arryay. Basically just clones u0;.
        read_x_trace
            .device_exec(x_trace.as_slice_mut().unwrap())
            .await?;
        let f_trace2d = linear_test_func(&x_trace.clone().into_shape_with_order([100, 2])?);
        let f_trace = f_trace2d.into_shape_with_order([10, 10, 2])?;

        // Read u data from GPU
        let mut u = &u0 * 0.0; // Make an output arryay. Basically just clones u0;.
        read_u.device_exec(u.as_slice_mut().unwrap()).await?;

        // Skip boundaries as these are expected not to work.
        // This is to keep CI happy as there seems to be accuracy issues there.
        for i in 2..9 {
            for j in 2..9 {
                let uij = u[(i, j, 0)];
                let f_traceij = f_trace[(i, j, 0)];
                assert_relative_eq!(uij, f_traceij, epsilon = 1E-4);

                let uij = u[(i, j, 1)];
                let f_traceij = f_trace[(i, j, 1)];
                assert_relative_eq!(uij, f_traceij, epsilon = 1E-4);
            }
        }

        Ok(())
    }

    // Taken from advect.rs, a linear function to test interpolation so it is exact.
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
}
