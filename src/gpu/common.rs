//! [`CommonOp`] that contains common buffers.
use std::{mem, rc::Rc};

use thiserror::*;

use super::{buffer_size, prelude::*, DeviceOp, DeviceStateError};
use crate::{Array3F, Float};

/// [`DeviceOp`] that initializes common simulation buffers and copies `u` to
/// `u0` at the end of a simulation step. The buffers added are:
/// | Buffer | Usage |
/// | ------ | ----- |
/// |  `dt`  | Timestep |
/// |  `dx`  | Grid spacing |
/// |  `N`   | `vec3` containing `(Nx, Ny, 2)` |
/// |  `u`   | velocity array of the current step. |
/// |  `u0`   | velocity array of the previous step. |
///
/// (`Nx` and `Ny` are the number of poitns in the x and y direction
/// respectively).
///
/// For this operation [`CommonOp::device_exec()`] is comparitvely less
/// important for this op than many others as it just performs the copy.
///
/// **Note: Unlike many other ops this should only be used once per
/// simulation.**
#[derive(Debug)]
pub struct CommonOp {
    dx_buf: Rc<Buffer>,
    dt_buf: Rc<Buffer>,
    N_buf: Rc<Buffer>,
    u_buf: Rc<Buffer>,
    u0_buf: Rc<Buffer>,
}

impl DeviceOp for CommonOp {
    type Input<'a> = ();
    type Output = Result<(), CommonOpError>;
    type BuildInput<'a> = CommonOpDesc<'a>;
    type BuildError = CommonOpError;
    async fn new_in_device(
        input: Self::BuildInput<'_>,
        state: &super::DeviceState,
    ) -> Result<Self, Self::BuildError> {
        let n_points = buffer_size(input.N[0] * input.N[1]);
        let vec_n_points = buffer_size(input.N[2] * n_points);
        let vec_n_points_u8 = vec_n_points * mem::size_of::<Float>();

        let buf_desc_arr = [
            wgpu::BufferDescriptor {
                label: Some("dt"),
                size: std::mem::size_of::<Float>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
            wgpu::BufferDescriptor {
                label: Some("dx"),
                size: std::mem::size_of::<Float>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
            wgpu::BufferDescriptor {
                label: Some("N"),
                size: 3 * std::mem::size_of::<u32>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
            wgpu::BufferDescriptor {
                label: Some("u"),
                size: vec_n_points_u8 as BufferAddress,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
            wgpu::BufferDescriptor {
                label: Some("u0"),
                size: vec_n_points_u8 as BufferAddress,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        ];

        let res = state.create_resources("common", &buf_desc_arr[..])?;
        // This never fails as Vec is fixed length.
        let [dt_buf, dx_buf, N_buf, u_buf, u0_buf] = res.buffers.try_into().unwrap();

        let N_arr_u32 = input.N.map(|x| x as u32);
        state
            .queue()
            .write_buffer(&dt_buf, 0, bytemuck::bytes_of(&input.dt));
        state
            .queue()
            .write_buffer(&dx_buf, 0, bytemuck::bytes_of(&input.dx));
        state
            .queue()
            .write_buffer(&N_buf, 0, bytemuck::bytes_of(&N_arr_u32));

        Ok(Self {
            dx_buf,
            dt_buf,
            N_buf,
            u_buf,
            u0_buf,
        })
    }

    async fn device_exec(
        &self,
        state: &super::DeviceState,
        _input: Self::Input<'_>,
    ) -> Self::Output {
        let mut encoder = state.create_encoder();
        encoder.copy_buffer_to_buffer(&self.u_buf, 0, &self.u0_buf, 0, self.u_buf.size());
        state.queue().submit([encoder.finish()]);

        Ok(())
    }

    fn label(&self) -> String {
        "common".to_owned()
    }

    fn op_kind() -> &'static str {
        "common"
    }
}

/// Builder struct for [`CommonOp`].
#[derive(Debug)]
pub struct CommonOpDesc<'a> {
    dt: Float,
    dx: Float,
    N: [usize; 3],
    u: &'a Array3F,
}

impl<'a> CommonOpDesc<'a> {
    fn new(dt: Float, dx: Float, u: &'a Array3F) -> Self {
        let u_dims = u.shape().try_into().unwrap();
        Self {
            dt,
            dx,
            N: u_dims,
            u,
        }
    }
}

/// Error type for [`CommonOp`].
#[derive(Debug, Error)]
pub enum CommonOpError {
    #[error("Error from device")]
    DeviceError(#[from] DeviceStateError),
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::prelude::*;

    use super::*;
    use crate::gpu::{
        copy::{ReadDeviceBuffer, WriteDeviceBuffer},
        DeviceInstance,
        DeviceState,
    };

    #[test_log::test(pollster::test)]
    async fn test_common_buffers() -> anyhow::Result<()> {
        let mut instance = DeviceInstance::new().await?;
        let mut u0_arr = Array3F::zeros([10, 12, 2]);

        // Non-square grids not supported yet, but GPU op does, so this makes the checks
        // a little better.
        for i in 0..10 {
            for j in 0..12 {
                u0_arr.slice_mut(s![i, j, 0]).fill(i as Float);
                u0_arr.slice_mut(s![i, j, 1]).fill(j as Float);
            }
        }

        let dt = 0.2;
        let dx = 3.5;
        let common_buf_desc = CommonOpDesc::new(dt, dx, &u0_arr);

        let common_op = instance.add_op::<CommonOp>(common_buf_desc).await?;
        let u0_buf = common_op.op.u0_buf.clone();
        let u_buf = common_op.op.u_buf.clone();

        let u0_write = instance
            .add_op::<WriteDeviceBuffer<Float>>(("u0_write", u0_buf).into())
            .await?;
        let u_read = instance
            .add_op::<ReadDeviceBuffer<Float>>(("u_read", u_buf).into())
            .await?;

        let test_op = instance.add_op::<TestOp>(()).await?;

        // Set to non-zero value just in case of some uninit memory weirdness. It will
        // be overwritten in a second.
        let mut u_arr = 1.5 * &u0_arr;
        // Check that doubling the array works.
        u0_write.device_exec(u0_arr.as_slice().unwrap()).await?;
        test_op.device_exec(TestOpKind::DoubleU).await?;
        u_read.device_exec(u_arr.as_slice_mut().unwrap()).await?;
        assert_relative_eq!(u_arr, 2.0 * &u0_arr);

        // This index assigns the x component of the grid to 2 * i and j to 3 * j. This
        // checks that the conversion in the shader between a 1D idx and 3D idx
        // is correct.
        test_op.device_exec(TestOpKind::IndexTest).await?;
        u_read.device_exec(u_arr.as_slice_mut().unwrap()).await?;
        for i in 0..10 {
            for j in 0..12 {
                assert_relative_eq!(u_arr[(i, j, 0)], 2.0 * i as Float);
                assert_relative_eq!(u_arr[(i, j, 1)], 3.0 * j as Float);
            }
        }

        // This test writes dt to all the x values and dt to the y values. This ensures
        // that all the uniform buffers are correct.
        test_op.device_exec(TestOpKind::DtDx).await?;
        u_read.device_exec(u_arr.as_slice_mut().unwrap()).await?;
        let ux = u_arr.slice(s![.., .., 0]);
        let uy = u_arr.slice(s![.., .., 1]);
        assert_relative_eq!(ux, &ux * 0.0 + dt);
        assert_relative_eq!(uy, &uy * 0.0 + dx);

        Ok(())
    }

    // Test DeviceOp for running tests using double_common shader.
    struct TestOp {
        double_u_pipeline: Rc<ComputePipeline>,
        dt_dx_pipeline: Rc<ComputePipeline>,
        index_test_pipeline: Rc<ComputePipeline>,
    }

    impl DeviceOp for TestOp {
        type Input<'a> = TestOpKind;
        type Output = anyhow::Result<()>;
        type BuildInput<'a> = ();
        type BuildError = DeviceStateError;

        async fn device_exec(&self, state: &DeviceState, input: Self::Input<'_>) -> Self::Output {
            let mut encoder = state.create_encoder();
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &*state.bind_group("common").unwrap(), &[]);
                match input {
                    TestOpKind::DoubleU => pass.set_pipeline(&self.double_u_pipeline),
                    TestOpKind::DtDx => pass.set_pipeline(&self.dt_dx_pipeline),
                    TestOpKind::IndexTest => pass.set_pipeline(&self.index_test_pipeline),
                };
                pass.dispatch_workgroups(32, 1, 1);
            }

            state.queue().submit([encoder.finish()]);
            Ok(())
        }

        async fn new_in_device(
            _input: Self::BuildInput<'_>,
            state: &DeviceState,
        ) -> Result<Self, Self::BuildError> {
            state.add_shader("test_common", include_str!("shaders/tests/common.wgsl"))?;
            let double_u = state.create_compute_pipeline(
                "double_u",
                "double_u",
                "test_common",
                &[&state.bind_group_layout("common").unwrap()],
            )?;

            let index_test = state.create_compute_pipeline(
                "index_test",
                "index_test",
                "test_common",
                &[&state.bind_group_layout("common").unwrap()],
            )?;

            let dt_dx = state.create_compute_pipeline(
                "dx_dt_test",
                "dx_dt_test",
                "test_common",
                &[&state.bind_group_layout("common").unwrap()],
            )?;

            Ok(Self {
                double_u_pipeline: double_u,
                dt_dx_pipeline: dt_dx,
                index_test_pipeline: index_test,
            })
        }

        fn label(&self) -> String {
            "test_common".to_owned()
        }

        fn op_kind() -> &'static str {
            unimplemented!();
        }
    }

    enum TestOpKind {
        DoubleU,
        DtDx,
        IndexTest,
    }
}
