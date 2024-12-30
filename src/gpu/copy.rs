//! [`DeviceOp`] types for copying between CPU and GPU memory.
use std::{convert::From, rc::Rc};

use thiserror::*;

use crate::gpu::{prelude::*, DeviceOp, DeviceState, DeviceStateError};

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum CopyBufferError {
    #[error(
        "Provided slice is larger than the provided buffer size. slice size: {0}, buffer size {1}"
    )]
    SliceMismatch(usize, usize),
    #[error("Error from DeviceState")]
    DeviceError(#[from] DeviceStateError),
}

/// Copies data from a [`Buffer`] into `&mut [T]`.
///
/// If the slice is smaller than the buffer, the copy will stop when the slice
/// has been filled. If the buffer is larger, the operation will return a
/// [`CopyBufferError`].
#[derive(Debug)]
pub struct ReadDeviceBuffer<T> {
    label: String,
    buffer: Rc<Buffer>,
    staging: Rc<Buffer>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> DeviceOp for ReadDeviceBuffer<T>
where
    T: bytemuck::Pod,
{
    type Input<'a> = &'a mut [T];
    type BuildInput<'a> = ReadDeviceBufferDesc<'a>;
    type BuildError = DeviceStateError;
    type Output = Result<(), anyhow::Error>;

    /// Reads a buffer on the GPU and writes its contents to `slice_out`.
    ///
    /// # Panics
    /// If `(Buffer.size() % std::mem::size_of<T>()) != 0` this function will
    /// panic.
    async fn device_exec(&self, state: &DeviceState, slice_out: Self::Input<'_>) -> Self::Output {
        let slice_out_u8: &mut [u8] = bytemuck::cast_slice_mut(slice_out);

        // Check that GPU buffer is at least as long as output slice.
        if self.buffer.size() < slice_out_u8.len() as BufferAddress {
            Err(CopyBufferError::SliceMismatch(
                slice_out_u8.len(),
                self.buffer.size() as usize,
            ))?;
        }

        // Submit a very small job to the GPU, just copy the Buffer and exit.
        let mut encoder = state.create_encoder();
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &self.staging,
            0,
            slice_out_u8.len() as BufferAddress,
        );
        state.queue().submit([encoder.finish()]);

        let range = 0..(slice_out_u8.len() as BufferAddress);
        let staging_slice = self.staging.slice(range.clone());

        // Map the Buffer to CPU memory, then write it. This does NOT block on the web
        // as the GPU is immediately polled and the channel reciever is async.
        // The unwraps on the channel are fine as one end is never dropped.
        let (tx, rx) = flume::bounded(1);
        staging_slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        state.device.poll(wgpu::Maintain::wait());
        rx.recv_async().await.unwrap()?;

        slice_out_u8.copy_from_slice(&staging_slice.get_mapped_range()[..]);

        self.staging.unmap();

        Ok(())
    }

    // Simply creates a staging buffer and stores it inside Self.
    async fn new_in_device(
        input: Self::BuildInput<'_>,
        state: &DeviceState,
    ) -> Result<Self, Self::BuildError> {
        let buf_label = format!("{}_read_staging", input.label);
        let staging = state.create_buffer(
            &buf_label,
            &wgpu::BufferDescriptor {
                label: Some(&buf_label),
                size: input.buffer.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;

        Ok(Self {
            label: input.label.to_owned(),
            buffer: input.buffer,
            staging,
            _marker: std::marker::PhantomData,
        })
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn op_kind() -> &'static str {
        "copy_buffer"
    }
}

/// Used to make a new [`ReadDeviceBufferDesc`].
#[derive(Debug)]
pub struct ReadDeviceBufferDesc<'a> {
    label: &'a str,
    buffer: Rc<Buffer>,
}

// For convenience.
impl<'a> From<(&'a str, Rc<Buffer>)> for ReadDeviceBufferDesc<'a> {
    fn from(value: (&'a str, Rc<Buffer>)) -> Self {
        Self {
            label: value.0,
            buffer: value.1,
        }
    }
}

/// Copies data from a [`& [T]`] to a [`Buffer`].
///
/// If the slice is smaller than the buffer, the copy will stop when the copy
/// reaches the end of the buffer. If it is larger, the operation will return
/// a [`CopyBufferError`].
pub struct WriteDeviceBuffer<T> {
    label: String,
    buffer: Rc<Buffer>,
    staging: Rc<Buffer>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> DeviceOp for WriteDeviceBuffer<T>
where
    T: bytemuck::Pod,
{
    type Input<'a> = &'a [T];
    type Output = Result<(), anyhow::Error>;
    type BuildInput<'a> = WriteDeviceBufferDesc<'a>;
    type BuildError = DeviceStateError;

    /// Copy a data from `s` to the [`Buffer`].
    ///
    /// # Panics
    /// If `(Buffer.size() % std::mem::size_of<T>()) != 0` this function will
    /// panic.
    async fn device_exec(&self, state: &DeviceState, s: Self::Input<'_>) -> Self::Output {
        let slice_out_u8: &[u8] = bytemuck::cast_slice(s);
        // This will be set if device_init has been called.
        // Check that GPU buffer is at least as long as output slice.
        if self.buffer.size() < slice_out_u8.len() as BufferAddress {
            Err(CopyBufferError::SliceMismatch(
                slice_out_u8.len(),
                self.buffer.size() as usize,
            ))?;
        }

        let range = 0..(slice_out_u8.len() as BufferAddress);
        let staging_slice = self.staging.slice(range.clone());

        // Map the Buffer to CPU memory, then write to it.
        // This does NOT block on the web as the GPU is immediately polled and the
        // channel reciever async. The unwraps on the channel are safe as the channel
        // cannot be dropped.
        let (tx, rx) = flume::bounded(1);
        staging_slice.map_async(wgpu::MapMode::Write, move |r| tx.send(r).unwrap());
        state.device.poll(wgpu::Maintain::wait());
        rx.recv_async().await.unwrap()?;

        staging_slice.get_mapped_range_mut()[..].copy_from_slice(slice_out_u8);
        self.staging.unmap();

        // Submit a copy from the staging buffer to the target buffer.
        let mut encoder = state.create_encoder();
        encoder.copy_buffer_to_buffer(
            &self.staging,
            0,
            &self.buffer,
            0,
            slice_out_u8.len() as BufferAddress,
        );
        state.queue().submit([encoder.finish()]);

        Ok(())
    }

    // Simply create a staging buffer for performance and convenience.
    async fn new_in_device(
        input: Self::BuildInput<'_>,
        state: &DeviceState,
    ) -> Result<Self, Self::BuildError> {
        let buf_label = format!("{}_staging", input.label);

        let staging = state.create_buffer(
            &buf_label,
            &wgpu::BufferDescriptor {
                label: Some(&buf_label),
                size: input.buffer.size(),
                usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
                mapped_at_creation: false,
            },
        )?;

        Ok(Self {
            label: input.label.to_owned(),
            buffer: input.buffer,
            staging,
            _marker: std::marker::PhantomData,
        })
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn op_kind() -> &'static str {
        "write_buffer"
    }
}

pub struct WriteDeviceBufferDesc<'a> {
    label: &'a str,
    buffer: Rc<Buffer>,
}

impl<'a> From<(&'a str, Rc<Buffer>)> for WriteDeviceBufferDesc<'a> {
    fn from(value: (&'a str, Rc<Buffer>)) -> Self {
        Self {
            label: value.0,
            buffer: value.1,
        }
    }
}
