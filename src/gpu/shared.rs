//! Shared memory representations of data.
use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
};

use thiserror::Error;

use super::{prelude::*, DeviceState};

#[derive(Debug)]
enum DataState {
    Clean,
    Dirty,
}

/// Struct representing data that exists on both the GPU and the CPU at the same
/// time.
///
/// This struct can provide exclusive access to either the GPU memory or the CPU
/// memory. If [`SharedBuffer::local()`] is called, this will return a
/// [`LocalView`] that implements `Deref<Target = T>`. On drop this will write
/// to the GPU buffer. If [`SharedBuffer::device()`] is called, a [`DeviceView`]
/// is created, which implements `Deref<Target = [wgpu::Buffer]`.
#[derive(Debug)]
pub struct SharedBuffer<T> {
    data_state: DataState,
    state: Rc<DeviceState>,
    device_buffer: Rc<Buffer>,
    read_staging: Buffer,
    write_staging: Buffer,
    local_vec: Vec<T>,
}

/// View into data stored on the CPU.
pub struct LocalView<'a, T: bytemuck::Pod> {
    shared_ref: &'a mut SharedBuffer<T>,
}

/// Handle to a [`wgpu::Buffer`].
#[derive(Debug)]
pub struct DeviceView<'a, T> {
    shared_ref: &'a mut SharedBuffer<T>,
}

impl<T> SharedBuffer<T>
where
    T: bytemuck::Pod,
{
    /// The default [`wgpu::BufferUsages`] when [`new()`] is called.
    const DEFAULT_STORAGE: BufferUsages = BufferUsages::STORAGE
        .union(BufferUsages::COPY_DST)
        .union(BufferUsages::COPY_SRC);

    /// The default [`wgpu::BufferUsages`] when [`new_uniform()`] is called.
    const DEFAULT_UNIFORM: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);

    /// Creates a new `SharedBuffer` that is compatible with storage buffers.
    pub async fn new(state: Rc<DeviceState>, data: Vec<T>) -> Self {
        Self::new_with_usage(state, data, Self::DEFAULT_STORAGE).await
    }

    /// Creates a new `SharedBuffer` that is compatible with uniform buffers.
    pub async fn new_uniform(state: Rc<DeviceState>, data: Vec<T>) -> Self {
        Self::new_with_usage(state, data, Self::DEFAULT_UNIFORM).await
    }

    /// Creates a new `SharedBuffer` with a custom [`wgpu::BufferUsages`].
    pub async fn new_with_usage(state: Rc<DeviceState>, data: Vec<T>, usage: BufferUsages) -> Self {
        let buffer = state.device.create_buffer(&BufferDescriptor {
            label: None,
            size: (data.len() * std::mem::size_of::<T>()) as BufferAddress,
            usage,
            mapped_at_creation: false,
        });

        let read_staging = state.device.create_buffer(&BufferDescriptor {
            label: Some("read"),
            size: (data.len() * std::mem::size_of::<T>()) as BufferAddress,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let write_staging = state.device.create_buffer(&BufferDescriptor {
            label: Some("write"),
            size: (data.len() * std::mem::size_of::<T>()) as BufferAddress,
            usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut val = Self {
            state,
            local_vec: data,
            data_state: DataState::Clean,
            device_buffer: Rc::new(buffer),
            read_staging,
            write_staging,
        };

        val.write_buffer().await;
        val
    }

    /// Generates a new [`LocalView<T>`], i.e a view into the CPU memory.
    ///
    /// # Errors
    /// - SharedBufferError::DeviceStillAlive: This is returned if a
    ///   [`DeviceView`] needs dropping before a view on the CPU can be
    ///   obtained.
    pub async fn local(&'_ mut self) -> Result<LocalView<'_, T>> {
        // Check that no copies of device_buffer still exist. I.e. that the Rc in self
        // is the only one.
        if Rc::strong_count(&self.device_buffer) > 1 || Rc::weak_count(&self.device_buffer) > 0 {
            return Err(SharedBufferError::DeviceStillAlive);
        }

        self.read_buffer().await;
        Ok(LocalView { shared_ref: self })
    }

    /// Generates a new [`DeviceView<T>`], i.e a view into the GPU memory,
    pub async fn device(&'_ mut self) -> DeviceView<'_, T> {
        self.write_buffer().await;
        DeviceView { shared_ref: self }
    }

    /// Reads GPU memory into CPU memory.
    pub(crate) async fn read_buffer(&mut self) {
        let local_u8 = bytemuck::cast_slice_mut(&mut self.local_vec[..]);

        let mut encoder = self.state.create_encoder();

        encoder.copy_buffer_to_buffer(
            &self.device_buffer,
            0,
            &self.read_staging,
            0,
            self.read_staging.size(),
        );

        self.state.queue().submit([encoder.finish()]);

        let staging_slice = self.read_staging.slice(..);

        let (tx, rx) = flume::bounded(1);
        staging_slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.state.device.poll(wgpu::Maintain::Wait);
        rx.recv_async()
            .await
            .unwrap()
            .expect("Error reading from buffer.");
        local_u8.copy_from_slice(&staging_slice.get_mapped_range()[..]);

        self.read_staging.unmap();
        self.state.device.poll(wgpu::Maintain::Wait);
    }

    /// Writes CPU memory to the GPU.
    pub(crate) async fn write_buffer(&mut self) {
        let local_u8 = bytemuck::cast_slice(&self.local_vec[..]);
        let staging_slice = self.write_staging.slice(..local_u8.len() as u64);

        let (tx, rx) = flume::bounded(1);

        staging_slice.map_async(wgpu::MapMode::Write, move |r| tx.send(r).unwrap());
        self.state.device.poll(wgpu::Maintain::wait());
        rx.recv_async()
            .await
            .unwrap()
            .expect("Error writing to buffer");

        staging_slice.get_mapped_range_mut()[..].copy_from_slice(local_u8);
        self.write_staging.unmap();

        // Submit a copy from the staging buffer to the target buffer.
        let mut encoder = self.state.create_encoder();
        encoder.copy_buffer_to_buffer(
            &self.write_staging,
            0,
            &self.device_buffer,
            0,
            local_u8.len() as BufferAddress,
        );

        self.state.queue().submit([encoder.finish()]);
        self.state.device.poll(wgpu::Maintain::wait());
    }

    /// Returns the size of the buffer as the number of T that it can hold..
    fn size_val(&self) -> usize {
        self.local_vec.len()
    }

    /// Returns the size of the buffer in bytes.
    fn size_u8(&self) -> usize {
        std::mem::size_of::<T>() * self.local_vec.len()
    }
}

/// Error type for SharedBuffer.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum SharedBufferError {
    #[error("Trying to get LocalView while at least one DeviceView is still alive.")]
    DeviceStillAlive,
}

type Result<T> = std::result::Result<T, SharedBufferError>;

impl<T> Deref for DeviceView<'_, T> {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.shared_ref.device_buffer
    }
}

impl<T> LocalView<'_, T>
where
    T: bytemuck::Pod,
{
    /// Consumes self, writing to GPU, then dropping the `LocalView<T>`.
    async fn consume(self) {
        self.shared_ref.write_buffer().await;
    }
}

impl<T> Deref for LocalView<'_, T>
where
    T: bytemuck::Pod,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.shared_ref.local_vec[..]
    }
}

impl<T> DerefMut for LocalView<'_, T>
where
    T: bytemuck::Pod,
{
    fn deref_mut(&mut self) -> &mut <Self as Deref>::Target {
        self.shared_ref.data_state = DataState::Dirty;
        &mut self.shared_ref.local_vec[..]
    }
}

impl<T> std::fmt::Debug for LocalView<'_, T>
where
    T: std::fmt::Debug + bytemuck::Pod,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalView")
            .field("data", &self.shared_ref.local_vec)
            .finish()
    }
}

impl<T> Drop for LocalView<'_, T>
where
    T: bytemuck::Pod,
{
    fn drop(&mut self) {
        match &self.shared_ref.data_state {
            DataState::Dirty => pollster::block_on(self.shared_ref.write_buffer()),
            DataState::Clean => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Float;

    #[test_log::test(pollster::test)]
    async fn test_shared_buffer() -> anyhow::Result<()> {
        let state = Rc::new(DeviceState::new().await.unwrap());

        let test_arr = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut a_buf = SharedBuffer::new(state.clone(), test_arr.to_vec()).await;
        let mut b_buf = SharedBuffer::new(state.clone(), vec![0.0_f32; 5]).await;

        let mut encoder = state.create_encoder();
        encoder.copy_buffer_to_buffer(
            &*a_buf.device().await,
            0,
            &*b_buf.device().await,
            0,
            5 * std::mem::size_of::<Float>() as u64,
        );
        state.queue().submit([encoder.finish()]);
        state.device.poll(wgpu::Maintain::Wait);

        assert_eq!(b_buf.local().await?.deref(), &test_arr);

        // Check going the other way.
        let test_arr = [2.0, 3.0, 4.0, 5.0, 6.0];
        b_buf.local().await?.copy_from_slice(&test_arr);

        let mut encoder = state.create_encoder();
        encoder.copy_buffer_to_buffer(
            &*b_buf.device().await,
            0,
            &*a_buf.device().await,
            0,
            5 * std::mem::size_of::<Float>() as u64,
        );
        state.queue().submit([encoder.finish()]);
        state.device.poll(wgpu::Maintain::Wait);

        assert_eq!(a_buf.local().await?.deref(), &test_arr);
        Ok(())
    }
}
