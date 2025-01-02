//! Module for GPU calculations.
//!
//! I am not sure about the design so far, but this is designed for reuse of GPU
//! components between simulation stages.

use std::{cell::RefCell, rc::Rc};

use indexmap::{map::Entry, IndexMap};
use num::{FromPrimitive, Integer, ToPrimitive};
use thiserror::*;

use crate::gpu::prelude::*;

pub mod advect;
pub mod common;
pub mod copy;
mod prelude;

// TODO: Update when there are actually shaders that are not test shaders.

/// Holds the statically loaded WGSL modules in the format `(module_label,
/// module_string)`.
const COMPUTE_SHADER_FILES: &[(&str, &str)] = &[("advect", include_str!("shaders/advect.wgsl"))];

/// Sets the default workgroup size used in the shaders.
///
/// This is set at compile time in the shaders so it is set as a const here.
pub const WORKGROUP_SIZE: usize = 32;

/// Converts n to the nearest multiple of [`WORKGROUP_SIZE`].
///
/// This corresponds to a buffer with the correct size for the overspill from
/// the last workgroup running.
fn buffer_size<T>(n: T) -> T
where
    T: Integer + FromPrimitive + ToPrimitive,
{
    // Workgroup size is fixed, so this conversion should never fail.
    let wg: u64 = WORKGROUP_SIZE.to_u64().unwrap();
    let div: u64 = n.to_u64().unwrap().div_ceil(wg);

    T::from_u64(div * wg).unwrap()
}

/// Calculates ceil(n/[`WORKGROUP_SIZE`]).
///
/// Calculates the maximum number of workgroups required to perform a
/// calculation over an array of size `n`.
fn dispatch_workgroup_size<T>(n: T) -> T
where
    T: Integer + FromPrimitive + ToPrimitive,
{
    // Workgroup size is fixed, so this conversion should never fail.
    let wg: u64 = WORKGROUP_SIZE.to_u64().unwrap();
    let div: u64 = n.to_u64().unwrap().div_ceil(wg);

    T::from_u64(div).unwrap()
}

/// Struct that owns all the GPU objects.
///
/// This contains all the types required to interact with the GPU. As the WebGPU
/// types use interior mutability it makes sense to do so here too. Therefore
/// `DeviceState` uses interior mutability too and is used a shared reference
/// everywhere.
#[derive(Debug)]
pub struct DeviceState {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    features: wgpu::Features,
    device: wgpu::Device,
    queue: Queue,
    compute_pipelines: RefCell<SMapRc<ComputePipeline>>,
    buffers: RefCell<SMapRc<Buffer>>,
    bind_groups: RefCell<SMapRc<BindGroup>>,
    bind_group_layouts: RefCell<SMapRc<BindGroupLayout>>,
    shaders: RefCell<SMap<ShaderModule>>,
}

impl DeviceState {
    async fn new() -> Result<Self, DeviceStateError> {
        let instance = wgpu::Instance::new(Default::default());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .ok_or(DeviceStateError::NoAdapterFromInstance)?;
        let features = adapter.features();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features & wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: Default::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let resources = Self {
            instance,
            adapter,
            features,
            device,
            queue,
            compute_pipelines: Default::default(),
            buffers: Default::default(),
            bind_groups: Default::default(),
            bind_group_layouts: Default::default(),
            shaders: Default::default(),
        };

        resources.load_default_shaders();
        Ok(resources)
    }

    // I didn't want to expose mutable references to the IndexMaps just in case it
    // makes it too easy to mutate whats inside and cause GPU calculations to go
    // badly wrong.
    //
    /// Gets an [`SMap`] of all the [`Buffer`]s in use.
    fn buffer(&self, label: &str) -> Option<Rc<Buffer>> {
        self.buffers.borrow().get(label).cloned()
    }

    /// Gets an [`SMap`] of all the [`BindGroup`]s in use.
    fn bind_group(&self, label: &str) -> Option<Rc<BindGroup>> {
        self.bind_groups.borrow().get(label).cloned()
    }

    /// Gets an [`SMap`] of all the [`ComputePipeline`]s in use.
    fn compute_pipeline(&self, label: &str) -> Option<Rc<ComputePipeline>> {
        self.compute_pipelines.borrow().get(label).cloned()
    }

    /// Gets an [`SMap`] of all the [`BindGroupLayout`]s in use.
    fn bind_group_layout(&self, label: &str) -> Option<Rc<BindGroupLayout>> {
        self.bind_group_layouts.borrow().get(label).cloned()
    }

    /// Gets the [`Queue`] produced from [`wgpu::Device`].
    fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Creates a [`CommandEncoder`] using [`wgpu::Device`].
    fn create_encoder(&self) -> CommandEncoder {
        self.device
            .create_command_encoder(&CommandEncoderDescriptor { label: None })
    }

    /// Creates a [`Buffer`] and returns a handle to it.
    fn create_buffer(
        &self,
        label: &str,
        buffer_desc: &BufferDescriptor,
    ) -> Result<Rc<Buffer>, DeviceStateError> {
        let buffer = Rc::new(self.device.create_buffer(buffer_desc));
        self.buffers
            .borrow_mut()
            .insert_if_empty(label, buffer.clone())
            .cloned()
            .ok_or(DeviceStateError::BufferAlreadyAllocated(label.to_string()))
    }

    /// Creates a [`BindGroupLayout`] with [`wgpu::Device`] and returns a handle
    /// to it.
    fn create_bind_group_layout(
        &self,
        label: &str,
        layout_desc: &BindGroupLayoutDescriptor,
    ) -> Result<Rc<BindGroupLayout>, DeviceStateError> {
        let layout = Rc::new(self.device.create_bind_group_layout(layout_desc));
        self.bind_group_layouts
            .borrow_mut()
            .insert_if_empty(label, layout.clone())
            .cloned()
            .ok_or(DeviceStateError::BindGroupLayoutAlreadyAllocated(
                label.to_string(),
            ))
    }

    /// Creates a [`BindGroup`] with [`wgpu::Device`] and returns a handle to
    /// it.
    fn create_bind_group(
        &self,
        label: &str,
        bind_group_desc: &BindGroupDescriptor,
    ) -> Result<Rc<BindGroup>, DeviceStateError> {
        let bind_group = Rc::new(self.device.create_bind_group(bind_group_desc));
        self.bind_groups
            .borrow_mut()
            .insert_if_empty(label, bind_group.clone())
            .cloned()
            .ok_or(DeviceStateError::BindGroupAlreadyAllocated(
                label.to_string(),
            ))
    }

    /// Creates a set of [`Buffers`] and a [`BindGroup`] and [`BindGroupLayout`]
    /// containing all of them.
    ///
    /// This is a convenience function as commonly a [`DeviceOp`] wants a series
    /// of new buffers with a respective [`BindGroup`] and
    /// [`BindGroupLayout`]. The return type [`ResourceBundle`]
    /// is just a wrapper type.
    fn create_resources(
        &self,
        label: &str,
        buf_desc: &[BufferDescriptor<'_>],
    ) -> Result<ResourceBundle, DeviceStateError> {
        let mut layout_entries = Vec::with_capacity(buf_desc.len());
        let mut buf_vec = Vec::with_capacity(buf_desc.len());

        for (n, b) in buf_desc.iter().enumerate() {
            // Create and push Buffer to output Vec.
            let buf =
                self.create_buffer(b.label.ok_or(DeviceStateError::UnnamedBufferCreation)?, b)?;
            buf_vec.push(buf.clone());

            let ty = match b.usage.intersects(BufferUsages::UNIFORM) {
                true => wgpu::BufferBindingType::Uniform,
                false => wgpu::BufferBindingType::Storage { read_only: false },
            };

            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: n as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: Some(b.size.try_into().unwrap()),
                },
                count: None,
            });
        }

        // This needs to be outside the loop to keep the borrow checker happy.
        let bg_entries: Vec<_> = buf_vec
            .iter()
            .enumerate()
            .map(|(n, b)| BindGroupEntry {
                binding: n as u32,
                resource: b.as_entire_binding(),
            })
            .collect();

        let layout_desc = &BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &layout_entries[..],
        };

        let layout = self.create_bind_group_layout(label, layout_desc)?;

        let bind_group = self.create_bind_group(
            label,
            &BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &bg_entries[..],
            },
        )?;

        Ok(ResourceBundle {
            buffers: buf_vec,
            bind_group_layout: layout,
            bind_group,
        })
    }

    /// Creates a [`ComputePipeline`] with [`wgpu::Device`] and returns a handle
    /// to it.
    ///
    /// `entry_point` is the entry point declared in the Shader `.wgsl` file,
    /// `module_label` is the label of the module as stored in the current
    /// `DeviceState`.
    fn create_compute_pipeline(
        &self,
        label: &str,
        entry_point: &str,
        module_label: &str,
        bind_group_layouts: &[&BindGroupLayout],
    ) -> Result<Rc<ComputePipeline>, DeviceStateError> {
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts,
                push_constant_ranges: &[],
            });

        let binding = self.shaders.borrow();
        // This error cannot occur unless a requested shader file is not found. As the
        // name-shader string pairs are generated at compile time this should never
        // fail.
        let module = binding.get(module_label)
            .expect("Could not find requested shader. This error should never occur. Please report as a bug :).");

        let pipeline = Rc::new(
            self.device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );

        self.compute_pipelines
            .borrow_mut()
            .insert(label.to_owned(), pipeline.clone());

        Ok(pipeline)
    }

    /// Loads the default shaders as described by [`COMPUTE_SHADER_FILES`].
    fn load_default_shaders(&self) {
        for (label, shader_path) in COMPUTE_SHADER_FILES {
            let module = self.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl((*shader_path).into()),
            });
            self.shaders
                .borrow_mut()
                .insert_if_empty(label, module)
                .expect("Could not load default shader. This error should not occur!");
        }
    }

    /// Adds a new shader, where `label` is the label to be used in
    /// [`create_compute_pipeline`] and shader is the `WGSL` *code* (not the
    /// path) as a string.
    fn add_shader(&self, label: &str, shader: &str) -> Result<(), DeviceStateError> {
        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(shader.into()),
        });
        self.shaders
            .borrow_mut()
            .insert_if_empty(label, module)
            .ok_or(DeviceStateError::ComputePipelineAlreadyAllocatead(
                label.to_string(),
            ))?;

        Ok(())
    }
}

/// Wrapper return type for [`DeviceState::create_resources`].
#[derive(Debug)]
pub(crate) struct ResourceBundle {
    pub buffers: Vec<Rc<Buffer>>,
    pub bind_group_layout: Rc<BindGroupLayout>,
    pub bind_group: Rc<BindGroup>,
}

/// Struct that owns a [`DeviceState`] and references to all the [`DeviceOp`]
/// that are in use so they can be found later in the simulation run.
pub struct DeviceInstance {
    state: Rc<DeviceState>,
    ops: SMapRc<dyn DynDeviceOp>,
}

impl DeviceInstance {
    pub async fn new() -> anyhow::Result<Self> {
        Ok(Self {
            state: Rc::new(DeviceState::new().await?),
            ops: Default::default(),
        })
    }

    /// Installs an particular [`DeviceOp`] into `DeviceInstance`.
    ///
    /// When doing so, this runs the [`DeviceOp::new_in_device()`] function to
    /// set up Device resources etc.
    pub async fn add_op<T: DeviceOp + 'static>(
        &mut self,
        op_input: T::BuildInput<'_>,
    ) -> Result<DeviceOpHandle<T>, DeviceStateError> {
        let op: Rc<T> = Rc::new(
            T::new_in_device(op_input, &self.state)
                .await
                .map_err(anyhow::Error::from)?,
        );

        self.ops
            .insert(op.label(), op.clone() as Rc<dyn DynDeviceOp>);

        Ok(DeviceOpHandle {
            state: self.state.clone(),
            op: op.clone(),
        })
    }
}

/// A handle for an [`DeviceOp`] that has been installed into a device. This is
/// returned the type returned from [`DeviceInstance::add_op()`].
#[derive(Debug)]
pub struct DeviceOpHandle<T> {
    state: Rc<DeviceState>,
    op: Rc<T>,
}

impl<T> DeviceOpHandle<T>
where
    T: DeviceOp,
{
    /// Calls [`<T as DeviceOp>::device_exec`] using the [`DeviceState`] that
    /// the the handle owns.
    async fn device_exec(&self, input: <T as DeviceOp>::Input<'_>) -> <T as DeviceOp>::Output {
        self.op.device_exec(&self.state, input).await
    }
}

pub type SMap<T> = IndexMap<String, T>;
pub type SMapRc<T> = SMap<Rc<T>>;

trait SMapExt<T> {
    fn insert_if_empty(&mut self, label: &str, val: T) -> Option<&mut T>;
}

impl<T> SMapExt<T> for SMap<T> {
    fn insert_if_empty(&mut self, label: &str, val: T) -> Option<&mut T> {
        match self.entry(label.to_string()) {
            Entry::Vacant(entry) => Some(entry.insert(val)),
            Entry::Occupied(_) => None,
        }
    }
}

/// Trait representing a generic operation using a "Device" (i.e. a GPU).
///
/// The key method is [`Self::device_exec()`], which is intended to perform the
/// chosen calculation. [`Self::device_exec()`] contains a [`Self::Input`],
/// which allows an arbitraty input (potentially a reference) type to be used as
/// an input with and a [`Self::Output`] which allows a return type to returned
/// from the operation.
///
/// This trait currently uses a GAT for [`Self::Input`] to allow reference types
/// to be used. This can be used, for example to pass a mutable slice as an
/// input to perform copies to a GPU buffer.
///
/// The [`Self::new_in_device()`] method to build a new instance of a `DeviceOp`
/// using a [`DeviceInstance`]. The resulting `DeviceOp` can be configured using
/// [`Self::BuildInput`] as an input. I may change this to a seperate builder
/// trait instead.
pub trait DeviceOp: Sized {
    /// Used as an input argument of [`Self::device_exec()`].
    ///
    /// For example may be `&[T]`, `Array` or `()`.
    type Input<'a>
    where
        Self: 'a;
    /// Used as an ouput argument of [`Self::device_exec()`].
    ///
    /// This is generally a [`Result`] of some kind.
    type Output;
    /// Used as an input argument of [`Self::new_in_device()`].
    ///
    /// This is very similar to [`Self::Input`] and usually will be some kind of
    /// simple descriptor struct.
    type BuildInput<'a>
    where
        Self: 'a;

    // Not sure that this is the best choice w.r.t. using anyhow::Error and
    // conversion between the two. I anticipate that this can generally just be
    // a [`DeviceStateError`] normally.
    //
    /// Error type returned by the [`Self::new_in_device()`] method.
    type BuildError: std::error::Error + Send + Sync;

    /// Executes the operation on a device.
    ///
    /// This is designed to be, for example a mathematical operation or an event
    /// such as copying a GPU buffer to an array.
    async fn device_exec(&self, state: &DeviceState, input: Self::Input<'_>) -> Self::Output;

    /// Builds a new instance of a `DeviceOp` using a particular WebGPU device.
    ///
    /// This is generally should not be used directly and should be called by
    /// [`DeviceInstance`] instead. It is intended that this function be
    /// used for things like the creation of new WebGPU [`Buffer`],
    /// [`BindGroup`], [`ComputePipeline`] etc...
    async fn new_in_device(
        input: Self::BuildInput<'_>,
        state: &DeviceState,
    ) -> Result<Self, Self::BuildError>;

    /// Returns the label of an operation.
    ///
    /// This label should be specific to that particular instance of an
    /// operation. This means that two operations of the same time should
    /// have different labels.
    fn label(&self) -> String;

    /// Returns the name of the [`DeviceOp`] type. This should be the same for
    /// every instance of a particular type.
    fn op_kind() -> &'static str;
}

/// Extension trait used for storing a trait object of a [`DeviceOp`].
trait DynDeviceOp {
    fn dyn_label(&self) -> String;
    fn dyn_op_kind(&self) -> &'static str;
}

impl<T: DeviceOp> DynDeviceOp for T {
    fn dyn_label(&self) -> String {
        self.label()
    }

    fn dyn_op_kind(&self) -> &'static str {
        Self::op_kind()
    }
}

/// Error type for a GPU device.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum DeviceStateError {
    #[error("Could not create WebGPU device from instance.")]
    NoAdapterFromInstance,
    #[error("Could not create WebGPU device from adapter.")]
    RequestAdapter(#[from] wgpu::RequestDeviceError),
    #[error("Buffer called {0} already allocated.")]
    BufferAlreadyAllocated(String),
    #[error("BindGroup called {0} already allocated.")]
    BindGroupAlreadyAllocated(String),
    #[error("BindGroupLayout called {0} already allocated.")]
    BindGroupLayoutAlreadyAllocated(String),
    #[error("ComputePipline called {0} already allocated.")]
    ComputePipelineAlreadyAllocatead(String),
    #[error("Could not install DeviceOp due to error from DeviceOp::exec_init()")]
    DeviceOpInstall(#[from] anyhow::Error),
    #[error("Unnamed buffer passed to create_resources")]
    UnnamedBufferCreation,
}

#[cfg(test)]
mod tests {
    use approx::*;
    use ndarray::prelude::*;
    use test_log::test;

    use super::*;
    use crate::Float;

    #[test]
    fn test_buffer_size() {
        assert_eq!(32, buffer_size(1));
        assert_eq!(32, buffer_size(16));
        assert_eq!(32, buffer_size(31));
        assert_eq!(32, buffer_size(32));

        assert_eq!(64, buffer_size(33));
        assert_eq!(64, buffer_size(63));
        assert_eq!(64, buffer_size(64));
        assert_eq!(64, buffer_size(64));

        assert_eq!(1024, buffer_size(1023));
        assert_eq!(1024, buffer_size(1024));
        assert_eq!(1056, buffer_size(1025));
    }

    #[test]
    fn test_workgroup_num() {
        assert_eq!(1, dispatch_workgroup_size(1));
        assert_eq!(1, dispatch_workgroup_size(31));
        assert_eq!(1, dispatch_workgroup_size(32));

        assert_eq!(2, dispatch_workgroup_size(33));
        assert_eq!(2, dispatch_workgroup_size(63));
        assert_eq!(2, dispatch_workgroup_size(64));

        assert_eq!(3, dispatch_workgroup_size(65));

        assert_eq!(32, dispatch_workgroup_size(1023));
        assert_eq!(32, dispatch_workgroup_size(1024));

        assert_eq!(33, dispatch_workgroup_size(1025));
    }

    // -----------------------------------------------------------------
    // The next few tests check the basic functionality of DeviceState.
    // -----------------------------------------------------------------
    #[allow(unused_braces)]
    #[test_log::test(pollster::test)]
    async fn test_device_state_buffer() -> anyhow::Result<()> {
        let state = DeviceState::new()
            .await
            .expect("Could not build GPU device. wgpu is likely incompatable on this system.");

        let mut buffer_desc = BufferDescriptor {
            label: Some("test"),
            size: 100,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        };

        // Test making a Buffer and test.
        let buffer1 = state.create_buffer("test1", &buffer_desc)?;
        assert_eq!(buffer1.size(), 100);

        // Make a second Buffer and test.
        buffer_desc.size = 125;
        let buffer2 = state.create_buffer("test2", &buffer_desc)?;
        assert_eq!(buffer2.size(), 125);
        assert_eq!(state.buffer("test2").unwrap(), buffer2);

        assert!(state.buffer("test3").is_none());
        assert!(state.create_buffer("test1", &buffer_desc).is_err());

        Ok(())
    }

    #[allow(unused_braces)] // Fixes clippy issue with test macros.
    #[test_log::test(pollster::test)]
    async fn test_device_state_bind_group_layout() -> anyhow::Result<()> {
        let state = DeviceState::new()
            .await
            .expect("Could not build GPU device. wgpu is likely incompatable on this system.");

        let bg_desc = BindGroupLayoutDescriptor {
            label: Some("test"),
            entries: &[],
        };

        // Test making a Buffer.
        let bg1 = state.create_bind_group_layout("test1", &bg_desc)?;
        let bg2 = state.create_bind_group_layout("test2", &bg_desc)?;

        assert_eq!(state.bind_group_layout("test1").unwrap(), bg1);
        assert_eq!(state.bind_group_layout("test2").unwrap(), bg2);

        assert!(state.bind_group_layout("test3").is_none());
        assert!(state.create_bind_group_layout("test1", &bg_desc).is_err());

        Ok(())
    }

    #[allow(unused_braces)] // Fixes clippy issue with test macros.
    #[test_log::test(pollster::test)]
    async fn test_device_state_bind_group() -> anyhow::Result<()> {
        let state = DeviceState::new().await.expect(
            "Could not build GPU device. wgpu is likely incompatable on
            this system.",
        );

        let layout = state.create_bind_group_layout(
            "test",
            &BindGroupLayoutDescriptor {
                label: Some("test"),
                entries: &[],
            },
        )?;

        let bg_desc = BindGroupDescriptor {
            label: Some("test"),
            layout: &layout,
            entries: &[],
        };

        // Test making a Buffer.
        let bg1 = state.create_bind_group("test1", &bg_desc)?;
        let bg2 = state.create_bind_group("test2", &bg_desc)?;

        assert_eq!(state.bind_group("test1").unwrap(), bg1);
        assert_eq!(state.bind_group("test2").unwrap(), bg2);

        assert!(state.bind_group("test3").is_none());
        assert!(state.create_bind_group("test1", &bg_desc).is_err());

        Ok(())
    }

    // Note that this on its own cannot test BindGroup and BindGroupLayout very
    // well, due to these types being totally opaque.
    #[allow(unused_braces)] // Fixes clippy issue with test macros.
    #[test_log::test(pollster::test)]
    async fn test_device_state_create_resources() -> anyhow::Result<()> {
        let state = DeviceState::new().await.expect(
            "Could not build GPU device. wgpu is likely incompatable on
            this system.",
        );

        let mut test_bufs = [
            BufferDescriptor {
                label: Some("test"),
                size: 100,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
            BufferDescriptor {
                label: Some("test2"),
                size: 150,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
            BufferDescriptor {
                label: Some("test3"),
                size: 125,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        ];

        let res = state.create_resources("test", &test_bufs)?;

        assert_eq!(res.buffers[0].size(), 100);
        assert_eq!(res.buffers[1].size(), 150);
        assert_eq!(res.buffers[2].size(), 125);

        assert_eq!(res.bind_group, state.bind_group("test").unwrap());
        assert_eq!(
            res.bind_group_layout,
            state.bind_group_layout("test").unwrap()
        );

        test_bufs[0].label = None;

        assert!(state.create_resources("test", &test_bufs).is_err());

        Ok(())
    }

    // ---------------------------------------------------------------------------
    // This test runs the simplest "front to back" GPU operation I can think of.
    // This doubles a set of numbers in a buffer.
    // ---------------------------------------------------------------------------
    #[allow(unused_braces)] // Fixes clippy issue with test macros.
    #[test_log::test(pollster::test)]
    async fn test_device_op_simple() -> anyhow::Result<()> {
        let mut instance = DeviceInstance::new().await?;
        // This is the op that doubles the set of numbers.
        let test_op = instance.add_op::<TestDeviceOp>(()).await?;
        let a_buf = instance.state.buffer("a").unwrap();
        let b_buf = instance.state.buffer("b").unwrap();

        let write_a = instance
            .add_op::<copy::WriteDeviceBuffer<Float>>(("write_a", a_buf.clone()).into())
            .await?;
        let read_b = instance
            .add_op::<copy::ReadDeviceBuffer<Float>>(("read_b", b_buf.clone()).into())
            .await?;

        // Check buffer sizes == div_ceil(50 * float_len, 32).
        let expected_buf_len = 32 * (4 * 50).div_ceil(&32);
        assert_eq!(a_buf.size(), expected_buf_len);
        assert_eq!(b_buf.size(), expected_buf_len);

        // Make an array of the series 0, 1, 2, 3, ..., 49 and copy to GPU.
        let test_arr = Array1::from_iter((0..50).map(|x| x as Float));
        write_a.device_exec(test_arr.as_slice().unwrap()).await?;

        // Execute shader that doubles test_arr. On GPU b = 0, 2, 4, ..., 96, 98.
        test_op.device_exec(()).await;

        // Read from GPU. Now test_arr is the same as b.
        let mut out_arr = Array1::zeros(50);
        read_b.device_exec(out_arr.as_slice_mut().unwrap()).await?;

        // Now test the result.
        let test_arr = Array1::from_iter((0..50).map(|x| 2.0 * x as Float));
        assert_relative_eq!(test_arr, out_arr);

        let mut too_long_arr = Array1::from(vec![0.0; 58]);

        // Check that buffers cannot use slices that are larger than the buffer.
        assert!(write_a
            .device_exec(too_long_arr.as_slice().unwrap())
            .await
            .is_err());
        assert!(read_b
            .device_exec(too_long_arr.as_slice_mut().unwrap())
            .await
            .is_err());
        Ok(())
    }

    // This is a minimal GPU operation that simply doubles a number on the GPU for
    // testing.
    struct TestDeviceOp {
        pub a_buf: Rc<Buffer>,
        pub b_buf: Rc<Buffer>,
        bind_group: Rc<BindGroup>,
        bind_group_layout: Rc<BindGroupLayout>,
        pipeline: Rc<ComputePipeline>,
        n: usize,
    }

    impl DeviceOp for TestDeviceOp {
        type Input<'a> = ();
        type Output = ();
        type BuildInput<'a> = ();
        type BuildError = DeviceStateError;

        async fn new_in_device(
            _input: Self::BuildInput<'_>,
            state: &DeviceState,
        ) -> Result<Self, Self::BuildError> {
            let buf_size = buffer_size(50 * std::mem::size_of::<Float>()) as u64;
            let a_buf = state.create_buffer(
                "a",
                &wgpu::BufferDescriptor {
                    label: Some("a"),
                    size: buf_size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )?;

            let b_buf = state.create_buffer(
                "b",
                &wgpu::BufferDescriptor {
                    label: Some("b"),
                    size: buf_size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            )?;

            let bind_group_layout = state.create_bind_group_layout(
                "test",
                &BindGroupLayoutDescriptor {
                    label: Some("test"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                },
            )?;

            let bind_group = state.create_bind_group(
                "test",
                &BindGroupDescriptor {
                    label: Some("test"),
                    layout: &bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: a_buf.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: b_buf.as_entire_binding(),
                        },
                    ],
                },
            )?;

            state.add_shader("double", include_str!("shaders/tests/double.wgsl"))?;
            let pipeline =
                state.create_compute_pipeline("test", "double", "double", &[&bind_group_layout])?;

            Ok(Self {
                a_buf,
                b_buf,
                bind_group,
                bind_group_layout,
                pipeline,
                n: 50,
            })
        }

        async fn device_exec(&self, state: &DeviceState, _input: Self::Input<'_>) -> Self::Output {
            let mut encoder = state.create_encoder();
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, Some(&*self.bind_group), &[]);
                pass.set_pipeline(&self.pipeline);
                pass.dispatch_workgroups(dispatch_workgroup_size(self.n as u32), 1, 1);
            }

            state.queue().submit([encoder.finish()]);
        }

        fn label(&self) -> String {
            "test".to_string()
        }

        fn op_kind() -> &'static str {
            "test"
        }
    }
}
