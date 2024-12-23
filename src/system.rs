//! Contains basic simulation structs and traits.

use std::{cell::RefCell, mem, rc::Rc};

use downcast_rs::{impl_downcast, Downcast};
use indexmap::IndexMap;
use ndarray::prelude::*;
use thiserror::Error;

use crate::Float;

/// Basic simulation operation, designed to be used by [System].
///
/// A System Op performs some kind operation on the simulation. This will
/// usually be either changing the simulation state, or performing some kind of
/// IO, like writing out the data.
///
/// Each operation will be performed once per step, in the order that they are
/// provided.
pub trait SystemOp: Downcast {
    /// Performs a single step in the main simulation loop.
    ///
    /// It is an error to run this before start has been called.
    fn exec(&mut self, data: &mut Data) -> anyhow::Result<()>;

    /// Gives a generic name describing the kind of operation.
    fn op_type(&self) -> &'static str;

    /// Performs any initialization at simulation start up.
    ///
    /// This should not be run more than once.
    fn start(&mut self, _data: &mut Data) -> anyhow::Result<()> {
        Ok(())
    }

    /// Performs any finalization at the end of a simulation.
    ///
    /// This should not be run more than once.
    fn finish(&mut self, _data: &mut Data) -> anyhow::Result<()> {
        Ok(())
    }
}
impl_downcast!(SystemOp);

/// Denotes if the arrays are up to date or dirty.
///
/// For Future GPU implementation, denotes whether the properties stored in
/// [Data] are current. Should be Clean unless a [SystemOp] makes Data dirty.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum SystemState {
    Clean,
    Dirty,
}

impl Default for SystemState {
    fn default() -> Self {
        Self::Clean
    }
}

/// Container for simulation data.
///
/// This is to prevent having to pass [System] to [SystemOp]s that complicates
/// ownership.
#[derive(Debug, Default, Clone)]
pub struct Data {
    pub(crate) state: SystemState,
    pub(crate) u: Array3<Float>,
    pub(crate) u_prev: Array3<Float>,
    pub(crate) dt: Float,
    pub(crate) step: usize,
    pub(crate) x: Array3<Float>,
    pub(crate) L: Float,
}

impl Data {
    pub(crate) fn grid(&self) -> Grid {
        let xlo = *self.x.slice(s![0, 0, 0]).into_scalar();
        let xhi = *self.x.slice(s![-1, 0, 0]).into_scalar();
        let dx = self.x.slice(s![1, 0, 0]).into_scalar() - xlo;
        Grid { dx, xlo, xhi }
    }
}

/// Contains information about the simulation grid.
pub struct Grid {
    pub(crate) dx: Float,
    pub(crate) xlo: Float,
    pub(crate) xhi: Float,
}

/// Basic struct that contains all simulation information.
#[derive(Default)]
pub struct System {
    pub(crate) data: Data,
    pub(crate) system_ops: IndexMap<String, Rc<RefCell<dyn SystemOp>>>,
}

impl System {
    /// Installs a single [SystemOp] into [System].
    ///
    /// name should be a unique identifier for the op.
    pub fn install_op(&mut self, name: &str, op: impl SystemOp + 'static) {
        let rc = Rc::new(RefCell::new(op));
        self.system_ops.insert(name.into(), rc);
    }

    /// Installs a single [SystemOp] at a particular position.
    ///
    /// This is used for when a [SystemOp] has to occur before another.
    pub fn install_op_before(&mut self, name: &str, index: usize, op: impl SystemOp + 'static) {
        self.system_ops
            .insert_before(index, name.into(), Rc::new(RefCell::new(op)));
    }

    /// Gets a [SystemOp] with a particular [name].
    fn get_op(&self, name: &str) -> Option<Rc<RefCell<dyn SystemOp>>> {
        self.system_ops.get(name).cloned()
    }

    /// Gets the velocity array (u) as an [ArrayView3].
    pub fn get_u(&self) -> ArrayView3<Float> {
        self.data.u.view()
    }

    /// Gets the velocity array (u) as an [ArrayViewMut3].
    pub fn get_u_mut(&mut self) -> ArrayViewMut3<Float> {
        self.data.u.view_mut()
    }

    /// Gets the velocity array (u) of the previous step as an [ArrayView3].
    pub fn get_u_prev(&self) -> ArrayView3<Float> {
        self.data.u_prev.view()
    }

    /// Gets the velocity array (u) of the previous step as an [ArrayViewMut3].
    pub fn get_u_prev_mut(&mut self) -> ArrayViewMut3<Float> {
        self.data.u_prev.view_mut()
    }

    /// Gets the position array (x) as a shared [ArrayView2].
    pub fn get_x(&self) -> ArrayView3<Float> {
        self.data.x.view()
    }

    /// Gets the time step size.
    pub fn dt(&self) -> Float {
        self.data.dt
    }

    /// Gets the number of steps that the simulation has run for.
    fn nstep(&self) -> usize {
        self.data.step
    }

    /// Starts a simulation, by calling [SystemOp::start] for each operation. It
    /// is a logic error for this to be called multiple times.
    pub fn start(&mut self) -> anyhow::Result<()> {
        for (_, op) in self.system_ops.iter_mut() {
            op.borrow_mut().start(&mut self.data)?;
        }
        Ok(())
    }

    /// Performs a simulation step, by calling [SystemOp::exec()] for each
    /// operation, advancing the state by dt.
    pub fn step(&mut self) -> anyhow::Result<()> {
        mem::swap(&mut self.data.u, &mut self.data.u_prev);

        self.data.u = self.data.u_prev.clone();

        for (_, op) in self.system_ops.iter_mut() {
            op.borrow_mut().exec(&mut self.data)?;
        }

        self.data.step += 1;
        Ok(())
    }

    /// Finalizes a simulation by calling [SystemOp::finish] for each operation.
    pub fn finish(&mut self) -> anyhow::Result<()> {
        for (_, op) in self.system_ops.iter_mut() {
            op.borrow_mut().finish(&mut self.data)?;
        }
        Ok(())
    }

    /// Performs a simulation run for nstep steps.
    pub fn run(&mut self, nstep: usize) -> anyhow::Result<()> {
        self.start()?;
        for _ in 0..nstep {
            self.step()?;
        }
        self.finish()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    pub use super::*;
    use crate::SystemBuilder;

    #[derive(Debug)]
    struct TestOp(&'static str);

    impl TestOp {
        fn works(&self) -> bool {
            true
        }
    }

    impl SystemOp for TestOp {
        fn start(&mut self, data: &mut Data) -> anyhow::Result<()> {
            data.u.fill(2.0);
            Ok(())
        }
        fn exec(&mut self, data: &mut Data) -> anyhow::Result<()> {
            data.u *= 2.0;
            Ok(())
        }

        fn finish(&mut self, data: &mut Data) -> anyhow::Result<()> {
            data.u += 3.0;
            Ok(())
        }

        fn op_type(&self) -> &'static str {
            "test_op"
        }
    }

    // Performs simple tests of system instantiation, setup, run and tear down.
    #[test]
    fn test_system_basics() {
        let mut system = SystemBuilder::default()
            .N(10)
            .dt(0.1)
            .L([1.0, 1.0])
            .build()
            .unwrap();

        system.install_op("test1", TestOp("one"));
        system.run(2).unwrap();

        assert_eq!(
            system.get_op("test1").unwrap().borrow().op_type(),
            "test_op"
        );

        // Checks that downcasting works and the bare object can be obtained.
        let down = system
            .get_op("test1")
            .unwrap()
            .borrow()
            .downcast_ref::<TestOp>()
            .unwrap()
            .works();
        assert!(down);
        assert_eq!(system.get_u(), Array3::from_elem([10, 10, 2], 11.0));
    }
}

/// Builder struct for [System].
pub struct SystemBuilder {
    N: Result<usize, BuildError>,
    dt: Result<Float, BuildError>,
    dims: Result<[Float; 2], BuildError>,
    system_ops: IndexMap<String, Rc<RefCell<dyn SystemOp>>>,
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self {
            N: Err(BuildError::NNotSet),
            dt: Err(BuildError::DtNotSet),
            dims: Err(BuildError::DimsNotSet),
            system_ops: IndexMap::new(),
        }
    }
}

impl SystemBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs a series of [SystemOp]s.
    pub fn install_ops<'a>(
        mut self,
        ops: impl IntoIterator<Item = (&'a str, Rc<RefCell<dyn SystemOp>>)>,
    ) -> Self {
        for (name, op) in ops {
            self.system_ops.insert(name.into(), op);
        }
        self
    }

    /// Sets the timestep, dt.
    pub fn dt(mut self, dt: Float) -> Self {
        if dt <= 0.0 || !dt.is_finite() {
            self.dt = Err(BuildError::DtInvalid);
        }
        self.dt = Ok(dt);
        self
    }

    /// Sets the number of grid points N.
    pub fn N(mut self, N: usize) -> Self {
        if N == 0 {
            self.N = Err(BuildError::NInvalid);
        }
        self.N = Ok(N);
        self
    }

    /// Sets the grid length L.
    pub fn L(mut self, dims: [Float; 2]) -> Self {
        if dims[0] <= 0.0 || dims[1] <= 0.0 || dims[0] != dims[1] {
            self.dims = Err(BuildError::DimsInvalid);
        }
        self.dims = Ok(dims);
        self
    }

    /// Builds [System].
    pub fn build(self) -> Result<System, BuildError> {
        let N = self.N?;
        let L = self.dims?[0];

        let x = make_grid(L, N);

        Ok(System {
            data: Data {
                state: SystemState::Clean,
                u: Array3::zeros([N, N, 2]),
                u_prev: Array3::zeros([N, N, 2]),
                x,
                dt: self.dt?,
                step: 0,
                L,
            },
            system_ops: self.system_ops,
        })
    }
}

/// Error type for [SystemBuilder].
#[derive(Error, Debug)]
pub enum BuildError {
    #[error("value of N must be > 0")]
    NNotSet,
    #[error("value of N must be > 0")]
    NInvalid,
    #[error("Dimensions have not been set.")]
    DimsNotSet,
    #[error("Only dimensions where x = y are currently supported.")]
    DimsInvalid,
    #[error("dt has not been set.")]
    DtNotSet,
    #[error("dt must be > 0")]
    DtInvalid,
}

/// Generates an [Array2] containing all grid positions.
///
/// The grid contains N grid points and has a length L.
pub(crate) fn make_grid(L: Float, N: usize) -> Array3<Float> {
    let mut x = Array3::zeros([N, N, 2]);

    let l_inv = 1.0 / L;
    for i in 0..N {
        for j in 0..N {
            x[(i, j, 0)] = (i as Float + 0.5) * l_inv;
            x[(i, j, 1)] = (j as Float + 0.5) * l_inv;
        }
    }
    x
}
