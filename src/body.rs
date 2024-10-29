//! Body forces.

use crate::{system::Data, Array3F, ArrayBase3F, DataF, DataMutF, Float, SystemOp};

/// Trait representing a body force.
///
/// Represents a body force for example a click or gravity. Can be applied to u
/// to produce effects such as dragging with a mouse.
pub trait BodyForce {
    /// Calculates the force as a function of the position.
    fn f<S: DataF>(&self, x: &ArrayBase3F<S>) -> Array3F;
    /// Applies the force to a u array.
    fn apply_body<SU: DataMutF, SX: DataF>(
        &mut self,
        u: &mut ArrayBase3F<SU>,
        x: &ArrayBase3F<SX>,
        dt: Float,
        _step: usize,
    ) {
        let u_tot = &*u + &self.f(x) * dt;
        u.assign(&u_tot)
    }
    fn name(&self) -> &'static str;
}

/// [BodyForce] that applies a constant force (f) to each grid point.
///
/// Accepts an array so that the force can be different at each point.
pub struct ConstantBodyForce {
    f: Array3F,
}

impl ConstantBodyForce {
    pub fn new(f: Array3F) -> Self {
        Self { f }
    }
}

impl BodyForce for ConstantBodyForce {
    fn f<S: DataF>(&self, _x: &ArrayBase3F<S>) -> Array3F {
        self.f.clone()
    }
    fn name(&self) -> &'static str {
        "constant_body_force"
    }
}

/// A [BodyForce] that is only applied on the initial step.
///
/// This can be used to animate a simulation with some initial conditions.
pub struct InitialBodyForce {
    f: Array3F,
}

impl InitialBodyForce {
    pub fn new(f: Array3F) -> Self {
        Self { f }
    }
}

impl BodyForce for InitialBodyForce {
    fn f<S: DataF>(&self, _x: &ArrayBase3F<S>) -> Array3F {
        self.f.clone()
    }

    // TODO: Is there a better way of doing this?
    fn apply_body<SU: DataMutF, SX: DataF>(
        &mut self,
        u: &mut ArrayBase3F<SU>,
        x: &ArrayBase3F<SX>,
        dt: Float,
        step: usize,
    ) {
        if step == 0 {
            let u_tot = &*u + &self.f(x) * dt;
            u.assign(&u_tot)
        }
    }

    fn name(&self) -> &'static str {
        "initial_body_force"
    }
}

impl<B: BodyForce + 'static> SystemOp for B {
    fn exec(&mut self, data: &mut Data) -> anyhow::Result<()> {
        self.apply_body(&mut data.u, &data.x, data.dt, data.step);

        Ok(())
    }

    fn op_type(&self) -> &'static str {
        self.name()
    }
}
