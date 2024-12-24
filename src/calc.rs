use ndarray::prelude::*;

use crate::{Array2F, ArrayBase2F, DataF, Float};

/// Calculates the Laplacian of a on a grid with a spacing dx.
pub(crate) fn laplacian<S: DataF>(f: &ArrayBase2F<S>, dx: Float) -> Array2F {
    let width = f.shape();
    let dfdx = &f.slice(s![..(width[1] - 2), 1..-1]) - 2.0 * &f.slice(s![1..-1, 1..-1])
        + &f.slice(s![2.., 1..-1]);
    let dfdy = &f.slice(s![1..-1, ..(width[1] - 2)]) - 2.0 * &f.slice(s![1..-1, 1..-1])
        + &f.slice(s![1..-1, 2..]);

    let out_dim: [usize; 2] = f.shape()[0..2].try_into().unwrap();
    let mut out = Array2::zeros(out_dim);
    // let lap = (dfdx + dfdy) / dx.powi(2);
    out.slice_mut(s![1..-1, 1..-1]).assign(&lap);

    out
}
