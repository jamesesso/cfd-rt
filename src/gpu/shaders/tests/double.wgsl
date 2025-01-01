@group(0)
@binding(0)
var<storage, read_write> a: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> b: array<f32>;

@compute
@workgroup_size(32)
fn double(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  b[i] = 2.0 * a[i];
}

