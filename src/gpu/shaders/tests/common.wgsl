@group(0)
@binding(0)
var<uniform> dt: f32;

@group(0)
@binding(1)
var<uniform> dx: f32;

@group(0)
@binding(2)
var<uniform> N: vec3<u32>;

@group(0)
@binding(3)
var<storage, read_write> u: array<f32>;

@group(0)
@binding(4)
var<storage, read_write> u0: array<f32>;

// Turns a 1D index into a 3D index, using N as the array strides.
fn idx(gid: u32) -> vec3<u32> {
    let x = gid / (N.y * 2);
    let y = (gid / N.z) % N.y;
    let z = gid % N.z;
    return vec3(x, y, z);
}

// Checks that idx works correctly.
@compute
@workgroup_size(32)
fn index_test(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid.x);
    var out = 0.0;
    switch i.z {
        case 0u: { out = 2.0 * f32(i.x); }
        case 1u: { out = 3.0 * f32(i.y); }
        default: { out = 10000.0; }
    }
    u[gid.x] = out;
}

// Checks that u and u0 are mapped correctly by copying the data between the two
@compute
@workgroup_size(32)
fn double_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    u[gid.x] = 2.0 * u0[gid.x];
}

// Checks that dx and dt are mapped correctly.
@compute
@workgroup_size(32)
fn dx_dt_test(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid.x);
    var out = 0.0;
    switch i.z {
        case 0u: { out = dt; }
        case 1u: { out = dx; }
        default: { out = 10000.0; }
    }
    u[gid.x] = out;
}
