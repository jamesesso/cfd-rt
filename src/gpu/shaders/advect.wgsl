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
var<storage, read_write> x: array<f32>;

@group(0)
@binding(4)
var<storage, read_write> u: array<f32>;

@group(0)
@binding(5)
var<storage, read_write> u0: array<f32>;

@group(1)
@binding(0)
var<storage, read_write> x_trace: array<f32>;

// Calcualtes the position of a particle at point x at (t - dt). The entry
// point for TraceType::Linear,
@compute
@workgroup_size(32)
fn trace_x_linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    x_trace[gid.x] = x[gid.x] - u0[gid.x] * dt;
}

// performs bilinear mixing, with f11 f12 f21 and f22 being the four grid points with
// known values, t is the x weighting and s is the y weighting.
fn bi_mix(f11: f32, f12: f32, f21: f32, f22: f32, t: f32, s: f32) -> f32 {
    let fx1_mix = mix(f11, f21, t);
    let fx2_mix = mix(f12, f22, t);

    return mix(fx1_mix, fx2_mix, s);
}

// Performs bilinear interpolation of a function (i.e. u) for an array of positions
// set in x_trace.
@compute
@workgroup_size(32)
fn interp_u_linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Convert 1D indices to indicies.
    // TODO: Make universal over vector with variable z dimension.
    let idx1d = 2 * gid.x;

    // Get maximum value of x.
    let xmax = vec2<f32>(N.xy) * vec2(dx);

    let idx3d = idx(idx1d);
    let ix: u32 = idx3d.x;
    let iy: u32 = idx3d.y;
    let iz: u32 = idx3d.z;

    // Get x trace and clamp for safety.
    let xt_uc = vec2(x_trace[idx1d], x_trace[idx1d + 1]);
    let xt = clamp(xt_uc, vec2<f32>(0.0), xmax);

    // Find the index of the nearest grid point to x.
    let Nf = vec2(f32(N.x), f32(N.y));
    let xd = xt / dx; // Get where x is on the grid.
    // Clamp again to prevent overruns.
    let xlo = clamp(xd - 0.5, vec2(0.0), vec2<f32>(N.xy));

    // Split xd into whole part (grid idx) and fractional (lerp weight).
    let x_mod = modf(xlo);
    let xi = vec2<u32>(x_mod.whole);
    let t = x_mod.fract.x;
    let s = x_mod.fract.y;

    // Get nearest grid points.
    let x1i = u32(xi.x);
    let y1i = u32(xi.y);
    let x2i = x1i + 1;
    let y2i = y1i + 1;

    // Get values of f at grid points.
    let f11x = u0[get1d(vec3(x1i, y1i, 0))];
    let f12x = u0[get1d(vec3(x1i, y2i, 0))];
    let f21x = u0[get1d(vec3(x2i, y1i, 0))];
    let f22x = u0[get1d(vec3(x2i, y2i, 0))];

    let fx = bi_mix(f11x, f12x, f21x, f22x, t, s);

    // Calc fy too.
    let f11y = u0[get1d(vec3(x1i, y1i, 1))];
    let f12y = u0[get1d(vec3(x1i, y2i, 1))];
    let f21y = u0[get1d(vec3(x2i, y1i, 1))];
    let f22y = u0[get1d(vec3(x2i, y2i, 1))];

    let fy = bi_mix(f11y, f12y, f21y, f22y, t, s);

    u[idx1d] = fx;
    u[idx1d + 1] = fy;
}

// Converts a 1D array to a 3D array.
fn get1d(i: vec3<u32>) -> u32 {
    return i.x * (N.y * N.z) + i.y * N.z + i.z;
}

// Turns a 1D index into a 3D index, using N as the array strides.
fn idx(gid: u32) -> vec3<u32> {
    let x = gid / (N.y * 2);
    let y = (gid / N.z) % N.y;
    let z = gid % N.z;
    return vec3(x, y, z);
}

// Used to skip trace stage for testing.
@compute
@workgroup_size(32)
fn noop_trace(@builtin(global_invocation_id) gid: vec3<u32>) { }
