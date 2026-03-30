const std = @import("std");
const builtin = @import("builtin");

/// SIMD width selection based on target architecture.
/// x86_64: 16 floats (AVX-512 = 512-bit) — ann-benchmarks target.
/// aarch64: 8 floats (2x NEON 128-bit ops) — M1 development target.
/// fallback: 4 floats (conservative).
const default_simd_width = switch (builtin.cpu.arch) {
    .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) 16 else 8,
    .aarch64 => 8,
    else => 4,
};

/// Squared Euclidean distance. Fully unrolled SIMD via inline for.
pub fn L2Squared(comptime dim: u32) type {
    return DistanceImplWith(dim, .l2, default_simd_width);
}

/// Negative inner product (smaller = more similar, consistent with L2 ordering).
pub fn InnerProduct(comptime dim: u32) type {
    return DistanceImplWith(dim, .ip, default_simd_width);
}

/// Exposed for benchmarking different SIMD widths. Not part of the public API.
pub fn L2SquaredWith(comptime dim: u32, comptime sw: u32) type {
    return DistanceImplWith(dim, .l2, sw);
}

fn DistanceImplWith(comptime dim: u32, comptime metric: enum { l2, ip }, comptime sw: u32) type {
    const full_iters = dim / sw;
    const remainder = dim % sw;

    // Dual accumulators break the FMA loop-carried dependency.
    // FMA latency=4, throughput=1/2 on M1. One accumulator wastes half
    // the throughput. Two independent chains saturate the FMA unit.
    const half_iters = full_iters / 2;
    const has_odd_iter = (full_iters % 2) != 0;

    return struct {
        pub inline fn distance(a: *const [dim]f32, b: *const [dim]f32) f32 {
            var acc0: @Vector(sw, f32) = @splat(0.0);
            var acc1: @Vector(sw, f32) = @splat(0.0);

            inline for (0..half_iters) |i| {
                const va0: @Vector(sw, f32) = a[(2 * i) * sw ..][0..sw].*;
                const vb0: @Vector(sw, f32) = b[(2 * i) * sw ..][0..sw].*;
                acc0 = accumulate(acc0, va0, vb0);

                const va1: @Vector(sw, f32) = a[(2 * i + 1) * sw ..][0..sw].*;
                const vb1: @Vector(sw, f32) = b[(2 * i + 1) * sw ..][0..sw].*;
                acc1 = accumulate(acc1, va1, vb1);
            }

            // Handle odd remaining SIMD iteration
            if (has_odd_iter) {
                const va: @Vector(sw, f32) = a[(full_iters - 1) * sw ..][0..sw].*;
                const vb: @Vector(sw, f32) = b[(full_iters - 1) * sw ..][0..sw].*;
                acc0 = accumulate(acc0, va, vb);
            }

            var result = @reduce(.Add, acc0 + acc1);

            inline for (0..remainder) |i| {
                const idx = full_iters * sw + i;
                result += scalarOp(a[idx], b[idx]);
            }

            return finalize(result);
        }

        pub inline fn distanceWithBound(
            a: *const [dim]f32,
            b: *const [dim]f32,
            upper_bound: f32,
        ) ?f32 {
            var acc0: @Vector(sw, f32) = @splat(0.0);
            var acc1: @Vector(sw, f32) = @splat(0.0);

            inline for (0..half_iters) |i| {
                const va0: @Vector(sw, f32) = a[(2 * i) * sw ..][0..sw].*;
                const vb0: @Vector(sw, f32) = b[(2 * i) * sw ..][0..sw].*;
                acc0 = accumulate(acc0, va0, vb0);

                const va1: @Vector(sw, f32) = a[(2 * i + 1) * sw ..][0..sw].*;
                const vb1: @Vector(sw, f32) = b[(2 * i + 1) * sw ..][0..sw].*;
                acc1 = accumulate(acc1, va1, vb1);

                // Early exit check every 4 dual iterations (64 floats)
                if (metric == .l2 and comptime (i + 1) % 4 == 0) {
                    if (@reduce(.Add, acc0 + acc1) > upper_bound) return null;
                }
            }

            if (has_odd_iter) {
                const va: @Vector(sw, f32) = a[(full_iters - 1) * sw ..][0..sw].*;
                const vb: @Vector(sw, f32) = b[(full_iters - 1) * sw ..][0..sw].*;
                acc0 = accumulate(acc0, va, vb);
            }

            var result = @reduce(.Add, acc0 + acc1);

            inline for (0..remainder) |i| {
                const idx = full_iters * sw + i;
                result += scalarOp(a[idx], b[idx]);
            }

            const final = finalize(result);
            return if (final > upper_bound) null else final;
        }

        inline fn accumulate(
            acc: @Vector(sw, f32),
            va: @Vector(sw, f32),
            vb: @Vector(sw, f32),
        ) @Vector(sw, f32) {
            return switch (metric) {
                .l2 => blk: {
                    const diff = va - vb;
                    break :blk @mulAdd(@Vector(sw, f32), diff, diff, acc);
                },
                .ip => @mulAdd(@Vector(sw, f32), va, vb, acc),
            };
        }

        inline fn scalarOp(a_val: f32, b_val: f32) f32 {
            return switch (metric) {
                .l2 => (a_val - b_val) * (a_val - b_val),
                .ip => a_val * b_val,
            };
        }

        inline fn finalize(raw: f32) f32 {
            return switch (metric) {
                .l2 => raw,
                .ip => -raw, // negate so smaller = more similar
            };
        }
    };
}

/// In-place L2 normalization. After this, ||vec|| = 1.0.
/// Cosine distance = normalize at insert + InnerProduct.
pub fn normalize(comptime dim: u32, vec: *[dim]f32) void {
    const sw = default_simd_width;
    const full_iters = dim / sw;
    const remainder = dim % sw;

    var acc: @Vector(sw, f32) = @splat(0.0);

    inline for (0..full_iters) |i| {
        const v: @Vector(sw, f32) = vec[i * sw ..][0..sw].*;
        acc = @mulAdd(@Vector(sw, f32), v, v, acc);
    }

    var norm_sq = @reduce(.Add, acc);

    inline for (0..remainder) |i| {
        const idx = full_iters * sw + i;
        norm_sq += vec[idx] * vec[idx];
    }

    if (norm_sq == 0.0) return;

    const inv_norm = 1.0 / @sqrt(norm_sq);
    const inv_splat: @Vector(sw, f32) = @splat(inv_norm);

    inline for (0..full_iters) |i| {
        const v: @Vector(sw, f32) = vec[i * sw ..][0..sw].*;
        vec[i * sw ..][0..sw].* = v * inv_splat;
    }

    inline for (0..remainder) |i| {
        const idx = full_iters * sw + i;
        vec[idx] *= inv_norm;
    }
}

// Tests

test "L2 distance of identical vectors is zero" {
    const L2 = L2Squared(4);
    const a = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    try std.testing.expectEqual(@as(f32, 0.0), L2.distance(&a, &a));
}

test "L2 distance known value" {
    const L2 = L2Squared(4);
    const a = [4]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [4]f32{ 0.0, 1.0, 0.0, 0.0 };
    try std.testing.expectEqual(@as(f32, 2.0), L2.distance(&a, &b));
}

test "L2 distance symmetry" {
    const L2 = L2Squared(8);
    const a = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [8]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
    try std.testing.expectEqual(L2.distance(&a, &b), L2.distance(&b, &a));
}

test "L2 distanceWithBound returns null when exceeded" {
    const L2 = L2Squared(4);
    const a = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
    const b = [4]f32{ 10.0, 10.0, 10.0, 10.0 };
    // actual distance = 400, bound = 1
    try std.testing.expectEqual(@as(?f32, null), L2.distanceWithBound(&a, &b, 1.0));
}

test "L2 distanceWithBound returns distance within bound" {
    const L2 = L2Squared(4);
    const a = [4]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [4]f32{ 0.0, 1.0, 0.0, 0.0 };
    try std.testing.expectEqual(@as(?f32, 2.0), L2.distanceWithBound(&a, &b, 5.0));
}

test "L2 distanceWithBound early exit on high dim" {
    // dim=128 exercises the early-exit check (128/8 = 16 SIMD iters, check at iter 8)
    const L2 = L2Squared(128);
    var a: [128]f32 = undefined;
    var b: [128]f32 = undefined;
    for (0..128) |i| {
        a[i] = 0.0;
        b[i] = 10.0;
    }
    // actual distance = 128 * 100 = 12800, bound = 1
    try std.testing.expectEqual(@as(?f32, null), L2.distanceWithBound(&a, &b, 1.0));
}

test "InnerProduct orthogonal vectors" {
    const IP = InnerProduct(4);
    const a = [4]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [4]f32{ 0.0, 1.0, 0.0, 0.0 };
    // dot = 0, negated = -0.0 which equals 0.0
    try std.testing.expectEqual(@as(f32, -0.0), IP.distance(&a, &b));
}

test "InnerProduct parallel vectors" {
    const IP = InnerProduct(4);
    const a = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    // dot(a,a) = 1+4+9+16 = 30, negated = -30
    try std.testing.expectEqual(@as(f32, -30.0), IP.distance(&a, &a));
}

test "normalize produces unit vector" {
    var v = [4]f32{ 3.0, 4.0, 0.0, 0.0 };
    normalize(4, &v);
    // ||v|| was 5, so normalized = [0.6, 0.8, 0, 0]
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), v[1], 1e-6);

    // Verify unit length
    var norm_sq: f32 = 0;
    for (v) |x| norm_sq += x * x;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm_sq, 1e-6);
}

test "normalize zero vector is no-op" {
    var v = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
    normalize(4, &v);
    for (v) |x| try std.testing.expectEqual(@as(f32, 0.0), x);
}

test "normalize + InnerProduct equals cosine distance" {
    var a = [4]f32{ 1.0, 2.0, 3.0, 0.0 };
    var b = [4]f32{ 4.0, 5.0, 6.0, 0.0 };

    // Compute cosine distance manually
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;
    for (0..4) |i| {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    const cosine_dist = 1.0 - dot / @sqrt(norm_a * norm_b);

    // Normalize then use IP
    normalize(4, &a);
    normalize(4, &b);
    const IP = InnerProduct(4);
    const ip_dist = IP.distance(&a, &b);
    // IP returns -dot, for unit vectors cosine_dist = 1 - dot = 1 + ip_dist
    const cosine_via_ip = 1.0 + ip_dist;

    try std.testing.expectApproxEqAbs(cosine_dist, cosine_via_ip, 1e-5);
}

test "L2 dim=128 matches naive scalar" {
    var a: [128]f32 = undefined;
    var b: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (0..128) |i| {
        a[i] = rng.float(f32) * 2.0 - 1.0;
        b[i] = rng.float(f32) * 2.0 - 1.0;
    }

    // Naive scalar
    var expected: f32 = 0;
    for (0..128) |i| {
        const d = a[i] - b[i];
        expected += d * d;
    }

    const L2 = L2Squared(128);
    const actual = L2.distance(&a, &b);
    try std.testing.expectApproxEqRel(expected, actual, 1e-5);
}

test "L2 dim=384 matches naive scalar" {
    var a: [384]f32 = undefined;
    var b: [384]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();
    for (0..384) |i| {
        a[i] = rng.float(f32) * 2.0 - 1.0;
        b[i] = rng.float(f32) * 2.0 - 1.0;
    }

    var expected: f32 = 0;
    for (0..384) |i| {
        const d = a[i] - b[i];
        expected += d * d;
    }

    const L2 = L2Squared(384);
    const actual = L2.distance(&a, &b);
    try std.testing.expectApproxEqRel(expected, actual, 1e-5);
}

test "normalize dim=128" {
    var v: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    for (0..128) |i| {
        v[i] = rng.float(f32) * 10.0 - 5.0;
    }

    normalize(128, &v);

    var norm_sq: f32 = 0;
    for (v) |x| norm_sq += x * x;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm_sq, 1e-5);
}
