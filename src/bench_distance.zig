const std = @import("std");
const distance = @import("distance");

fn benchL2(comptime dim: u32, comptime sw: u32, vectors_a: []const [dim]f32, vectors_b: []const [dim]f32) f64 {
    const L2 = distance.L2SquaredWith(dim, sw);
    const n = vectors_a.len;

    var timer = std.time.Timer.start() catch unreachable;
    var sink: f32 = 0;
    for (0..n) |i| {
        sink += L2.distance(&vectors_a[i], &vectors_b[i]);
    }
    const elapsed_ns = timer.read();
    std.mem.doNotOptimizeAway(&sink);

    return @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(n));
}

pub fn main() !void {
    const n = 1_000_000;

    const allocator = std.heap.page_allocator;
    const vectors_a = try allocator.alloc([128]f32, n);
    defer allocator.free(vectors_a);
    const vectors_b = try allocator.alloc([128]f32, n);
    defer allocator.free(vectors_b);

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (0..n) |i| {
        for (0..128) |j| {
            vectors_a[i][j] = rng.float(f32) * 2.0 - 1.0;
            vectors_b[i][j] = rng.float(f32) * 2.0 - 1.0;
        }
    }

    std.debug.print("L2Squared dim=128, {d} calls\n", .{n});
    std.debug.print("  simd_width=4: {d:.1} ns/call\n", .{benchL2(128, 4, vectors_a, vectors_b)});
    std.debug.print("  simd_width=8: {d:.1} ns/call\n", .{benchL2(128, 8, vectors_a, vectors_b)});
}
