const std = @import("std");
const hnsw = @import("hnsw.zig");
const distance = @import("distance.zig");
const heap = @import("heap.zig");

// To benchmark other dims: change this const and rebuild
const bench_dim = 128;
const Dist = distance.L2Squared(bench_dim);
const Index = hnsw.HnswIndex(bench_dim, Dist);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next(); // skip program name

    const cmd = args.next() orelse return usage();
    if (std.mem.eql(u8, cmd, "build")) {
        try cmdBuild(allocator, &args);
    } else if (std.mem.eql(u8, cmd, "search")) {
        try cmdSearch(allocator, &args);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        try cmdBench(allocator, &args);
    } else {
        return usage();
    }
}

fn usage() void {
    const msg =
        \\Usage: hnsw-zig <command> [options]
        \\
        \\Commands:
        \\  build   Build index from fvecs file
        \\  search  Search index and compute recall
        \\  bench   Benchmark QPS/recall at multiple ef values
        \\
        \\build options:
        \\  --input <path>    Input .fvecs file
        \\  --output <path>   Output index file
        \\  --ef <int>        ef_construction (default 200)
        \\
        \\search options:
        \\  --index <path>    Index file
        \\  --queries <path>  Query .fvecs file
        \\  --k <int>         Number of neighbors (default 10)
        \\  --ef <int>        ef_search (default 50)
        \\  --gt <path>       Ground truth .ivecs file
        \\
        \\bench options:
        \\  --index <path>    Index file
        \\  --queries <path>  Query .fvecs file
        \\  --k <int>         Number of neighbors (default 10)
        \\  --ef <list>       Comma-separated ef values (e.g. 10,20,50,100)
        \\  --gt <path>       Ground truth .ivecs file
        \\
    ;
    std.debug.print("{s}", .{msg});
}

// Build command

fn cmdBuild(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var ef_construction: u32 = 200;
    var num_threads: usize = std.Thread.getCpuCount() catch 1;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--input")) {
            input_path = args.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--output")) {
            output_path = args.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--ef")) {
            const v = args.next() orelse return usage();
            ef_construction = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, arg, "--threads")) {
            const v = args.next() orelse return usage();
            num_threads = try std.fmt.parseInt(usize, v, 10);
        }
    }

    const in = input_path orelse return usage();
    const out = output_path orelse return usage();

    std.debug.print("Loading vectors from {s}...\n", .{in});
    const vectors = try loadFvecs(allocator, in);
    defer allocator.free(vectors);
    const num_vectors: u32 = @intCast(vectors.len);
    std.debug.print("Loaded {d} vectors (dim={d})\n", .{ num_vectors, bench_dim });

    std.debug.print("Building index (M={d}, ef_construction={d}, threads={d})...\n", .{ Index.M, ef_construction, num_threads });
    var index = try Index.init(allocator, num_vectors, ef_construction);
    defer index.deinit();

    // Generate labels 0..N
    const labels_buf = try allocator.alloc(u64, num_vectors);
    defer allocator.free(labels_buf);
    for (0..num_vectors) |i| labels_buf[i] = @intCast(i);

    var timer = try std.time.Timer.start();
    try index.buildBatch(vectors, labels_buf, num_threads);
    const build_ns = timer.read();
    const build_s = @as(f64, @floatFromInt(build_ns)) / 1e9;
    std.debug.print("Build complete: {d:.1}s ({d:.0} vectors/sec)\n", .{
        build_s,
        @as(f64, @floatFromInt(num_vectors)) / build_s,
    });

    std.debug.print("Reordering by graph locality...\n", .{});
    try index.reorder();

    std.debug.print("Saving to {s}...\n", .{out});
    try index.save(out);
    std.debug.print("Done.\n", .{});
}

// Search command

fn cmdSearch(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    var index_path: ?[]const u8 = null;
    var queries_path: ?[]const u8 = null;
    var gt_path: ?[]const u8 = null;
    var k: u32 = 10;
    var ef_search: u32 = 50;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--index")) {
            index_path = args.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--queries")) {
            queries_path = args.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--gt")) {
            gt_path = args.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--k")) {
            const v = args.next() orelse return usage();
            k = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, arg, "--ef")) {
            const v = args.next() orelse return usage();
            ef_search = try std.fmt.parseInt(u32, v, 10);
        }
    }

    const ip = index_path orelse return usage();
    const qp = queries_path orelse return usage();

    std.debug.print("Loading index from {s}...\n", .{ip});
    var index = try Index.load(allocator, ip);
    defer index.deinit();
    std.debug.print("Loaded index: {d} nodes\n", .{index.num_nodes});

    const queries = try loadFvecs(allocator, qp);
    defer allocator.free(queries);

    const gt = if (gt_path) |p| try loadIvecs(allocator, p) else null;
    defer if (gt) |g| allocator.free(g);

    var timer = try std.time.Timer.start();
    var total_recall: f64 = 0;

    for (queries, 0..) |*query, qi| {
        const results = index.searchKnn(query, k, ef_search);

        if (gt) |ground_truth| {
            const hits = computeRecall(results, ground_truth[qi][0..k], index.labels);
            total_recall += @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(k));
        }
    }

    const elapsed_ns = timer.read();
    const nq = queries.len;
    const qps = @as(f64, @floatFromInt(nq)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);
    const avg_us = @as(f64, @floatFromInt(elapsed_ns / nq)) / 1000.0;

    std.debug.print("ef={d}  recall@{d}={d:.4}  QPS={d:.0}  avg={d:.0}μs\n", .{
        ef_search,
        k,
        total_recall / @as(f64, @floatFromInt(nq)),
        qps,
        avg_us,
    });
}

// Bench command

fn cmdBench(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    var index_path: ?[]const u8 = null;
    var queries_path: ?[]const u8 = null;
    var gt_path: ?[]const u8 = null;
    var k: u32 = 10;
    var ef_list_str: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--index")) {
            index_path = args.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--queries")) {
            queries_path = args.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--gt")) {
            gt_path = args.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--k")) {
            const v = args.next() orelse return usage();
            k = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, arg, "--ef")) {
            ef_list_str = args.next() orelse return usage();
        }
    }

    const ip = index_path orelse return usage();
    const qp = queries_path orelse return usage();
    const ef_str = ef_list_str orelse "10,20,50,100,200,500";

    std.debug.print("Loading index from {s}...\n", .{ip});
    var index = try Index.load(allocator, ip);
    defer index.deinit();
    std.debug.print("Loaded: {d} nodes, ef_construction={d}\n", .{ index.num_nodes, index.ef_construction });

    const queries = try loadFvecs(allocator, qp);
    defer allocator.free(queries);

    const gt = if (gt_path) |p| try loadIvecs(allocator, p) else null;
    defer if (gt) |g| allocator.free(g);

    const nq = queries.len;
    const latencies = try allocator.alloc(u64, nq);
    defer allocator.free(latencies);

    // Parse ef values
    var ef_iter = std.mem.splitScalar(u8, ef_str, ',');
    while (ef_iter.next()) |ef_val| {
        const ef = try std.fmt.parseInt(u32, ef_val, 10);

        var total_recall: f64 = 0;
        for (queries, 0..) |*query, qi| {
            var timer = try std.time.Timer.start();
            const results = index.searchKnn(query, k, ef);
            latencies[qi] = timer.read();

            if (gt) |ground_truth| {
                const hits = computeRecall(results, ground_truth[qi][0..k], index.labels);
                total_recall += @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(k));
            }
        }

        // Sort latencies for percentiles
        std.sort.pdq(u64, latencies[0..nq], {}, struct {
            fn f(_: void, a: u64, b: u64) bool {
                return a < b;
            }
        }.f);

        const total_ns: u64 = blk: {
            var sum: u64 = 0;
            for (latencies[0..nq]) |l| sum += l;
            break :blk sum;
        };
        const qps = @as(f64, @floatFromInt(nq)) / (@as(f64, @floatFromInt(total_ns)) / 1e9);
        const p50 = @as(f64, @floatFromInt(latencies[nq / 2])) / 1000.0;
        const p99 = @as(f64, @floatFromInt(latencies[nq * 99 / 100])) / 1000.0;

        if (gt != null) {
            const recall = total_recall / @as(f64, @floatFromInt(nq));
            std.debug.print("ef={d:<4}  recall@{d}={d:.4}  QPS={d:<8.0}  p50={d:.0}μs  p99={d:.0}μs\n", .{
                ef, k, recall, qps, p50, p99,
            });
        } else {
            std.debug.print("ef={d:<4}  QPS={d:<8.0}  p50={d:.0}μs  p99={d:.0}μs\n", .{
                ef, qps, p50, p99,
            });
        }
    }
}

// File format helpers

fn loadFvecs(allocator: std.mem.Allocator, path: []const u8) ![][bench_dim]f32 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const file_size = stat.size;
    const vec_bytes = 4 + bench_dim * 4; // 4-byte dim header + data
    if (file_size % vec_bytes != 0) return error.InvalidFormat;
    const num_vectors: usize = @intCast(file_size / vec_bytes);

    const vectors = try allocator.alloc([bench_dim]f32, num_vectors);
    errdefer allocator.free(vectors);

    for (vectors) |*vec| {
        var dim_buf: [4]u8 = undefined;
        try readExact(file, &dim_buf);
        const d = std.mem.readInt(i32, &dim_buf, .little);
        if (d != bench_dim) return error.DimensionMismatch;
        try readExact(file, std.mem.sliceAsBytes(vec));
    }

    return vectors;
}

fn loadIvecs(allocator: std.mem.Allocator, path: []const u8) ![]const [100]u32 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const file_size = stat.size;

    // ivecs: each row is 4 bytes (dim as i32) + dim * 4 bytes (i32 values)
    // Ground truth typically has 100 neighbors per query
    var dim_buf: [4]u8 = undefined;
    try readExact(file, &dim_buf);
    const gt_dim: u32 = @intCast(std.mem.readInt(i32, &dim_buf, .little));
    if (gt_dim < 100) return error.InvalidFormat;

    // Seek back to start
    try file.seekTo(0);

    const row_bytes = 4 + gt_dim * 4;
    if (file_size % row_bytes != 0) return error.InvalidFormat;
    const num_queries: usize = @intCast(file_size / row_bytes);

    const result = try allocator.alloc([100]u32, num_queries);
    errdefer allocator.free(result);

    const row_buf = try allocator.alloc(i32, gt_dim);
    defer allocator.free(row_buf);

    for (result) |*row| {
        var d_buf: [4]u8 = undefined;
        try readExact(file, &d_buf);
        try readExact(file, std.mem.sliceAsBytes(row_buf));
        // Copy first 100 i32 IDs as u32
        for (0..100) |i| {
            row[i] = @intCast(row_buf[i]);
        }
    }

    return result;
}

fn computeRecall(results: []const heap.Entry, ground_truth: []const u32, labels: []const u64) u32 {
    var hits: u32 = 0;
    for (results) |r| {
        // Compare by label (original vector index), not internal node ID.
        // After BFS reorder, node IDs ≠ original indices.
        const result_label: u32 = @intCast(labels[r.id]);
        for (ground_truth) |gt_id| {
            if (result_label == gt_id) {
                hits += 1;
                break;
            }
        }
    }
    return hits;
}

fn readExact(file: std.fs.File, buf: []u8) !void {
    var pos: usize = 0;
    while (pos < buf.len) {
        const n = try file.read(buf[pos..]);
        if (n == 0) return error.UnexpectedEndOfFile;
        pos += n;
    }
}
