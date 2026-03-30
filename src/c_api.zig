//! C-compatible API for ann-benchmarks integration.
//! Build as shared library: zig build -Doptimize=ReleaseFast -Dlib
//! Python loads via ctypes.

const std = @import("std");
const hnsw = @import("hnsw.zig");
const distance = @import("distance.zig");

const dim = 128;
const Dist = distance.L2Squared(dim);
const Index = hnsw.HnswIndex(dim, Dist);

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Opaque handle: pointer to Index stored as *anyopaque for C

export fn hnsw_init(max_nodes: u32, ef_construction: u32) ?*anyopaque {
    const index = allocator.create(Index) catch return null;
    index.* = Index.init(allocator, max_nodes, ef_construction) catch {
        allocator.destroy(index);
        return null;
    };
    return @ptrCast(index);
}

export fn hnsw_free(handle: ?*anyopaque) void {
    if (handle) |h| {
        const index: *Index = @ptrCast(@alignCast(h));
        index.deinit();
        allocator.destroy(index);
    }
}

export fn hnsw_build(
    handle: *anyopaque,
    vectors_ptr: [*]const f32,
    labels_ptr: [*]const u64,
    n: u32,
    num_threads: u32,
) bool {
    const index: *Index = @ptrCast(@alignCast(handle));
    const vectors: []const [dim]f32 = @as([*]const [dim]f32, @ptrCast(@alignCast(vectors_ptr)))[0..n];
    const labels = labels_ptr[0..n];
    index.buildBatch(vectors, labels, num_threads) catch return false;
    return true;
}

export fn hnsw_reorder(handle: *anyopaque) bool {
    const index: *Index = @ptrCast(@alignCast(handle));
    index.reorder() catch return false;
    return true;
}

export fn hnsw_search(
    handle: *anyopaque,
    query_ptr: [*]const f32,
    k: u32,
    ef_search: u32,
    result_ids: [*]u32,
) u32 {
    const index: *Index = @ptrCast(@alignCast(handle));
    const query: *const [dim]f32 = @ptrCast(@alignCast(query_ptr));
    const results = index.searchKnn(query, k, ef_search);
    for (results, 0..) |r, i| {
        // Return labels (original indices), not internal node IDs
        result_ids[i] = @intCast(index.labels[r.id]);
    }
    return @intCast(results.len);
}

export fn hnsw_save(handle: *anyopaque, path_ptr: [*:0]const u8) bool {
    const index: *Index = @ptrCast(@alignCast(handle));
    const path = std.mem.span(path_ptr);
    index.save(path) catch return false;
    return true;
}

export fn hnsw_load(path_ptr: [*:0]const u8) ?*anyopaque {
    const path = std.mem.span(path_ptr);
    const index = allocator.create(Index) catch return null;
    index.* = Index.load(allocator, path) catch {
        allocator.destroy(index);
        return null;
    };
    return @ptrCast(index);
}

export fn hnsw_num_nodes(handle: *anyopaque) u32 {
    const index: *Index = @ptrCast(@alignCast(handle));
    return index.num_nodes;
}
