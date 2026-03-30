//! C-compatible API for ann-benchmarks integration.
//! Supports multiple dimensions via comptime dispatch.
//! Python loads via ctypes.

const std = @import("std");
const hnsw = @import("hnsw.zig");
const distance = @import("distance.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// ann-benchmarks datasets: SIFT(128), GloVe(25,50,100,200), Fashion-MNIST(784),
// GIST(960), NYTimes(256), COCO(512)
const supported_dims = [_]u32{ 25, 50, 100, 128, 200, 256, 384, 512, 768, 784, 960 };

// Wrapper that erases the comptime dim from the type for a uniform C interface.
// Each supported dim gets its own compiled code path with full SIMD unrolling.
const Vtable = struct {
    deinit: *const fn (*anyopaque) void,
    build: *const fn (*anyopaque, [*]const f32, [*]const u64, u32, u32) bool,
    reorder: *const fn (*anyopaque) bool,
    search: *const fn (*anyopaque, [*]const f32, u32, u32, [*]u32) u32,
    save: *const fn (*anyopaque, [*:0]const u8) bool,
    num_nodes: *const fn (*anyopaque) u32,
};

const Handle = struct {
    ptr: *anyopaque,
    vt: *const Vtable,
    dim: u32,
};

fn makeVtable(comptime dim: u32) Vtable {
    const Dist = distance.L2Squared(dim);
    const Index = hnsw.HnswIndex(dim, Dist);

    return .{
        .deinit = struct {
            fn f(p: *anyopaque) void {
                const idx: *Index = @ptrCast(@alignCast(p));
                idx.deinit();
                allocator.destroy(idx);
            }
        }.f,
        .build = struct {
            fn f(p: *anyopaque, vecs: [*]const f32, labels: [*]const u64, n: u32, threads: u32) bool {
                const idx: *Index = @ptrCast(@alignCast(p));
                const vectors: []const [dim]f32 = @as([*]const [dim]f32, @ptrCast(@alignCast(vecs)))[0..n];
                idx.buildBatch(vectors, labels[0..n], threads) catch return false;
                return true;
            }
        }.f,
        .reorder = struct {
            fn f(p: *anyopaque) bool {
                const idx: *Index = @ptrCast(@alignCast(p));
                idx.reorder() catch return false;
                return true;
            }
        }.f,
        .search = struct {
            fn f(p: *anyopaque, query_ptr: [*]const f32, k: u32, ef: u32, out: [*]u32) u32 {
                const idx: *Index = @ptrCast(@alignCast(p));
                const query: *const [dim]f32 = @ptrCast(@alignCast(query_ptr));
                const results = idx.searchKnn(query, k, ef);
                for (results, 0..) |r, i| {
                    out[i] = @intCast(idx.labels[r.id]);
                }
                return @intCast(results.len);
            }
        }.f,
        .save = struct {
            fn f(p: *anyopaque, path_ptr: [*:0]const u8) bool {
                const idx: *Index = @ptrCast(@alignCast(p));
                idx.save(std.mem.span(path_ptr)) catch return false;
                return true;
            }
        }.f,
        .num_nodes = struct {
            fn f(p: *anyopaque) u32 {
                const idx: *Index = @ptrCast(@alignCast(p));
                return idx.num_nodes;
            }
        }.f,
    };
}

// Comptime-generated vtable array, one per supported dim
const vtables = blk: {
    var vts: [supported_dims.len]Vtable = undefined;
    for (supported_dims, 0..) |d, i| {
        vts[i] = makeVtable(d);
    }
    break :blk vts;
};

fn findDimIndex(dim: u32) ?usize {
    for (supported_dims, 0..) |d, i| {
        if (d == dim) return i;
    }
    return null;
}

fn initForDim(comptime dim: u32, max_nodes: u32, ef_construction: u32) ?*anyopaque {
    const Dist = distance.L2Squared(dim);
    const Index = hnsw.HnswIndex(dim, Dist);
    const index = allocator.create(Index) catch return null;
    index.* = Index.init(allocator, max_nodes, ef_construction) catch {
        allocator.destroy(index);
        return null;
    };
    return @ptrCast(index);
}

// C API

export fn hnsw_init(max_nodes: u32, ef_construction: u32, dim: u32) ?*anyopaque {
    const di = findDimIndex(dim) orelse return null;
    const ptr = inline for (supported_dims, 0..) |d, i| {
        if (i == di) break initForDim(d, max_nodes, ef_construction);
    } else null;
    if (ptr == null) return null;

    const handle = allocator.create(Handle) catch return null;
    handle.* = .{ .ptr = ptr.?, .vt = &vtables[di], .dim = dim };
    return @ptrCast(handle);
}

export fn hnsw_free(h: ?*anyopaque) void {
    if (h) |raw| {
        const handle: *Handle = @ptrCast(@alignCast(raw));
        handle.vt.deinit(handle.ptr);
        allocator.destroy(handle);
    }
}

export fn hnsw_build(h: *anyopaque, vecs: [*]const f32, labels: [*]const u64, n: u32, threads: u32) bool {
    const handle: *Handle = @ptrCast(@alignCast(h));
    return handle.vt.build(handle.ptr, vecs, labels, n, threads);
}

export fn hnsw_reorder(h: *anyopaque) bool {
    const handle: *Handle = @ptrCast(@alignCast(h));
    return handle.vt.reorder(handle.ptr);
}

export fn hnsw_search(h: *anyopaque, query: [*]const f32, k: u32, ef: u32, out: [*]u32) u32 {
    const handle: *Handle = @ptrCast(@alignCast(h));
    return handle.vt.search(handle.ptr, query, k, ef, out);
}

export fn hnsw_save(h: *anyopaque, path: [*:0]const u8) bool {
    const handle: *Handle = @ptrCast(@alignCast(h));
    return handle.vt.save(handle.ptr, path);
}

export fn hnsw_load(path: [*:0]const u8, dim: u32) ?*anyopaque {
    const di = findDimIndex(dim) orelse return null;
    const Loader = struct {
        fn load(comptime d: u32, p: []const u8) ?*anyopaque {
            const Dist = distance.L2Squared(d);
            const Index = hnsw.HnswIndex(d, Dist);
            const index = allocator.create(Index) catch return null;
            index.* = Index.load(allocator, p) catch {
                allocator.destroy(index);
                return null;
            };
            return @ptrCast(index);
        }
    };
    const ptr = inline for (supported_dims, 0..) |d, i| {
        if (i == di) break Loader.load(d, std.mem.span(path));
    } else null;
    if (ptr == null) return null;

    const handle = allocator.create(Handle) catch return null;
    handle.* = .{ .ptr = ptr.?, .vt = &vtables[di], .dim = dim };
    return @ptrCast(handle);
}

export fn hnsw_num_nodes(h: *anyopaque) u32 {
    const handle: *Handle = @ptrCast(@alignCast(h));
    return handle.vt.num_nodes(handle.ptr);
}
