const std = @import("std");
const Bitset = @import("bitset.zig").Bitset;
const VisitedSet = @import("bitset.zig").VisitedSet;
const bh = @import("heap.zig");
const Entry = bh.Entry;
const BoundedMinHeap = bh.BoundedMinHeap;
const BoundedMaxHeap = bh.BoundedMaxHeap;

pub fn HnswIndex(comptime dim: u32, comptime Dist: type) type {
    return struct {
        const Self = @This();

        pub const M: u32 = 16;
        pub const M0: u32 = 2 * M;
        const m_plus_1: u32 = M + 1;
        const m0_plus_1: u32 = M0 + 1;
        const max_level_limit: u8 = 16;
        const STRIPE_COUNT: u32 = 1024;
        const sq4_dim: u32 = dim / 2;

        // Per-thread scratch buffers for searchLayer/selectNeighbors.
        // Single-threaded insert/searchKnn use _scratch on the index.
        // buildBatch allocates one per worker thread.
        pub const ThreadScratch = struct {
            result_buf: []Entry,
            candidate_buf: []Entry,
            visited: VisitedSet,
            alloc: std.mem.Allocator,

            pub fn init(allocator: std.mem.Allocator, max_nodes: u32, max_ef: u32) !ThreadScratch {
                const result_buf = try allocator.alloc(Entry, max_ef);
                errdefer allocator.free(result_buf);
                const candidate_buf = try allocator.alloc(Entry, max_nodes);
                errdefer allocator.free(candidate_buf);
                var visited = try VisitedSet.init(allocator, max_nodes);
                errdefer visited.deinit();
                return .{ .result_buf = result_buf, .candidate_buf = candidate_buf, .visited = visited, .alloc = allocator };
            }

            pub fn deinit(self: *ThreadScratch) void {
                self.visited.deinit();
                self.alloc.free(self.candidate_buf);
                self.alloc.free(self.result_buf);
            }
        };

        vectors: [][dim]f32,
        neighbors_l0: []u32,
        upper_layers: [][]u32,
        levels: []u8,
        labels: []u64,

        // SQ8 quantized vectors for fast search (computed from f32 vectors)
        sq_vectors: [][dim]u8,
        sq_mins: [dim]f32,
        sq_inv_ranges: [dim]f32,
        sq_calibrated: bool,

        // SQ4 quantized vectors (4-bit per dim, packed 2 per byte)
        sq4_vectors: [][sq4_dim]u8,
        sq4_mins: [dim]f32,
        sq4_inv_ranges: [dim]f32,

        entry_point: ?u32,
        num_nodes: u32,
        max_nodes: u32,

        ef_construction: u32,
        ml: f32,

        _scratch: ThreadScratch,
        _prng: std.Random.DefaultPrng,
        _upper_arena: std.heap.ArenaAllocator,
        _ep_atomic: std.atomic.Value(u32),
        locks: [STRIPE_COUNT]std.Thread.Mutex,
        allocator: std.mem.Allocator,

        // Init / Deinit

        pub fn init(allocator: std.mem.Allocator, max_nodes: u32, ef_construction: u32) !Self {
            const max_ef: u32 = @max(ef_construction, 512);

            const vectors = try allocator.alloc([dim]f32, max_nodes);
            errdefer allocator.free(vectors);

            const sq_vectors = try allocator.alloc([dim]u8, max_nodes);
            errdefer allocator.free(sq_vectors);

            const sq4_vectors = try allocator.alloc([sq4_dim]u8, max_nodes);
            errdefer allocator.free(sq4_vectors);

            const neighbors_l0 = try allocator.alloc(u32, @as(usize, max_nodes) * m0_plus_1);
            errdefer allocator.free(neighbors_l0);
            @memset(neighbors_l0, 0);

            const upper_layers = try allocator.alloc([]u32, max_nodes);
            errdefer allocator.free(upper_layers);
            for (upper_layers) |*ul| ul.* = &.{};

            const levels = try allocator.alloc(u8, max_nodes);
            errdefer allocator.free(levels);
            @memset(levels, 0);

            const labels = try allocator.alloc(u64, max_nodes);
            errdefer allocator.free(labels);

            var scratch = try ThreadScratch.init(allocator, max_nodes, max_ef);
            errdefer scratch.deinit();

            var locks: [STRIPE_COUNT]std.Thread.Mutex = undefined;
            for (&locks) |*l| l.* = .{};

            return .{
                .vectors = vectors,
                .neighbors_l0 = neighbors_l0,
                .upper_layers = upper_layers,
                .levels = levels,
                .labels = labels,
                .sq_vectors = sq_vectors,
                .sq_mins = @splat(0.0),
                .sq_inv_ranges = @splat(1.0),
                .sq_calibrated = false,
                .sq4_vectors = sq4_vectors,
                .sq4_mins = @splat(0.0),
                .sq4_inv_ranges = @splat(1.0),
                .entry_point = null,
                .num_nodes = 0,
                .max_nodes = max_nodes,
                .ef_construction = ef_construction,
                .ml = 1.0 / @log(@as(f32, @floatFromInt(M))),
                ._scratch = scratch,
                ._prng = std.Random.DefaultPrng.init(42),
                ._upper_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator),
                ._ep_atomic = std.atomic.Value(u32).init(0xFFFFFFFF),
                .locks = locks,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self._upper_arena.deinit();
            self._scratch.deinit();
            self.allocator.free(self.labels);
            self.allocator.free(self.levels);
            self.allocator.free(self.upper_layers);
            self.allocator.free(self.neighbors_l0);
            self.allocator.free(self.sq4_vectors);
            self.allocator.free(self.sq_vectors);
            self.allocator.free(self.vectors);
        }

        // Level selection

        fn selectLevel(self: *Self) u8 {
            const r = self._prng.random().float(f32);
            const safe_r = @max(r, std.math.floatMin(f32));
            const level_f = -@log(safe_r) * self.ml;
            return @intFromFloat(@min(@floor(level_f), @as(f32, @floatFromInt(max_level_limit))));
        }

        // Neighbor access

        fn getNeighbors(self: *const Self, id: u32, layer: u8) []const u32 {
            if (layer == 0) {
                const start: usize = @as(usize, id) * m0_plus_1;
                const count: usize = @intCast(self.neighbors_l0[start]);
                return self.neighbors_l0[start + 1 .. start + 1 + count];
            }
            const data = self.upper_layers[id];
            const offset: usize = (@as(usize, layer) - 1) * m_plus_1;
            const count: usize = @intCast(data[offset]);
            return data[offset + 1 .. offset + 1 + count];
        }

        fn setNeighbors(self: *Self, id: u32, layer: u8, neighbors: []const u32) void {
            if (layer == 0) {
                const start: usize = @as(usize, id) * m0_plus_1;
                self.neighbors_l0[start] = @intCast(neighbors.len);
                @memcpy(self.neighbors_l0[start + 1 .. start + 1 + neighbors.len], neighbors);
            } else {
                const data = self.upper_layers[id];
                const offset: usize = (@as(usize, layer) - 1) * m_plus_1;
                data[offset] = @intCast(neighbors.len);
                @memcpy(data[offset + 1 .. offset + 1 + neighbors.len], neighbors);
            }
        }

        fn addNeighbor(self: *Self, id: u32, layer: u8, neighbor: u32) void {
            if (layer == 0) {
                const start: usize = @as(usize, id) * m0_plus_1;
                const count = self.neighbors_l0[start];
                self.neighbors_l0[start + 1 + count] = neighbor;
                self.neighbors_l0[start] = count + 1;
            } else {
                const data = self.upper_layers[id];
                const offset: usize = (@as(usize, layer) - 1) * m_plus_1;
                const count = data[offset];
                data[offset + 1 + count] = neighbor;
                data[offset] = count + 1;
            }
        }

        fn maxNeighborsForLayer(layer: u8) u32 {
            return if (layer == 0) M0 else M;
        }

        // SEARCH-LAYER — Algorithm 2
        //
        // Uses scratch buffers (per-thread in buildBatch, self._scratch otherwise).
        // Returns number of results in scratch.result_buf[0..returned_count].

        fn searchLayer(
            self: *Self,
            query: *const [dim]f32,
            entry_points: []const u32,
            ef: u32,
            layer: u8,
            scratch: *ThreadScratch,
        ) usize {
            std.debug.assert(ef <= scratch.result_buf.len);

            var candidates = BoundedMinHeap.init(scratch.candidate_buf);
            var results = BoundedMaxHeap.init(scratch.result_buf[0..ef]);
            scratch.visited.reset();

            for (entry_points) |ep| {
                if (scratch.visited.testAndSet(ep)) continue;
                const dist = Dist.distance(query, &self.vectors[ep]);
                candidates.insert(.{ .dist = dist, .id = ep });
                results.insert(.{ .dist = dist, .id = ep });
            }

            while (candidates.pop()) |candidate| {
                if (results.len == ef) {
                    if (candidate.dist > results.peek().?.dist) break;
                }

                // Prefetch next candidate's neighbor list while we process current one
                if (layer == 0) {
                    if (candidates.peek()) |next_cand| {
                        const nl_ptr = @as([*]const u8, @ptrCast(&self.neighbors_l0[@as(usize, next_cand.id) * m0_plus_1]));
                        @prefetch(nl_ptr, .{ .rw = .read, .locality = 0 });
                        @prefetch(nl_ptr + 128, .{ .rw = .read, .locality = 0 });
                    }
                }

                const neighbors = self.getNeighbors(candidate.id, layer);

                // Two-phase batch: filter visited first, then prefetch+compute unvisited.
                // Avoids wasting prefetch bandwidth on already-visited neighbors (~50-70%).
                var unvisited: [M0]u32 = undefined;
                var uv_count: u32 = 0;

                // Phase 1: filter — check visited, collect unvisited IDs
                for (neighbors) |nbr_id| {
                    if (!scratch.visited.testAndSet(nbr_id)) {
                        unvisited[uv_count] = nbr_id;
                        uv_count += 1;
                    }
                }

                // Phase 2: prefetch all unvisited vectors (all 4 CLs each), then compute
                // Prefetch first batch (up to 3 ahead)
                const pf_ahead = @min(uv_count, 3);
                for (unvisited[0..pf_ahead]) |pf_id| {
                    const vec_ptr = @as([*]const u8, @ptrCast(&self.vectors[pf_id]));
                    @prefetch(vec_ptr, .{ .rw = .read, .locality = 0 });
                    @prefetch(vec_ptr + 128, .{ .rw = .read, .locality = 0 });
                    @prefetch(vec_ptr + 256, .{ .rw = .read, .locality = 0 });
                    @prefetch(vec_ptr + 384, .{ .rw = .read, .locality = 0 });
                }

                for (0..uv_count) |j| {
                    // Prefetch j+3 ahead
                    if (j + 3 < uv_count) {
                        const pf_id = unvisited[j + 3];
                        const vec_ptr = @as([*]const u8, @ptrCast(&self.vectors[pf_id]));
                        @prefetch(vec_ptr, .{ .rw = .read, .locality = 0 });
                        @prefetch(vec_ptr + 128, .{ .rw = .read, .locality = 0 });
                        @prefetch(vec_ptr + 256, .{ .rw = .read, .locality = 0 });
                        @prefetch(vec_ptr + 384, .{ .rw = .read, .locality = 0 });
                    }

                    const nbr_id = unvisited[j];
                    const dist = Dist.distance(query, &self.vectors[nbr_id]);

                    if (results.len < ef) {
                        candidates.insert(.{ .dist = dist, .id = nbr_id });
                        results.insert(.{ .dist = dist, .id = nbr_id });
                    } else if (dist < results.peek().?.dist) {
                        candidates.insert(.{ .dist = dist, .id = nbr_id });
                        results.replaceTop(.{ .dist = dist, .id = nbr_id });
                    }
                }

                // SITE C: prefetch neighbor list for next best candidate
                if (candidates.peek()) |nc| {
                    if (layer == 0) {
                        const nl_ptr = @as([*]const u8, @ptrCast(&self.neighbors_l0[@as(usize, nc.id) * m0_plus_1]));
                        @prefetch(nl_ptr, .{ .rw = .read, .locality = 0 });
                        @prefetch(nl_ptr + 128, .{ .rw = .read, .locality = 0 });
                    }
                }
            }

            return results.len;
        }

        // SELECT-NEIGHBORS — Algorithm 4 (heuristic)

        const NeighborResult = struct {
            ids: [M0]u32,
            count: u8,
        };

        fn selectNeighbors(
            self: *const Self,
            candidates: []Entry,
            m_max: u32,
        ) NeighborResult {
            std.sort.pdq(Entry, candidates, {}, entryDistLess);

            var result = NeighborResult{ .ids = undefined, .count = 0 };
            var pruned: [M0]u32 = undefined;
            var pruned_count: u8 = 0;

            for (candidates) |candidate| {
                if (result.count >= m_max) break;

                var dominated = false;
                for (result.ids[0..result.count]) |sel_id| {
                    const dist_to_sel = Dist.distance(
                        &self.vectors[candidate.id],
                        &self.vectors[sel_id],
                    );
                    if (dist_to_sel < candidate.dist) {
                        dominated = true;
                        break;
                    }
                }

                if (!dominated) {
                    result.ids[result.count] = candidate.id;
                    result.count += 1;
                } else if (pruned_count < M0) {
                    pruned[pruned_count] = candidate.id;
                    pruned_count += 1;
                }
            }

            var pi: u8 = 0;
            while (result.count < m_max and pi < pruned_count) {
                result.ids[result.count] = pruned[pi];
                result.count += 1;
                pi += 1;
            }

            return result;
        }

        fn entryDistLess(_: void, a: Entry, b: Entry) bool {
            return a.dist < b.dist;
        }

        // INSERT — Algorithm 1 (single-threaded)

        pub fn insert(self: *Self, vector: [dim]f32, label: u64) !u32 {
            std.debug.assert(self.num_nodes < self.max_nodes);

            const node_id = self.num_nodes;
            self.vectors[node_id] = vector;
            self.labels[node_id] = label;

            const level = self.selectLevel();
            self.levels[node_id] = level;

            if (level > 0) {
                const size: usize = @as(usize, level) * m_plus_1;
                const mem = try self._upper_arena.allocator().alloc(u32, size);
                @memset(mem, 0);
                self.upper_layers[node_id] = mem;
            }

            self.neighbors_l0[@as(usize, node_id) * m0_plus_1] = 0;
            self.num_nodes += 1;

            if (self.num_nodes == 1) {
                self.entry_point = node_id;
                return node_id;
            }

            const ep_id = self.entry_point.?;
            var current_ep = ep_id;
            const top_level = self.levels[ep_id];

            if (top_level > level) {
                var lc = top_level;
                while (lc > level) : (lc -= 1) {
                    const count = self.searchLayer(&vector, &[_]u32{current_ep}, 1, lc, &self._scratch);
                    if (count > 0) current_ep = nearestInBuf(self._scratch.result_buf, count);
                }
            }

            const insert_from: u8 = @min(level, top_level);
            var lc: u8 = insert_from;
            while (true) {
                const m_max = maxNeighborsForLayer(lc);
                const count = self.searchLayer(&vector, &[_]u32{current_ep}, self.ef_construction, lc, &self._scratch);

                const selected = self.selectNeighbors(self._scratch.result_buf[0..count], m_max);
                self.setNeighbors(node_id, lc, selected.ids[0..selected.count]);

                for (selected.ids[0..selected.count]) |neighbor_id| {
                    const existing = self.getNeighbors(neighbor_id, lc);

                    if (existing.len < m_max) {
                        self.addNeighbor(neighbor_id, lc, node_id);
                    } else {
                        var shrink_buf: [M0 + 1]Entry = undefined;
                        for (existing, 0..) |n, i| {
                            shrink_buf[i] = .{
                                .dist = Dist.distance(&self.vectors[neighbor_id], &self.vectors[n]),
                                .id = n,
                            };
                        }
                        shrink_buf[existing.len] = .{
                            .dist = Dist.distance(&self.vectors[neighbor_id], &self.vectors[node_id]),
                            .id = node_id,
                        };
                        const new_nbrs = self.selectNeighbors(
                            shrink_buf[0 .. existing.len + 1],
                            m_max,
                        );
                        self.setNeighbors(neighbor_id, lc, new_nbrs.ids[0..new_nbrs.count]);
                    }
                }

                if (count > 0) current_ep = self._scratch.result_buf[0].id;

                if (lc == 0) break;
                lc -= 1;
            }

            if (level > top_level) {
                self.entry_point = node_id;
            }

            return node_id;
        }

        // SEARCH-KNN — Algorithm 5
        //
        // Zero allocations. Returns slice into _scratch.result_buf.

        pub fn searchKnn(self: *Self, query: *const [dim]f32, k: u32, ef_search: u32) []const Entry {
            if (self.entry_point == null or self.num_nodes == 0) return &.{};

            const ep_id = self.entry_point.?;
            var current_ep = ep_id;
            const top_level = self.levels[ep_id];

            // Upper layer greedy descent (f32, ef=1)
            if (top_level > 0) {
                var lc = top_level;
                while (lc > 0) : (lc -= 1) {
                    const count = self.searchLayer(query, &[_]u32{current_ep}, 1, lc, &self._scratch);
                    if (count > 0) current_ep = nearestInBuf(self._scratch.result_buf, count);
                }
            }

            const ef = @max(ef_search, k);

            if (self.sq_calibrated) {
                // SQ4: quantize query to 4-bit, traverse graph, rescore top candidates with f32
                var sq4_query: [sq4_dim]u8 = undefined;
                self.quantizeVectorSQ4(query, &sq4_query);

                const sq_count = self.searchLayerSQ4(&sq4_query, &[_]u32{current_ep}, ef, &self._scratch);

                // Sort by SQ4 distance, rescore top 6k with f32 (SQ4 has more quant. error than SQ8)
                const results = self._scratch.result_buf[0..sq_count];
                std.sort.pdq(Entry, results, {}, entryDistLess);
                const rescore_count = @min(k * 6, sq_count);

                // Prefetch first 3 f32 vectors for rescore
                for (0..@min(rescore_count, 3)) |pf| {
                    const vec_ptr = @as([*]const u8, @ptrCast(&self.vectors[results[pf].id]));
                    @prefetch(vec_ptr, .{ .rw = .read, .locality = 0 });
                    @prefetch(vec_ptr + 128, .{ .rw = .read, .locality = 0 });
                    @prefetch(vec_ptr + 256, .{ .rw = .read, .locality = 0 });
                    @prefetch(vec_ptr + 384, .{ .rw = .read, .locality = 0 });
                }
                for (results[0..rescore_count], 0..) |*entry, ri| {
                    if (ri + 3 < rescore_count) {
                        const vec_ptr = @as([*]const u8, @ptrCast(&self.vectors[results[ri + 3].id]));
                        @prefetch(vec_ptr, .{ .rw = .read, .locality = 0 });
                        @prefetch(vec_ptr + 128, .{ .rw = .read, .locality = 0 });
                        @prefetch(vec_ptr + 256, .{ .rw = .read, .locality = 0 });
                        @prefetch(vec_ptr + 384, .{ .rw = .read, .locality = 0 });
                    }
                    entry.dist = Dist.distance(query, &self.vectors[entry.id]);
                }

                std.sort.pdq(Entry, results[0..rescore_count], {}, entryDistLess);
                return results[0..@min(k, rescore_count)];
            } else {
                const count = self.searchLayer(query, &[_]u32{current_ep}, ef, 0, &self._scratch);
                std.sort.pdq(Entry, self._scratch.result_buf[0..count], {}, entryDistLess);
                return self._scratch.result_buf[0..@min(k, count)];
            }
        }

        fn nearestInBuf(result_buf: []const Entry, count: usize) u32 {
            var best = result_buf[0];
            for (result_buf[1..count]) |entry| {
                if (entry.dist < best.dist) best = entry;
            }
            return best.id;
        }

        // SQ8 — Scalar Quantization (8-bit)

        /// Compute per-dimension min/range and quantize all vectors to u8.
        pub fn calibrateAndQuantize(self: *Self) void {
            @setEvalBranchQuota(dim * 20);
            const n = self.num_nodes;
            if (n == 0) return;

            // Compute per-dimension min and max
            var mins: [dim]f32 = self.vectors[0];
            var maxs: [dim]f32 = self.vectors[0];

            for (1..n) |i| {
                const v = self.vectors[i];
                inline for (0..dim) |d| {
                    if (v[d] < mins[d]) mins[d] = v[d];
                    if (v[d] > maxs[d]) maxs[d] = v[d];
                }
            }

            // Compute inverse ranges for quantization: 255 / (max - min)
            var inv_ranges: [dim]f32 = undefined;
            inline for (0..dim) |d| {
                const range = maxs[d] - mins[d];
                inv_ranges[d] = if (range > 0) 255.0 / range else 0.0;
            }

            self.sq_mins = mins;
            self.sq_inv_ranges = inv_ranges;

            // Quantize all vectors
            for (0..n) |i| {
                self.quantizeVector(&self.vectors[i], &self.sq_vectors[i]);
            }

            // SQ4: same min/max, but scale to 0-15 instead of 0-255
            var sq4_inv: [dim]f32 = undefined;
            inline for (0..dim) |d| {
                const range = maxs[d] - mins[d];
                sq4_inv[d] = if (range > 0) 15.0 / range else 0.0;
            }
            self.sq4_mins = mins;
            self.sq4_inv_ranges = sq4_inv;

            for (0..n) |i| {
                self.quantizeVectorSQ4(&self.vectors[i], &self.sq4_vectors[i]);
            }

            self.sq_calibrated = true;
        }

        /// Quantize a single f32 vector to u8 using stored calibration.
        inline fn quantizeVector(self: *const Self, vec: *const [dim]f32, out: *[dim]u8) void {
            inline for (0..dim) |d| {
                const scaled = (vec[d] - self.sq_mins[d]) * self.sq_inv_ranges[d];
                const clamped = @max(@as(f32, 0.0), @min(scaled, @as(f32, 255.0)));
                out[d] = @intFromFloat(clamped);
            }
        }

        /// SQ8 squared L2 distance (integer). Returns f32 cast of the integer sum
        /// for compatibility with heap Entry. Ranking is preserved.
        inline fn sq8Distance(a: *const [dim]u8, b: *const [dim]u8) f32 {
            // Process 8 u8s at a time: widen to i16, subtract, widen to i32, square, accumulate
            const sw = 8;
            const iters = dim / sw;
            const remainder = dim % sw;

            var acc: @Vector(sw, i32) = @splat(@as(i32, 0));

            inline for (0..iters) |i| {
                const va: @Vector(sw, u8) = a[i * sw ..][0..sw].*;
                const vb: @Vector(sw, u8) = b[i * sw ..][0..sw].*;
                const va16: @Vector(sw, i16) = @intCast(va);
                const vb16: @Vector(sw, i16) = @intCast(vb);
                const diff = va16 - vb16;
                const diff32: @Vector(sw, i32) = diff;
                acc += diff32 * diff32;
            }

            var sum: i32 = @reduce(.Add, acc);

            inline for (0..remainder) |i| {
                const idx = iters * sw + i;
                const d: i32 = @as(i32, a[idx]) - @as(i32, b[idx]);
                sum += d * d;
            }

            return @floatFromInt(sum);
        }

        // SQ4 — Scalar Quantization (4-bit)

        /// Quantize f32 vector to packed 4-bit nibbles.
        inline fn quantizeVectorSQ4(self: *const Self, vec: *const [dim]f32, out: *[sq4_dim]u8) void {
            inline for (0..sq4_dim) |i| {
                const d0 = i * 2;
                const d1 = d0 + 1;
                const s0 = (vec[d0] - self.sq4_mins[d0]) * self.sq4_inv_ranges[d0];
                const s1 = (vec[d1] - self.sq4_mins[d1]) * self.sq4_inv_ranges[d1];
                const q0: u8 = @intFromFloat(@max(@as(f32, 0.0), @min(s0, @as(f32, 15.0))));
                const q1: u8 = @intFromFloat(@max(@as(f32, 0.0), @min(s1, @as(f32, 15.0))));
                out[i] = (q0 << 4) | q1;
            }
        }

        /// SQ4 squared L2 distance on packed nibbles.
        /// Processes 16 packed bytes (32 dims) per SIMD iteration.
        inline fn sq4Distance(a: *const [sq4_dim]u8, b: *const [sq4_dim]u8) f32 {
            const sw = 16;
            const iters = sq4_dim / sw;
            const remainder = sq4_dim % sw;

            var acc_hi: @Vector(sw, u16) = @splat(@as(u16, 0));
            var acc_lo: @Vector(sw, u16) = @splat(@as(u16, 0));

            const mask_lo: @Vector(sw, u8) = @splat(0x0F);

            inline for (0..iters) |i| {
                const va: @Vector(sw, u8) = a[i * sw ..][0..sw].*;
                const vb: @Vector(sw, u8) = b[i * sw ..][0..sw].*;

                // Extract high and low nibbles
                const va_hi = va >> @as(@Vector(sw, u3), @splat(4));
                const va_lo = va & mask_lo;
                const vb_hi = vb >> @as(@Vector(sw, u3), @splat(4));
                const vb_lo = vb & mask_lo;

                // Absolute differences (u8 nibbles, max diff = 15)
                const diff_hi = @max(va_hi, vb_hi) - @min(va_hi, vb_hi);
                const diff_lo = @max(va_lo, vb_lo) - @min(va_lo, vb_lo);

                // Square (max 225, fits in u8) then widen to u16 for safe accumulation
                const sq_hi: @Vector(sw, u16) = @intCast(diff_hi * diff_hi);
                const sq_lo: @Vector(sw, u16) = @intCast(diff_lo * diff_lo);

                acc_hi += sq_hi;
                acc_lo += sq_lo;
            }

            var sum: u32 = @reduce(.Add, acc_hi) + @reduce(.Add, acc_lo);

            // Scalar tail
            inline for (0..remainder) |i| {
                const idx = iters * sw + i;
                const byte_a = a[idx];
                const byte_b = b[idx];
                const d_hi: i16 = @as(i16, byte_a >> 4) - @as(i16, byte_b >> 4);
                const d_lo: i16 = @as(i16, byte_a & 0x0F) - @as(i16, byte_b & 0x0F);
                sum += @intCast(@as(u16, @abs(d_hi)) * @as(u16, @abs(d_hi)));
                sum += @intCast(@as(u16, @abs(d_lo)) * @as(u16, @abs(d_lo)));
            }

            return @floatFromInt(sum);
        }

        /// searchLayer using SQ4 distances. Layer 0 only.
        fn searchLayerSQ4(
            self: *Self,
            sq4_query: *const [sq4_dim]u8,
            entry_points: []const u32,
            ef: u32,
            scratch: *ThreadScratch,
        ) usize {
            std.debug.assert(ef <= scratch.result_buf.len);

            var candidates = BoundedMinHeap.init(scratch.candidate_buf);
            var results = BoundedMaxHeap.init(scratch.result_buf[0..ef]);
            scratch.visited.reset();

            for (entry_points) |ep| {
                if (scratch.visited.testAndSet(ep)) continue;
                const dist = sq4Distance(sq4_query, &self.sq4_vectors[ep]);
                candidates.insert(.{ .dist = dist, .id = ep });
                results.insert(.{ .dist = dist, .id = ep });
            }

            while (candidates.pop()) |candidate| {
                if (results.len == ef) {
                    if (candidate.dist > results.peek().?.dist) break;
                }

                // Prefetch next candidate's neighbor list
                if (candidates.peek()) |next_cand| {
                    const nl_ptr = @as([*]const u8, @ptrCast(&self.neighbors_l0[@as(usize, next_cand.id) * m0_plus_1]));
                    @prefetch(nl_ptr, .{ .rw = .read, .locality = 0 });
                    @prefetch(nl_ptr + 128, .{ .rw = .read, .locality = 0 });
                }

                const neighbors = self.getNeighbors(candidate.id, 0);

                // Two-phase: filter visited, then prefetch+compute
                var unvisited: [M0]u32 = undefined;
                var uv_count: u32 = 0;

                for (neighbors) |nbr_id| {
                    if (!scratch.visited.testAndSet(nbr_id)) {
                        unvisited[uv_count] = nbr_id;
                        uv_count += 1;
                    }
                }

                // SQ4 vectors = 64B = half a M1 cache line. Prefetch brings 128B
                // so each prefetch covers 2 vectors if they're adjacent.
                const pf_ahead = @min(uv_count, 3);
                for (unvisited[0..pf_ahead]) |pf_id| {
                    @prefetch(@as([*]const u8, @ptrCast(&self.sq4_vectors[pf_id])), .{ .rw = .read, .locality = 0 });
                }

                for (0..uv_count) |j| {
                    if (j + 3 < uv_count) {
                        @prefetch(@as([*]const u8, @ptrCast(&self.sq4_vectors[unvisited[j + 3]])), .{ .rw = .read, .locality = 0 });
                    }

                    const nbr_id = unvisited[j];
                    const dist = sq4Distance(sq4_query, &self.sq4_vectors[nbr_id]);

                    if (results.len < ef) {
                        candidates.insert(.{ .dist = dist, .id = nbr_id });
                        results.insert(.{ .dist = dist, .id = nbr_id });
                    } else if (dist < results.peek().?.dist) {
                        candidates.insert(.{ .dist = dist, .id = nbr_id });
                        results.replaceTop(.{ .dist = dist, .id = nbr_id });
                    }
                }

                // SITE C: prefetch neighbor list for next best candidate
                if (candidates.peek()) |nc| {
                    const nl_ptr = @as([*]const u8, @ptrCast(&self.neighbors_l0[@as(usize, nc.id) * m0_plus_1]));
                    @prefetch(nl_ptr, .{ .rw = .read, .locality = 0 });
                    @prefetch(nl_ptr + 128, .{ .rw = .read, .locality = 0 });
                }
            }

            return results.len;
        }

        /// searchLayer variant using SQ8 distances. Same algorithm as searchLayer
        /// but operates on quantized vectors. Layer 0 only.
        fn searchLayerSQ8(
            self: *Self,
            sq_query: *const [dim]u8,
            entry_points: []const u32,
            ef: u32,
            scratch: *ThreadScratch,
        ) usize {
            std.debug.assert(ef <= scratch.result_buf.len);

            var candidates = BoundedMinHeap.init(scratch.candidate_buf);
            var results = BoundedMaxHeap.init(scratch.result_buf[0..ef]);
            scratch.visited.reset();

            for (entry_points) |ep| {
                if (scratch.visited.testAndSet(ep)) continue;
                const dist = sq8Distance(sq_query, &self.sq_vectors[ep]);
                candidates.insert(.{ .dist = dist, .id = ep });
                results.insert(.{ .dist = dist, .id = ep });
            }

            while (candidates.pop()) |candidate| {
                if (results.len == ef) {
                    if (candidate.dist > results.peek().?.dist) break;
                }

                // Prefetch next candidate's neighbor list
                if (candidates.peek()) |next_cand| {
                    const nl_ptr = @as([*]const u8, @ptrCast(&self.neighbors_l0[@as(usize, next_cand.id) * m0_plus_1]));
                    @prefetch(nl_ptr, .{ .rw = .read, .locality = 0 });
                    @prefetch(nl_ptr + 128, .{ .rw = .read, .locality = 0 });
                }

                const neighbors = self.getNeighbors(candidate.id, 0);

                // Two-phase: filter visited, then prefetch+compute
                var unvisited: [M0]u32 = undefined;
                var uv_count: u32 = 0;

                for (neighbors) |nbr_id| {
                    if (!scratch.visited.testAndSet(nbr_id)) {
                        unvisited[uv_count] = nbr_id;
                        uv_count += 1;
                    }
                }

                // Prefetch SQ8 vectors (128B per vector = 1 M1 cache line!)
                const pf_ahead = @min(uv_count, 3);
                for (unvisited[0..pf_ahead]) |pf_id| {
                    @prefetch(@as([*]const u8, @ptrCast(&self.sq_vectors[pf_id])), .{ .rw = .read, .locality = 0 });
                }

                for (0..uv_count) |j| {
                    if (j + 3 < uv_count) {
                        @prefetch(@as([*]const u8, @ptrCast(&self.sq_vectors[unvisited[j + 3]])), .{ .rw = .read, .locality = 0 });
                    }

                    const nbr_id = unvisited[j];
                    const dist = sq8Distance(sq_query, &self.sq_vectors[nbr_id]);

                    if (results.len < ef) {
                        candidates.insert(.{ .dist = dist, .id = nbr_id });
                        results.insert(.{ .dist = dist, .id = nbr_id });
                    } else if (dist < results.peek().?.dist) {
                        candidates.insert(.{ .dist = dist, .id = nbr_id });
                        results.replaceTop(.{ .dist = dist, .id = nbr_id });
                    }
                }

                // SITE C: prefetch neighbor list for next best candidate
                if (candidates.peek()) |nc| {
                    const nl_ptr = @as([*]const u8, @ptrCast(&self.neighbors_l0[@as(usize, nc.id) * m0_plus_1]));
                    @prefetch(nl_ptr, .{ .rw = .read, .locality = 0 });
                    @prefetch(nl_ptr + 128, .{ .rw = .read, .locality = 0 });
                }
            }

            return results.len;
        }

        // BUILD-BATCH — Parallel bulk insert
        //
        // Phase 1: sequential data population + level selection + arena alloc
        // Phase 1.5: connect node 0 (entry point must be fully wired before threads)
        // Phase 2: parallel graph connection with striped locks

        pub fn buildBatch(
            self: *Self,
            vectors: []const [dim]f32,
            labels: []const u64,
            num_threads: usize,
        ) !void {
            const n: u32 = @intCast(vectors.len);
            std.debug.assert(n <= self.max_nodes);

            // Phase 1: sequential — populate data, select levels, init neighbors
            for (0..n) |i| {
                self.vectors[i] = vectors[i];
                self.labels[i] = labels[i];
                self.levels[i] = self.selectLevel();
                if (self.levels[i] > 0) {
                    const size: usize = @as(usize, self.levels[i]) * m_plus_1;
                    const mem = try self._upper_arena.allocator().alloc(u32, size);
                    @memset(mem, 0);
                    self.upper_layers[i] = mem;
                }
                self.neighbors_l0[i * m0_plus_1] = 0;
            }
            self.num_nodes = n;
            self.entry_point = 0;
            self._ep_atomic.store(0, .release);

            if (n <= 1) return;

            // Phase 1.5: connect node 0 sequentially.
            // Node 0 is the entry point. If threads start before it's connected,
            // early inserts do greedy descent from a node with zero neighbors —
            // they build garbage connections until the CAS fires. Connect node 0
            // fully first, same as hnswlib.
            self.connectNode(0, &self._scratch);

            if (n <= 2) return;

            // Phase 2: parallel graph connection
            const worker_count = @min(num_threads, @as(usize, n - 1));
            var next_id = std.atomic.Value(u32).init(1);
            var progress = std.atomic.Value(u32).init(1);

            if (worker_count <= 1) {
                // Sequential fallback
                for (1..n) |i| {
                    self.connectNode(@intCast(i), &self._scratch);
                }
                self.entry_point = self._ep_atomic.load(.acquire);
                return;
            }

            const scratches = try self.allocator.alloc(ThreadScratch, worker_count);
            defer {
                for (scratches) |*s| s.deinit();
                self.allocator.free(scratches);
            }
            for (scratches) |*s| {
                s.* = try ThreadScratch.init(
                    self.allocator,
                    self.max_nodes,
                    @max(self.ef_construction, 512),
                );
            }

            const threads = try self.allocator.alloc(std.Thread, worker_count);
            defer self.allocator.free(threads);

            for (0..worker_count) |i| {
                threads[i] = try std.Thread.spawn(.{}, connectWorker, .{
                    self, &next_id, &progress, &scratches[i],
                });
            }

            // Progress reporting from main thread
            while (true) {
                std.Thread.sleep(500_000_000); // 500ms
                const done = progress.load(.monotonic);
                if (done >= n) break;
                std.debug.print("  connected {d}/{d}\n", .{ done, n });
            }

            for (threads) |t| t.join();

            // Sync entry point back from atomic
            self.entry_point = self._ep_atomic.load(.acquire);
        }

        fn connectWorker(
            self: *Self,
            next_id: *std.atomic.Value(u32),
            progress: *std.atomic.Value(u32),
            scratch: *ThreadScratch,
        ) void {
            while (true) {
                const node_id = next_id.fetchAdd(1, .monotonic);
                if (node_id >= self.num_nodes) break;
                self.connectNode(node_id, scratch);
                _ = progress.fetchAdd(1, .monotonic);
            }
        }

        /// Connect a single node into the graph. Used by both single-threaded
        /// insert (no locking needed) and parallel buildBatch (locks neighbors).
        fn connectNode(self: *Self, node_id: u32, scratch: *ThreadScratch) void {
            const vector = &self.vectors[node_id];
            const level = self.levels[node_id];

            // Read entry point atomically (may be updated by other threads)
            const ep_id = self._ep_atomic.load(.acquire);
            var current_ep = ep_id;
            const top_level = self.levels[ep_id];

            // Greedy descent through layers above this node's level
            if (top_level > level) {
                var lc = top_level;
                while (lc > level) : (lc -= 1) {
                    const count = self.searchLayer(vector, &[_]u32{current_ep}, 1, lc, scratch);
                    if (count > 0) current_ep = nearestInBuf(scratch.result_buf, count);
                }
            }

            // Insert at each layer from min(level, top_level) down to 0
            const insert_from: u8 = @min(level, top_level);
            var lc: u8 = insert_from;
            while (true) {
                const m_max = maxNeighborsForLayer(lc);
                const count = self.searchLayer(vector, &[_]u32{current_ep}, self.ef_construction, lc, scratch);

                const selected = self.selectNeighbors(scratch.result_buf[0..count], m_max);
                self.setNeighbors(node_id, lc, selected.ids[0..selected.count]);

                // Bidirectional connections — lock each neighbor
                for (selected.ids[0..selected.count]) |neighbor_id| {
                    self.locks[neighbor_id % STRIPE_COUNT].lock();
                    defer self.locks[neighbor_id % STRIPE_COUNT].unlock();

                    const existing = self.getNeighbors(neighbor_id, lc);

                    if (existing.len < m_max) {
                        self.addNeighbor(neighbor_id, lc, node_id);
                    } else {
                        var shrink_buf: [M0 + 1]Entry = undefined;
                        for (existing, 0..) |nbr, i| {
                            shrink_buf[i] = .{
                                .dist = Dist.distance(&self.vectors[neighbor_id], &self.vectors[nbr]),
                                .id = nbr,
                            };
                        }
                        shrink_buf[existing.len] = .{
                            .dist = Dist.distance(&self.vectors[neighbor_id], &self.vectors[node_id]),
                            .id = node_id,
                        };
                        const new_nbrs = self.selectNeighbors(
                            shrink_buf[0 .. existing.len + 1],
                            m_max,
                        );
                        self.setNeighbors(neighbor_id, lc, new_nbrs.ids[0..new_nbrs.count]);
                    }
                }

                if (count > 0) current_ep = scratch.result_buf[0].id;

                if (lc == 0) break;
                lc -= 1;
            }

            // CAS loop to update entry point if this node has a higher level
            if (level > top_level) {
                while (true) {
                    const cur = self._ep_atomic.load(.acquire);
                    if (self.levels[node_id] <= self.levels[cur]) break;
                    if (self._ep_atomic.cmpxchgWeak(cur, node_id, .release, .monotonic) == null) break;
                }
            }
        }

        // REORDER — BFS graph-locality reordering

        pub fn reorder(self: *Self) !void {
            const n = self.num_nodes;
            if (n < 2) return;

            const new_to_old = try self.allocator.alloc(u32, n);
            defer self.allocator.free(new_to_old);
            const old_to_new = try self.allocator.alloc(u32, n);
            defer self.allocator.free(old_to_new);
            @memset(old_to_new, 0xFFFFFFFF);

            var head: u32 = 0;
            var tail: u32 = 0;
            const start = self.entry_point orelse return;
            new_to_old[tail] = start;
            old_to_new[start] = tail;
            tail += 1;

            while (head < tail) {
                const old_id = new_to_old[head];
                head += 1;
                const neighbors = self.getNeighbors(old_id, 0);
                for (neighbors) |nbr| {
                    if (old_to_new[nbr] == 0xFFFFFFFF) {
                        old_to_new[nbr] = tail;
                        new_to_old[tail] = nbr;
                        tail += 1;
                    }
                }
            }
            for (0..n) |i| {
                if (old_to_new[i] == 0xFFFFFFFF) {
                    old_to_new[@intCast(i)] = tail;
                    new_to_old[tail] = @intCast(i);
                    tail += 1;
                }
            }

            const new_vectors = try self.allocator.alloc([dim]f32, self.max_nodes);
            for (0..n) |old| {
                new_vectors[old_to_new[old]] = self.vectors[old];
            }
            self.allocator.free(self.vectors);
            self.vectors = new_vectors;

            const new_labels = try self.allocator.alloc(u64, self.max_nodes);
            for (0..n) |old| {
                new_labels[old_to_new[old]] = self.labels[old];
            }
            self.allocator.free(self.labels);
            self.labels = new_labels;

            const new_levels = try self.allocator.alloc(u8, self.max_nodes);
            @memset(new_levels, 0);
            for (0..n) |old| {
                new_levels[old_to_new[old]] = self.levels[old];
            }
            self.allocator.free(self.levels);
            self.levels = new_levels;

            const new_l0 = try self.allocator.alloc(u32, @as(usize, self.max_nodes) * m0_plus_1);
            @memset(new_l0, 0);
            for (0..n) |old| {
                const new_id = old_to_new[old];
                const old_start: usize = old * m0_plus_1;
                const new_start: usize = @as(usize, new_id) * m0_plus_1;
                const count = self.neighbors_l0[old_start];
                new_l0[new_start] = count;
                for (0..count) |j| {
                    new_l0[new_start + 1 + j] = old_to_new[self.neighbors_l0[old_start + 1 + j]];
                }
            }
            self.allocator.free(self.neighbors_l0);
            self.neighbors_l0 = new_l0;

            const new_upper = try self.allocator.alloc([]u32, self.max_nodes);
            for (new_upper) |*ul| ul.* = &.{};
            for (0..n) |old| {
                const new_id = old_to_new[old];
                new_upper[new_id] = self.upper_layers[old];
                const level = new_levels[new_id];
                if (level > 0) {
                    const data = new_upper[new_id];
                    for (0..level) |lyr| {
                        const off = lyr * m_plus_1;
                        const cnt = data[off];
                        for (0..cnt) |j| {
                            data[off + 1 + j] = old_to_new[data[off + 1 + j]];
                        }
                    }
                }
            }
            self.allocator.free(self.upper_layers);
            self.upper_layers = new_upper;

            self.entry_point = old_to_new[start];

            // Calibrate SQ8 on the reordered vectors
            self.calibrateAndQuantize();
        }

        // SAVE / LOAD

        const Header = extern struct {
            magic: u32,
            version: u32,
            dim_val: u32,
            m_val: u16,
            m0_val: u16,
            ef_construction_val: u32,
            num_nodes: u32,
            max_nodes_val: u32,
            entry_point_val: u32,
            upper_links_total: u32,
        };

        pub fn save(self: *const Self, path: []const u8) !void {
            const file = try std.fs.cwd().createFile(path, .{});
            defer file.close();

            const n: usize = self.num_nodes;

            var upper_total: u32 = 0;
            const offsets = try self.allocator.alloc(u32, n);
            defer self.allocator.free(offsets);

            for (0..n) |i| {
                if (self.levels[i] > 0) {
                    offsets[i] = upper_total;
                    upper_total += @as(u32, self.levels[i]) * m_plus_1;
                } else {
                    offsets[i] = 0;
                }
            }

            const links = try self.allocator.alloc(u32, upper_total);
            defer self.allocator.free(links);

            for (0..n) |i| {
                if (self.levels[i] > 0) {
                    const size: usize = @as(usize, self.levels[i]) * m_plus_1;
                    const off: usize = offsets[i];
                    @memcpy(links[off .. off + size], self.upper_layers[i][0..size]);
                }
            }

            const header = Header{
                .magic = 0x484E5357,
                .version = 1,
                .dim_val = dim,
                .m_val = @intCast(M),
                .m0_val = @intCast(M0),
                .ef_construction_val = self.ef_construction,
                .num_nodes = self.num_nodes,
                .max_nodes_val = self.max_nodes,
                .entry_point_val = self.entry_point orelse 0xFFFFFFFF,
                .upper_links_total = upper_total,
            };

            try file.writeAll(std.mem.asBytes(&header));
            try file.writeAll(std.mem.sliceAsBytes(self.vectors[0..n]));
            try file.writeAll(std.mem.sliceAsBytes(self.neighbors_l0[0 .. n * m0_plus_1]));
            try file.writeAll(self.levels[0..n]);
            try file.writeAll(std.mem.sliceAsBytes(self.labels[0..n]));
            try file.writeAll(std.mem.sliceAsBytes(offsets));
            try file.writeAll(std.mem.sliceAsBytes(links));
        }

        pub fn load(allocator: std.mem.Allocator, path: []const u8) !Self {
            const file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            var header: Header = undefined;
            try readExact(file, std.mem.asBytes(&header));

            if (header.magic != 0x484E5357) return error.InvalidFormat;
            if (header.version != 1) return error.InvalidFormat;
            if (header.dim_val != dim) return error.DimensionMismatch;
            if (header.m_val != @as(u16, @intCast(M))) return error.ParameterMismatch;

            const n: usize = header.num_nodes;
            const max_n: u32 = header.max_nodes_val;
            const max_ef: u32 = @max(header.ef_construction_val, 512);

            const vectors = try allocator.alloc([dim]f32, max_n);
            errdefer allocator.free(vectors);
            const sq_vectors = try allocator.alloc([dim]u8, max_n);
            errdefer allocator.free(sq_vectors);
            const sq4_vectors_load = try allocator.alloc([sq4_dim]u8, max_n);
            errdefer allocator.free(sq4_vectors_load);
            const neighbors_l0 = try allocator.alloc(u32, @as(usize, max_n) * m0_plus_1);
            errdefer allocator.free(neighbors_l0);
            @memset(neighbors_l0, 0);
            const upper_layers_arr = try allocator.alloc([]u32, max_n);
            errdefer allocator.free(upper_layers_arr);
            for (upper_layers_arr) |*ul| ul.* = &.{};
            const levels = try allocator.alloc(u8, max_n);
            errdefer allocator.free(levels);
            @memset(levels, 0);
            const labels = try allocator.alloc(u64, max_n);
            errdefer allocator.free(labels);

            var scratch = try ThreadScratch.init(allocator, max_n, max_ef);
            errdefer scratch.deinit();

            try readExact(file, std.mem.sliceAsBytes(vectors[0..n]));
            try readExact(file, std.mem.sliceAsBytes(neighbors_l0[0 .. n * m0_plus_1]));
            try readExact(file, levels[0..n]);
            try readExact(file, std.mem.sliceAsBytes(labels[0..n]));

            const offsets = try allocator.alloc(u32, n);
            defer allocator.free(offsets);
            try readExact(file, std.mem.sliceAsBytes(offsets));

            var upper_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            errdefer upper_arena.deinit();

            if (header.upper_links_total > 0) {
                const link_data = try upper_arena.allocator().alloc(u32, header.upper_links_total);
                try readExact(file, std.mem.sliceAsBytes(link_data));
                for (0..n) |i| {
                    if (levels[i] > 0) {
                        const size: usize = @as(usize, levels[i]) * m_plus_1;
                        const off: usize = offsets[i];
                        upper_layers_arr[i] = link_data[off .. off + size];
                    }
                }
            }

            var locks: [STRIPE_COUNT]std.Thread.Mutex = undefined;
            for (&locks) |*l| l.* = .{};

            var self = Self{
                .vectors = vectors,
                .neighbors_l0 = neighbors_l0,
                .upper_layers = upper_layers_arr,
                .levels = levels,
                .labels = labels,
                .sq_vectors = sq_vectors,
                .sq_mins = @splat(0.0),
                .sq_inv_ranges = @splat(1.0),
                .sq_calibrated = false,
                .sq4_vectors = sq4_vectors_load,
                .sq4_mins = @splat(0.0),
                .sq4_inv_ranges = @splat(1.0),
                .entry_point = if (header.entry_point_val == 0xFFFFFFFF) null else header.entry_point_val,
                .num_nodes = header.num_nodes,
                .max_nodes = max_n,
                .ef_construction = header.ef_construction_val,
                .ml = 1.0 / @log(@as(f32, @floatFromInt(M))),
                ._scratch = scratch,
                ._prng = std.Random.DefaultPrng.init(42),
                ._upper_arena = upper_arena,
                ._ep_atomic = std.atomic.Value(u32).init(header.entry_point_val),
                .locks = locks,
                .allocator = allocator,
            };

            // Compute SQ8 quantization from loaded f32 vectors
            self.calibrateAndQuantize();
            return self;
        }

        fn readExact(file: std.fs.File, buf: []u8) !void {
            var pos: usize = 0;
            while (pos < buf.len) {
                const n = try file.read(buf[pos..]);
                if (n == 0) return error.UnexpectedEndOfFile;
                pos += n;
            }
        }
    };
}

// Tests

const distance = @import("distance.zig");

test "init and deinit — no leaks" {
    const Index = HnswIndex(4, distance.L2Squared(4));
    var index = try Index.init(std.testing.allocator, 100, 32);
    defer index.deinit();
    try std.testing.expectEqual(@as(u32, 0), index.num_nodes);
    try std.testing.expectEqual(@as(?u32, null), index.entry_point);
}

test "insert single vector and search returns it" {
    const Dist = distance.L2Squared(4);
    const Index = HnswIndex(4, Dist);
    var index = try Index.init(std.testing.allocator, 100, 32);
    defer index.deinit();

    const vec = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    const id = try index.insert(vec, 0);
    try std.testing.expectEqual(@as(u32, 0), id);
    try std.testing.expectEqual(@as(u32, 1), index.num_nodes);

    const results = index.searchKnn(&vec, 1, 10);
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(u32, 0), results[0].id);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), results[0].dist, 1e-6);
}

test "insert two vectors, search returns correct nearest" {
    const Dist = distance.L2Squared(4);
    const Index = HnswIndex(4, Dist);
    var index = try Index.init(std.testing.allocator, 100, 32);
    defer index.deinit();

    _ = try index.insert([4]f32{ 0.0, 0.0, 0.0, 0.0 }, 0);
    _ = try index.insert([4]f32{ 10.0, 10.0, 10.0, 10.0 }, 1);

    const query = [4]f32{ 0.1, 0.1, 0.1, 0.1 };
    const results = index.searchKnn(&query, 1, 10);
    try std.testing.expectEqual(@as(u32, 0), results[0].id);
}

test "insert 100 vectors, brute-force verify k=1" {
    const Dist = distance.L2Squared(8);
    const Index = HnswIndex(8, Dist);
    var index = try Index.init(std.testing.allocator, 200, 64);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();

    for (0..100) |i| {
        var vec: [8]f32 = undefined;
        for (&vec) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
        _ = try index.insert(vec, @intCast(i));
    }

    var correct: u32 = 0;
    for (0..20) |_| {
        var query: [8]f32 = undefined;
        for (&query) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

        const results = index.searchKnn(&query, 1, 50);
        if (results.len == 0) continue;

        var best_id: u32 = 0;
        var best_dist = Dist.distance(&query, &index.vectors[0]);
        for (1..index.num_nodes) |i| {
            const d = Dist.distance(&query, &index.vectors[i]);
            if (d < best_dist) {
                best_dist = d;
                best_id = @intCast(i);
            }
        }

        if (results[0].id == best_id) correct += 1;
    }

    try std.testing.expect(correct >= 16);
}

test "insert 1000 vectors, recall@10 > 0.9" {
    const Dist = distance.L2Squared(8);
    const Index = HnswIndex(8, Dist);
    var index = try Index.init(std.testing.allocator, 1500, 100);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(456);
    const rng = prng.random();

    for (0..1000) |i| {
        var vec: [8]f32 = undefined;
        for (&vec) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
        _ = try index.insert(vec, @intCast(i));
    }

    const k: u32 = 10;
    var total_recall: f32 = 0;
    const num_queries: u32 = 20;

    for (0..num_queries) |_| {
        var query: [8]f32 = undefined;
        for (&query) |*v| v.* = rng.float(f32) * 2.0 - 1.0;

        const results = index.searchKnn(&query, k, 100);

        var all_dists: [1000]Entry = undefined;
        for (0..1000) |i| {
            all_dists[i] = .{
                .dist = Dist.distance(&query, &index.vectors[i]),
                .id = @intCast(i),
            };
        }
        std.sort.pdq(Entry, &all_dists, {}, struct {
            fn f(_: void, a: Entry, b: Entry) bool {
                return a.dist < b.dist;
            }
        }.f);

        var hits: u32 = 0;
        for (results) |r| {
            for (all_dists[0..k]) |gt| {
                if (r.id == gt.id) {
                    hits += 1;
                    break;
                }
            }
        }

        total_recall += @as(f32, @floatFromInt(hits)) / @as(f32, @floatFromInt(k));
    }

    const avg_recall = total_recall / @as(f32, @floatFromInt(num_queries));
    try std.testing.expect(avg_recall >= 0.9);
}

test "search on empty index returns empty" {
    const Index = HnswIndex(4, distance.L2Squared(4));
    var index = try Index.init(std.testing.allocator, 100, 32);
    defer index.deinit();

    const query = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    const results = index.searchKnn(&query, 5, 10);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "save and load round-trip preserves search results" {
    const Dist = distance.L2Squared(8);
    const Index = HnswIndex(8, Dist);
    var index = try Index.init(std.testing.allocator, 200, 64);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(789);
    const rng = prng.random();

    for (0..100) |i| {
        var vec: [8]f32 = undefined;
        for (&vec) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
        _ = try index.insert(vec, @intCast(i));
    }

    var query: [8]f32 = undefined;
    for (&query) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
    const before = index.searchKnn(&query, 5, 50);
    var before_ids: [5]u32 = undefined;
    for (before, 0..) |r, i| before_ids[i] = r.id;

    const path = "/tmp/hnsw_test_roundtrip.bin";
    try index.save(path);

    var loaded = try Index.load(std.testing.allocator, path);
    defer loaded.deinit();

    try std.testing.expectEqual(index.num_nodes, loaded.num_nodes);
    try std.testing.expectEqual(index.entry_point, loaded.entry_point);
    try std.testing.expectEqual(index.max_nodes, loaded.max_nodes);

    const after = loaded.searchKnn(&query, 5, 50);
    try std.testing.expectEqual(before.len, after.len);
    for (after, 0..) |r, i| {
        try std.testing.expectEqual(before_ids[i], r.id);
    }

    try std.fs.cwd().deleteFile(path);
}

test "load rejects wrong magic" {
    const Index = HnswIndex(4, distance.L2Squared(4));

    const path = "/tmp/hnsw_test_bad_magic.bin";
    const file = try std.fs.cwd().createFile(path, .{});
    var header: Index.Header = undefined;
    @memset(std.mem.asBytes(&header), 0);
    header.magic = 0xDEADBEEF;
    try file.writeAll(std.mem.asBytes(&header));
    file.close();
    defer std.fs.cwd().deleteFile(path) catch {};

    const result = Index.load(std.testing.allocator, path);
    try std.testing.expectError(error.InvalidFormat, result);
}
