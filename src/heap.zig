const std = @import("std");

pub const Entry = struct { dist: f32, id: u32 };

pub fn BoundedHeap(comptime order: enum { min, max }) type {
    return struct {
        const Self = @This();

        items: []Entry,
        len: usize,

        pub fn init(buffer: []Entry) Self {
            return .{ .items = buffer, .len = 0 };
        }

        pub fn insert(self: *Self, entry: Entry) void {
            std.debug.assert(self.len < self.items.len);
            self.items[self.len] = entry;
            self.siftUp(self.len);
            self.len += 1;
        }

        pub fn pop(self: *Self) ?Entry {
            if (self.len == 0) return null;
            const top = self.items[0];
            self.len -= 1;
            if (self.len > 0) {
                self.items[0] = self.items[self.len];
                self.siftDown(0);
            }
            return top;
        }

        pub fn peek(self: *const Self) ?Entry {
            return if (self.len == 0) null else self.items[0];
        }

        /// Replace the top element and restore heap invariant.
        /// Caller must ensure heap is non-empty.
        pub fn replaceTop(self: *Self, entry: Entry) void {
            std.debug.assert(self.len > 0);
            self.items[0] = entry;
            self.siftDown(0);
        }

        pub fn clear(self: *Self) void {
            self.len = 0;
        }

        inline fn siftUp(self: *Self, start: usize) void {
            var idx = start;
            while (idx > 0) {
                const parent = (idx - 1) / 2;
                if (better(self.items[idx], self.items[parent])) {
                    const tmp = self.items[idx];
                    self.items[idx] = self.items[parent];
                    self.items[parent] = tmp;
                    idx = parent;
                } else break;
            }
        }

        inline fn siftDown(self: *Self, start: usize) void {
            var idx = start;
            while (true) {
                var best = idx;
                const left = 2 * idx + 1;
                const right = 2 * idx + 2;
                if (left < self.len and better(self.items[left], self.items[best])) {
                    best = left;
                }
                if (right < self.len and better(self.items[right], self.items[best])) {
                    best = right;
                }
                if (best == idx) break;
                const tmp = self.items[idx];
                self.items[idx] = self.items[best];
                self.items[best] = tmp;
                idx = best;
            }
        }

        inline fn better(a: Entry, b: Entry) bool {
            return switch (order) {
                .min => a.dist < b.dist,
                .max => a.dist > b.dist,
            };
        }
    };
}

pub const BoundedMinHeap = BoundedHeap(.min);
pub const BoundedMaxHeap = BoundedHeap(.max);

// Tests

test "min-heap pop returns ascending order" {
    var buf: [8]Entry = undefined;
    var h = BoundedMinHeap.init(&buf);
    h.insert(.{ .dist = 3.0, .id = 3 });
    h.insert(.{ .dist = 1.0, .id = 1 });
    h.insert(.{ .dist = 4.0, .id = 4 });
    h.insert(.{ .dist = 0.5, .id = 0 });
    h.insert(.{ .dist = 2.0, .id = 2 });

    const expected = [_]f32{ 0.5, 1.0, 2.0, 3.0, 4.0 };
    for (expected) |e| {
        const got = h.pop().?;
        try std.testing.expectEqual(e, got.dist);
    }
}

test "max-heap pop returns descending order" {
    var buf: [8]Entry = undefined;
    var h = BoundedMaxHeap.init(&buf);
    h.insert(.{ .dist = 3.0, .id = 3 });
    h.insert(.{ .dist = 1.0, .id = 1 });
    h.insert(.{ .dist = 4.0, .id = 4 });
    h.insert(.{ .dist = 0.5, .id = 0 });
    h.insert(.{ .dist = 2.0, .id = 2 });

    const expected = [_]f32{ 4.0, 3.0, 2.0, 1.0, 0.5 };
    for (expected) |e| {
        const got = h.pop().?;
        try std.testing.expectEqual(e, got.dist);
    }
}

test "replaceTop on full max-heap" {
    var buf: [3]Entry = undefined;
    var h = BoundedMaxHeap.init(&buf);
    h.insert(.{ .dist = 10.0, .id = 0 });
    h.insert(.{ .dist = 20.0, .id = 1 });
    h.insert(.{ .dist = 15.0, .id = 2 });

    // Top is 20.0. Replace with 5.0 (closer).
    h.replaceTop(.{ .dist = 5.0, .id = 3 });

    // Max-heap should now yield 15, 10, 5
    const expected = [_]f32{ 15.0, 10.0, 5.0 };
    for (expected) |e| {
        const got = h.pop().?;
        try std.testing.expectEqual(e, got.dist);
    }
}

test "peek does not modify heap" {
    var buf: [4]Entry = undefined;
    var h = BoundedMinHeap.init(&buf);
    h.insert(.{ .dist = 5.0, .id = 0 });
    h.insert(.{ .dist = 2.0, .id = 1 });

    const first = h.peek().?;
    const second = h.peek().?;
    try std.testing.expectEqual(first.dist, second.dist);
    try std.testing.expectEqual(first.id, second.id);
    try std.testing.expectEqual(@as(usize, 2), h.len);
}

test "pop on empty returns null" {
    var buf: [4]Entry = undefined;
    var h = BoundedMinHeap.init(&buf);
    try std.testing.expectEqual(@as(?Entry, null), h.pop());
}

test "peek on empty returns null" {
    var buf: [4]Entry = undefined;
    var h = BoundedMaxHeap.init(&buf);
    try std.testing.expectEqual(@as(?Entry, null), h.peek());
}

test "capacity: insert exactly cap elements then pop all" {
    var buf: [5]Entry = undefined;
    var h = BoundedMinHeap.init(&buf);
    h.insert(.{ .dist = 9.0, .id = 9 });
    h.insert(.{ .dist = 1.0, .id = 1 });
    h.insert(.{ .dist = 5.0, .id = 5 });
    h.insert(.{ .dist = 3.0, .id = 3 });
    h.insert(.{ .dist = 7.0, .id = 7 });

    const expected = [_]f32{ 1.0, 3.0, 5.0, 7.0, 9.0 };
    for (expected) |e| {
        try std.testing.expectEqual(e, h.pop().?.dist);
    }
    try std.testing.expectEqual(@as(?Entry, null), h.pop());
}

test "insert and pop interleaved" {
    var buf: [8]Entry = undefined;
    var h = BoundedMinHeap.init(&buf);

    // Insert 3
    h.insert(.{ .dist = 5.0, .id = 5 });
    h.insert(.{ .dist = 2.0, .id = 2 });
    h.insert(.{ .dist = 8.0, .id = 8 });

    // Pop 1 (should be 2.0)
    try std.testing.expectEqual(@as(f32, 2.0), h.pop().?.dist);

    // Insert 2 more
    h.insert(.{ .dist = 1.0, .id = 1 });
    h.insert(.{ .dist = 6.0, .id = 6 });

    // Pop all remaining: 1, 5, 6, 8
    const expected = [_]f32{ 1.0, 5.0, 6.0, 8.0 };
    for (expected) |e| {
        try std.testing.expectEqual(e, h.pop().?.dist);
    }
    try std.testing.expectEqual(@as(?Entry, null), h.pop());
}

test "aliases BoundedMinHeap and BoundedMaxHeap" {
    // Verify the aliases produce working types
    var min_buf: [2]Entry = undefined;
    var max_buf: [2]Entry = undefined;
    var min_h = BoundedMinHeap.init(&min_buf);
    var max_h = BoundedMaxHeap.init(&max_buf);

    min_h.insert(.{ .dist = 3.0, .id = 0 });
    min_h.insert(.{ .dist = 1.0, .id = 1 });
    max_h.insert(.{ .dist = 3.0, .id = 0 });
    max_h.insert(.{ .dist = 1.0, .id = 1 });

    // Min-heap top is smallest, max-heap top is largest
    try std.testing.expectEqual(@as(f32, 1.0), min_h.peek().?.dist);
    try std.testing.expectEqual(@as(f32, 3.0), max_h.peek().?.dist);
}
