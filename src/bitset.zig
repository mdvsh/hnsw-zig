const std = @import("std");

pub const Bitset = struct {
    words: []u64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: u32) !Bitset {
        const num_words = (capacity + 63) >> 6; // ceil(capacity / 64)
        const words = try allocator.alloc(u64, num_words);
        @memset(words, 0);
        return .{ .words = words, .allocator = allocator };
    }

    pub fn deinit(self: *Bitset) void {
        self.allocator.free(self.words);
    }

    /// Returns true if bit was already set (test-and-set).
    pub inline fn testAndSet(self: *Bitset, id: u32) bool {
        const word_idx = id >> 6;
        const bit_mask = @as(u64, 1) << @truncate(id);
        const was_set = (self.words[word_idx] & bit_mask) != 0;
        self.words[word_idx] |= bit_mask;
        return was_set;
    }

    /// Read-only check: returns true if bit is set.
    pub inline fn isSet(self: *const Bitset, id: u32) bool {
        const word_idx = id >> 6;
        const bit_mask = @as(u64, 1) << @truncate(id);
        return (self.words[word_idx] & bit_mask) != 0;
    }

    /// Reset all bits to zero.
    pub fn clear(self: *Bitset) void {
        @memset(self.words, 0);
    }
};

/// Epoch-based visited set. O(1) reset via epoch increment instead of O(n) memset.
/// Uses 2 bytes per node (vs 1 bit for Bitset) but avoids 125KB memset on every
/// searchLayer call. Full memset only fires every 65534 resets.
pub const VisitedSet = struct {
    entries: []u16,
    epoch: u16,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: u32) !VisitedSet {
        const entries = try allocator.alloc(u16, capacity);
        @memset(entries, 0);
        return .{ .entries = entries, .epoch = 1, .allocator = allocator };
    }

    pub fn deinit(self: *VisitedSet) void {
        self.allocator.free(self.entries);
    }

    pub inline fn reset(self: *VisitedSet) void {
        self.epoch +%= 1;
        if (self.epoch == 0) {
            @memset(self.entries, 0);
            self.epoch = 1;
        }
    }

    pub inline fn testAndSet(self: *VisitedSet, id: u32) bool {
        const was_visited = self.entries[id] == self.epoch;
        self.entries[id] = self.epoch;
        return was_visited;
    }
};

// --- Tests ---

test "isSet returns false for unset bits, true after testAndSet" {
    var bs = try Bitset.init(std.testing.allocator, 256);
    defer bs.deinit();

    try std.testing.expect(!bs.isSet(42));
    _ = bs.testAndSet(42);
    try std.testing.expect(bs.isSet(42));
    try std.testing.expect(!bs.isSet(43));
}

test "testAndSet returns false first, true on repeat" {
    var bs = try Bitset.init(std.testing.allocator, 128);
    defer bs.deinit();

    try std.testing.expect(!bs.testAndSet(10));
    try std.testing.expect(bs.testAndSet(10));
    try std.testing.expect(bs.testAndSet(10));
}

test "clear resets all bits" {
    var bs = try Bitset.init(std.testing.allocator, 256);
    defer bs.deinit();

    _ = bs.testAndSet(0);
    _ = bs.testAndSet(100);
    _ = bs.testAndSet(255);
    try std.testing.expect(bs.isSet(0));
    try std.testing.expect(bs.isSet(100));
    try std.testing.expect(bs.isSet(255));

    bs.clear();
    try std.testing.expect(!bs.isSet(0));
    try std.testing.expect(!bs.isSet(100));
    try std.testing.expect(!bs.isSet(255));
}

test "boundary bits: 0, 63, 64, capacity-1" {
    const cap: u32 = 256;
    var bs = try Bitset.init(std.testing.allocator, cap);
    defer bs.deinit();

    const boundaries = [_]u32{ 0, 63, 64, cap - 1 };
    for (boundaries) |id| {
        try std.testing.expect(!bs.isSet(id));
        try std.testing.expect(!bs.testAndSet(id));
        try std.testing.expect(bs.isSet(id));
        try std.testing.expect(bs.testAndSet(id));
    }
}

test "large capacity: 1_000_000 nodes" {
    var bs = try Bitset.init(std.testing.allocator, 1_000_000);
    defer bs.deinit();

    _ = bs.testAndSet(0);
    _ = bs.testAndSet(500_000);
    _ = bs.testAndSet(999_999);
    try std.testing.expect(bs.isSet(0));
    try std.testing.expect(bs.isSet(500_000));
    try std.testing.expect(bs.isSet(999_999));
    try std.testing.expect(!bs.isSet(1));
    try std.testing.expect(!bs.isSet(999_998));

    bs.clear();
    try std.testing.expect(!bs.isSet(0));
    try std.testing.expect(!bs.isSet(500_000));
    try std.testing.expect(!bs.isSet(999_999));
}
