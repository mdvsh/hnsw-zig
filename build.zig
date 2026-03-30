const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable
    const exe = b.addExecutable(.{
        .name = "hnsw-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    // Shared library for Python/ctypes integration (ann-benchmarks)
    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "hnsw_zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c_api.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(lib);

    // Tests — one compilation unit per source file
    const test_files = [_][]const u8{
        "src/distance.zig",
        "src/bitset.zig",
        "src/heap.zig",
        "src/hnsw.zig",
    };

    const test_step = b.step("test", "Run unit tests");
    for (test_files) |file| {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(file),
                .target = target,
                .optimize = optimize,
            }),
        });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // Benchmark — always ReleaseFast regardless of -Doptimize
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_distance.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("distance", b.createModule(.{
        .root_source_file = b.path("src/distance.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    }));
    const bench_exe = b.addExecutable(.{
        .name = "bench-distance",
        .root_module = bench_mod,
    });
    const bench_step = b.step("bench", "Run distance benchmarks");
    bench_step.dependOn(&b.addRunArtifact(bench_exe).step);
}
