# hnsw-zig

HNSW approximate nearest neighbor search in Zig. From-scratch implementation targeting competitive performance with hnswlib through better memory layout, comptime monomorphization, and quantized search paths.

## Performance (SIFT-1M, M=16, ef_construction=200, Apple M4 16GB)

| ef | recall@10 | QPS | p50 | p99 |
|----|-----------|-----|-----|-----|
| 50 | 0.963 | 34,510 | 31μs | 57μs |
| 100 | 0.984 | 20,019 | 52μs | 71μs |
| 200 | 0.994 | 10,916 | 93μs | 128μs |

Build time: 50s (10 threads). Single-threaded search.

## Design

Separate-arrays layout (SoA): vectors, neighbor lists, and metadata in distinct contiguous allocations. This optimizes for the search access pattern (scanning neighbors doesn't pollute cache with vector data) at the cost of slightly slower construction.

Search uses a three-tier quantization pipeline: SQ4 (4-bit, 64 bytes/vector) for graph traversal, SQ8 for intermediate ranking, float32 for final rescore. The graph traversal working set shrinks from 512MB to 64MB, fitting closer to L3.

Key techniques:
- comptime-monomorphized distance functions via `@Vector` SIMD (dim and metric are comptime params)
- Epoch-based visited set (O(1) reset between queries)
- BFS graph reorder for cache locality
- Batched prefetch pipeline: filter visited neighbors, then prefetch all unvisited vectors before computing distances
- Multi-threaded build with striped locks and atomic entry point updates

## Build

Requires Zig 0.15.x.

```
zig build -Doptimize=ReleaseFast
```

## Usage

Download [SIFT-1M](http://corpus-texmex.irisa.fr/):

```
mkdir -p data/sift1m && cd data/sift1m
curl -L -o sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz
```

Build, search, benchmark:

```
./zig-out/bin/hnsw-zig build --input data/sift1m/sift/sift_base.fvecs --output index.bin --ef 200
./zig-out/bin/hnsw-zig bench --index index.bin --queries data/sift1m/sift/sift_query.fvecs --k 10 --ef 50,100,200 --gt data/sift1m/sift/sift_groundtruth.ivecs
```

## Files

```
src/
  distance.zig    L2, inner product, normalize. SIMD via @Vector, comptime dim.
  bitset.zig      Fixed-capacity bitset for visited-node tracking.
  heap.zig        Fixed-capacity binary min/max heap.
  hnsw.zig        Index: insert, searchKnn, selectNeighbors, buildBatch, save/load, reorder.
  c_api.zig       C-compatible FFI for Python/ann-benchmarks integration.
  main.zig        CLI: build, search, bench commands.
build.zig         Executable + shared library targets.
```

## References

- Malkov, Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." IEEE TPAMI, 2018.
- [hnswlib](https://github.com/nmslib/hnswlib) (C++ reference implementation)
- [ann-benchmarks](http://ann-benchmarks.com/)
