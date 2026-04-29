# cuSgemm — Optimized SGEMM Kernels for CUDA

A collection of seven iterative CUDA kernels implementing single-precision general matrix multiplication (SGEMM), benchmarked against NVIDIA's cuBLAS library. Each kernel builds on the previous one, progressively applying GPU optimization techniques to approach peak hardware performance.

**Goal:** Demonstrate how algorithmic and architectural optimizations — from naive direct mapping to shared-memory tiling, vectorized access, double buffering, and Tensor Cores — enable hand-written kernels to close the gap with highly tuned library implementations like cuBLAS.

## Requirements

- NVIDIA GPU (compute capability 8.0, 8.6, 8.9, or 12.0 — Ampere / Ada Lovelace / Blackwell)
- NVIDIA CUDA Toolkit ≥ 12.x
- CMake ≥ 3.18

## Build & Run

```bash
cd cuSgemm
mkdir build && cd build
cmake ..
make -j$(nproc)
./sgemm
```

The benchmark runs square matrix multiplications from 256×256 up to 2048×2048, comparing all seven kernels against cuBLAS at each size. Results are printed as GFLOPS/s.

## Kernels

| # | Name | Key Technique | Block Size | Tile (M×N×K) | Vectorization | Shared Memory |
|---|---|---|---|---|---|---|
| 1 | Naive | Direct 1:1 thread-to-output mapping | 32×32 | — | None | No |
| 2 | Coalesced Access | Swapped thread→dimension mapping for coalesced global memory reads/writes | 32×32 | — | None | No |
| 3 | Shared-Memory Tiling | 2D shared-memory tiles with explicit load/compute synchronization | 32×32 | 32×32×32 | None | Yes (BS²) |
| 4 | Vectorized + Transposed B | `float4` loads/stores, transposed shared-memory layout for B, 8×8 register tiling | 16×16 | 128×128×8 | float4 | Yes |
| 5 | Improved C-Indexing | Better warp/lane assignment to C registers for coalesced output writes | 16×16 | 128×128×8 | float4 | Yes |
| 6 | Double Buffering | Ping-pong shared-memory staging to overlap global memory loads with compute | 16×16 | 128×128×8 | float4 | Yes (2×) |
| 7 | TF32 Tensor Cores | `mma.sync.m16n8k4` inline PTX, warp-level 16×8 output tiles, K=4 shared-memory tiles with double buffering | 32 | 16×16×4 | TF32 | Yes (2×) |

### Kernel Descriptions

**Kernel 1 — Naive:** Each thread computes one element of C via a dot product over A and B. Thread x maps to the column index and thread y maps to the row index, resulting in strided (uncoalesced) reads from A and B, and an uncoalesced write to C.

**Kernel 2 — Coalesced Access:** Swaps the thread-to-dimension mapping so that x maps to rows and y maps to columns. This aligns threads within a warp to consecutive memory addresses, enabling coalesced global memory access for reads from A, reads from B (via warp broadcasting), and writes to C.

**Kernel 3 — Shared-Memory Tiling:** Loads 32×32 tiles of A and B into shared memory, where each thread copies one element. Elements are reused within the tile via `__syncthreads()`, dramatically reducing global memory traffic.

**Kernel 4 — Vectorized + Transposed B:** Uses a 2D block of 16×16 threads to compute a 128×128 subblock of C. A is loaded with `float4` vectorized loads. B is transposed into shared memory during load so that subsequent row access is also vectorized. The 8×8 register tiling maximizes arithmetic throughput.

**Kernel 5 — Improved C-Indexing:** Retains the vectorization strategy of Kernel 4 but improves the thread-to-register mapping for writing C. Warp and lane IDs are computed explicitly to produce coalesced `float4` output stores.

**Kernel 6 — Double Buffering:** Adds a ping-pong double buffer to shared memory (two stages for A and B). The next tile is prefetched into the non-active stage while the current stage is being used for computation, overlapping global memory latency with arithmetic.

**Kernel 7 — TF32 Tensor Cores:** Uses inline PTX `mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32` to leverage Blackwell TF32 Tensor Cores. Each warp (32 threads, `__launch_bounds__(32)`) computes one 16×8 tile of C. Threads are organized into 8 groups of 4 lanes; each lane holds 2 TF32 registers for A, 1 for B, and 4 FP32 accumulators. A 16×4 tile of A and a 4×8 tile of B are loaded from global memory into double-buffered shared memory in steps of K=4, streamed across the K dimension. Input FP32 is converted to TF32 via `cvt.rna.tf32.f32` before MMA. Grid layout: `(N/8, M/16)` blocks of 32 threads. Column-major matrix layout matches cuBLAS. Requires `sm_120` or higher.

## Benchmark Results

Performance in GFLOPS/s across increasing matrix sizes (all kernels run on the same GPU, GB10 / Blackwell):

| Size | cuBLAS | K1 | K2 | K3 | K4 | K5 | K6 | K7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256×256 | 2194 | 152 | 1079 | 1364 | 851 | 891 | 952 | **2295** |
| 512×512 | 7959 | 206 | 1597 | 2108 | 3822 | 3942 | 4137 | **4236** |
| 768×768 | 13677 | 208 | 1764 | 2216 | 8615 | 8979 | 9564 | 4322 |
| 1024×1024 | 15629 | 199 | 1782 | 2183 | 10648 | 11347 | 11737 | 3939 |
| 1280×1280 | 14869 | 194 | 1441 | 2317 | 8633 | 10291 | 9552 | 2824 |
| 1536×1536 | 17094 | 200 | 1669 | 1937 | 12340 | 13023 | 12955 | 2474 |
| 1792×1792 | 16370 | 210 | 1419 | 1844 | 11694 | 11602 | 12379 | 2882 |
| 2048×2048 | 18159 | 205 | 1560 | 2143 | 12631 | 14330 | **16595** | 2621 |

> **Note:** Kernel 7 (TF32 Tensor Cores) outperforms all other kernels at small-to-matrix sizes (256–512) where occupancy is sufficient for a single warp per 16×8 tile. At larger sizes, reduced warp throughput and limited per-block occupancy cause lower GFLOPS. Kernels 4–6 scale better due to larger 128×128 block-level tiles. Kernel performance may fluctuate due to L2/L3 cache effects and memory bandwidth saturation. cuBLAS itself varies significantly across sizes depending on which precompiled kernels it selects internally.

## Optimization Trajectory

The speedup from one kernel to the next highlights key GPU optimization principles:

| Transition | Improvement | Why |
|---|---|---|
| 1 → 2 | **7×** | Coalesced global memory access eliminates strided reads |
| 2 → 3 | **1.3×** | Shared-memory tiling reuses loaded data, reducing global bandwidth |
| 3 → 4 | **5.7×** | Vectorized `float4` loads + transposed B layout + larger tile (128×128) |
| 4 → 5 | **1.05×** | Improved coalescing on output writes to C |
| 5 → 6 | **1.03×** | Double buffering overlaps memory fetches with computation |
| 6 → 7 | **2.4× at 256²** | TF32 Tensor Cores (mma.sync.m16n8k4) deliver ~16 FP32 per warp per cycle |
| Best custom vs cuBLAS | **K7 beats cuBLAS at 256² (2295 vs 2194 GFLOPS)** | Tensor Cores + warp-level tiles dominate at small matrices; cuBLAS leads at large sizes via multi-block scheduling |

## Project Structure

```
cuSgemm/
├── CMakeLists.txt      # Build configuration (CUDA architectures 80/86/89/120)
├── src/
│   ├── main.cu          # Benchmark harness, cuBLAS comparison, correctness checks
│   ├── kernel_1.cu      # Naive direct mapping
│   ├── kernel_2.cu      # Coalesced global memory access
│   ├── kernel_3.cu      # Shared-memory tiling (32×32 tiles)
│   ├── kernel_4.cu      # Vectorized loads, transposed B, 128×128×8 blocks
│   ├── kernel_5.cu      # Improved output coalescing
│   ├── kernel_6.cu      # Double-buffered shared memory
│   └── kernel_7.cu      # TF32 Tensor Core GEMM (mma.sync.m16n8k4, sm_120)
```

## License

Educational / academic use.
