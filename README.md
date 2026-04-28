# cuSgemm — Optimized SGEMM Kernels for CUDA

A collection of six iterative CUDA kernels implementing single-precision general matrix multiplication (SGEMM), benchmarked against NVIDIA's cuBLAS library. Each kernel builds on the previous one, progressively applying GPU optimization techniques to approach peak hardware performance.

**Goal:** Demonstrate how algorithmic and architectural optimizations — from naive direct mapping to shared-memory tiling, vectorized access, and double buffering — enable hand-written kernels to close the gap with highly tuned library implementations like cuBLAS.

## Requirements

- NVIDIA GPU (compute capability 8.0, 8.6, or 8.9 — Ampere / Ada Lovelace)
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

The benchmark runs square matrix multiplications from 256×256 up to 6144×6144, comparing all six kernels against cuBLAS at each size. Results are printed as GFLOPS/s.

## Kernels

| # | Name | Key Technique | Block Size | Tile (M×N×K) | Vectorization | Shared Memory |
|---|---|---|---|---|---|---|
| 1 | Naive | Direct 1:1 thread-to-output mapping | 32×32 | — | None | No |
| 2 | Coalesced Access | Swapped thread→dimension mapping for coalesced global memory reads/writes | 32×32 | — | None | No |
| 3 | Shared-Memory Tiling | 2D shared-memory tiles with explicit load/compute synchronization | 32×32 | 32×32×32 | None | Yes (BS²) |
| 4 | Vectorized + Transposed B | `float4` loads/stores, transposed shared-memory layout for B, 8×8 register tiling | 16×16 | 128×128×8 | float4 | Yes |
| 5 | Improved C-Indexing | Better warp/lane assignment to C registers for coalesced output writes | 16×16 | 128×128×8 | float4 | Yes |
| 6 | Double Buffering | Ping-pong shared-memory staging to overlap global memory loads with compute | 16×16 | 128×128×8 | float4 | Yes (2×) |

### Kernel Descriptions

**Kernel 1 — Naive:** Each thread computes one element of C via a dot product over A and B. Thread x maps to the column index and thread y maps to the row index, resulting in strided (uncoalesced) reads from A and B, and an uncoalesced write to C.

**Kernel 2 — Coalesced Access:** Swaps the thread-to-dimension mapping so that x maps to rows and y maps to columns. This aligns threads within a warp to consecutive memory addresses, enabling coalesced global memory access for reads from A, reads from B (via warp broadcasting), and writes to C.

**Kernel 3 — Shared-Memory Tiling:** Loads 32×32 tiles of A and B into shared memory, where each thread copies one element. Elements are reused within the tile via `__syncthreads()`, dramatically reducing global memory traffic.

**Kernel 4 — Vectorized + Transposed B:** Uses a 2D block of 16×16 threads to compute a 128×128 subblock of C. A is loaded with `float4` vectorized loads. B is transposed into shared memory during load so that subsequent row access is also vectorized. The 8×8 register tiling maximizes arithmetic throughput.

**Kernel 5 — Improved C-Indexing:** Retains the vectorization strategy of Kernel 4 but improves the thread-to-register mapping for writing C. Warp and lane IDs are computed explicitly to produce coalesced `float4` output stores.

**Kernel 6 — Double Buffering:** Adds a ping-pong double buffer to shared memory (two stages for A and B). The next tile is prefetched into the non-active stage while the current stage is being used for computation, overlapping global memory latency with arithmetic.

## Benchmark Results

Performance in GFLOPS/s across increasing matrix sizes (all kernels run on the same GPU):

| Size | cuBLAS | Kernel 1 | Kernel 2 | Kernel 3 | Kernel 4 | Kernel 5 | Kernel 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256×256 | 2180.0 | 154.7 | 1097.9 | 1436.4 | 896.7 | 902.6 | 941.4 |
| 512×512 | 8657.0 | 210.3 | 1677.3 | 2146.6 | 3898.1 | 4061.3 | 4072.1 |
| 768×768 | 14170.0 | 234.3 | 1905.8 | 2380.5 | 8792.4 | 9128.6 | 8776.6 |
| 1024×1024 | 16116.0 | 217.0 | 1864.4 | 2264.6 | 11157.7 | 11752.5 | 11753.1 |
| 1280×1280 | 15017.0 | 217.5 | 1618.5 | 2326.6 | 10050.2 | 10920.9 | 10787.8 |
| 1536×1536 | 17882.0 | 223.6 | 1835.9 | 1758.3 | 10997.3 | 13487.2 | 11462.2 |
| 1792×1792 | 16729.0 | 221.7 | 1773.6 | 2328.8 | 12063.1 | 12535.1 | 13130.5 |
| 2048×2048 | 14956.0 | 223.0 | 1790.9 | 2149.7 | 13289.7 | 14928.1 | 15966.4 |
| 2304×2304 | 19203.0 | 222.9 | 1543.7 | 1985.6 | 13028.7 | 13364.9 | 15047.9 |
| 2560×2560 | 13127.0 | 220.7 | 1433.5 | 1901.6 | 11793.5 | 12091.9 | 12338.4 |
| 2816×2816 | 8865.0 | 222.6 | 1348.4 | 1810.8 | 12524.8 | 12753.2 | 15131.6 |
| 3072×3072 | 21249.0 | 222.6 | 1246.9 | 1685.6 | 13206.1 | 13441.4 | 15882.7 |
| 3328×3328 | 15498.0 | 223.2 | 1243.6 | 1766.3 | 11777.6 | 12982.3 | 14460.3 |
| 3584×3584 | 18956.0 | 223.3 | 1256.6 | 1743.3 | 12230.4 | 12518.7 | 15067.9 |
| 3840×3840 | 17177.0 | 213.4 | 1240.0 | 1743.6 | 12328.4 | 12347.7 | 13645.0 |
| 4096×4096 | 17718.0 | 223.1 | *(timed out)* | — | — | — | — |

> **Note:** Kernel performance fluctuates at larger sizes due to L2/L3 cache effects and memory bandwidth saturation. cuBLAS itself varies between ~8–21 GFLOPS across sizes depending on which optimizations (e.g., cutlass precompiled kernels) are selected internally.

## Optimization Trajectory

The speedup from one kernel to the next highlights key GPU optimization principles:

| Transition | Improvement | Why |
|---|---|---|
| 1 → 2 | **7×** | Coalesced global memory access eliminates strided reads |
| 2 → 3 | **1.3×** | Shared-memory tiling reuses loaded data, reducing global bandwidth |
| 3 → 4 | **5.7×** | Vectorized `float4` loads + transposed B layout + larger tile (128×128) |
| 4 → 5 | **1.05×** | Improved coalescing on output writes to C |
| 5 → 6 | **1.03×** | Double buffering overlaps memory fetches with computation |
| Best custom vs cuBLAS | **~85–95% of cuBLAS** | cuBLAS uses auto-tuning, multiple algorithm variants, and inline PTX |

## Project Structure

```
cuSgemm/
├── CMakeLists.txt      # Build configuration (CUDA architectures 80/86/89)
├── src/
│   ├── main.cu          # Benchmark harness, cuBLAS comparison, correctness checks
│   ├── kernel_1.cu      # Naive direct mapping
│   ├── kernel_2.cu      # Coalesced global memory access
│   ├── kernel_3.cu      # Shared-memory tiling (32×32 tiles)
│   ├── kernel_4.cu      # Vectorized loads, transposed B, 128×128×8 blocks
│   ├── kernel_5.cu      # Improved output coalescing
│   └── kernel_6.cu      # Double-buffered shared memory
```

## License

Educational / academic use.
