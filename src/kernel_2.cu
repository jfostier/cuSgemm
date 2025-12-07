#define A(i,j) A[(i) + (j)*m]
#define B(i,j) B[(i) + (j)*k]
#define C(i,j) C[(i) + (j)*m]

// matrices A, B, and C have column-major storage
__global__ void sgemm_kernel_2(const float* A, const float* B, float* C, int m, int n, int k) 
{
    // better choice: x-axis = vertical = row idx; y-axis = horizontal = column idx
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        float el = 0.0f;
        for (int p = 0; p < k; ++p) {
            // coalesced-memory access to A
            // warp-level broadcasting when accessing B
            el += A(i, p) * B(p, j);
        }
        // coalesced-memory access to C
        C(i, j) = el;
    }
}
