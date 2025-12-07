#define A(i,j) A[(i) + (j)*m]
#define B(i,j) B[(i) + (j)*k]
#define C(i,j) C[(i) + (j)*m]

// matrices A, B, and C have column-major storage
__global__ void sgemm_kernel_1(const float* A, const float* B, float* C, int m, int n, int k) 
{
    // natural choice: x-axis = horizontal = column idx; y-axis = vertical = row idx
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n) {
        float el = 0.0f;
        for (int p = 0; p < k; ++p) {
            // thread suffers from strided-memory access to A
            // no memory coalescing when reading from B
            el += A(i, p) * B(p, j);
        }
        // no memory coalescing when writing to C
        C(i, j) = el;
    }
}
