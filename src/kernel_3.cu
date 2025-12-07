#define A(i,j) A[(i) + (j)*m]
#define B(i,j) B[(i) + (j)*k]
#define C(i,j) C[(i) + (j)*m]

#define BS 32

#define As(li,lj) As[(li) + (lj)*BS]
#define Bs(li,lj) Bs[(li) + (lj)*BS]

// matrices A, B, and C have column-major storage
// limitations: 
//  - call this kernel with (32 x 32) threads per block
//  - ensure m, n, and k are integer multiples of 32
__global__ __launch_bounds__(BS * BS)
void sgemm_kernel_3(const float* A, const float* B, float* C, int m, int n, int k) 
{
    // global coordinates (within matrix A and B)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // local coordinates (within shared-memory block)
    const int& li = threadIdx.x;
    const int& lj = threadIdx.y;

    // shared-memory declaration
    __shared__ float As[BS * BS];
    __shared__ float Bs[BS * BS];

    float tmp = 0;
    for (int p = 0; p < k; p += BS) {
        // each thread copies one element of A and B to resp. As and Bs
        As(li, lj) = A(i, p + lj);
        Bs(li, lj) = B(p + li, j);

        // synchronize threads in CTA to ensure that As and Bs are all loaded
        __syncthreads();

        #pragma unroll
        for (int lp = 0; lp < BS; lp++)
            tmp += As(li, lp) * Bs(lp, lj);

        // synchronize thread in CTA to ensure that work is done before
        // overwriting As and Bs with new data
        __syncthreads();
    }

    C(i, j) = tmp;
}
