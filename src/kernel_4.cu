#define A(i,j) A[(i) + (j)*m]
#define B(i,j) B[(i) + (j)*k]
#define C(i,j) C[(i) + (j)*m]

const int MS = 128;
const int NS = 128;
const int KS = 8;

#define As(li,lj) As[(li) + (lj)*MS]
#define Bs(li,lj) Bs[(li) + (lj)*MS]

#define VLOAD(v, address)\
    v = *((float4 *)(address));

#define VSTORE(address, v)\
    *((float4 *)(address)) = v;

// y = y + a*x, with x, y = float4 and a = float
#define AXPY(vec_y, vec_x, a)\
    vec_y.x += a * vec_x.x;\
    vec_y.y += a * vec_x.y;\
    vec_y.z += a * vec_x.z;\
    vec_y.w += a * vec_x.w;

// matrices A, B, and C have column-major storage
// limitations: 
//  - call this kernel with (16 x 16) threads per block
//  - ensure m, n are multiples of 128 and k is a multiple of 8
__global__ __launch_bounds__(256)
void sgemm_kernel_4(const float* A, const float* B, float* C, int m, int n, int k) 
{
    // convert 2D threadIdx.{x,y} to 1D tIdx [0..256)
    int tIdx = threadIdx.y * blockDim.x + threadIdx.x;

    // global indices for block of C
    const int bi = blockIdx.x * MS;
    const int bj = blockIdx.y * NS;

    // local indices for each thread to access A (32 x 8 threads)
    int li_a = 4 * (tIdx % 32);     // 0, 4, 8, ..., 124, 0, 4, 8, ...
    int lj_a = (tIdx / 32);         // 0...0, 1...1,  ..., 7...7

    // local indices for each thread to access B (2 x 128 threads)
    int li_b = 4 * (tIdx % 2);      // 0, 4, 0, 4, ...
    int lj_b = (tIdx / 2);          // 0, 0, 1, 1, ..., 127, 127

    // local indices for each thread to access C (8 x 8 threads)
    int li_c = 8 * (tIdx % 16);     // 0, 8, 16, ..., 124, 0, 8, ...
    int lj_c = 8 * (tIdx / 16);     // 0...0, 8...8, 16...16, ...

    // shared-memory buffers
    __shared__ float As[MS * KS];   // 128 x 8 = 1024 elements of A
    __shared__ float Bs[KS * NS];   // 8 x 128 = 1024 elements of B

    // registers    
    float4 Cr[16];                  // 8 x 8 submatrix of C
    memset(Cr, 0, sizeof(Cr));      // clear registers
    float4 Ar1, Ar2, Br1, Br2;      // 8 x 1 column of A, 1 x 8 row of B

    for (int p = 0; p < k; p += KS) {
        // thread loads four elements from a block of A (128 x 8) and stores in As (128 x 8)
        VLOAD(((float4 *)As)[tIdx], &A(bi + li_a, p + lj_a));

        // thread loads four elements from a block of B (8 x 128) and stores in Bs in transposed 
        // form, i.e. (128 x 8). Transposed form allows vectorized access to rows of B.      
        VLOAD(Br1, &B(p + li_b, bj + lj_b));
        Bs(lj_b, li_b    ) = Br1.x;
        Bs(lj_b, li_b + 1) = Br1.y;
        Bs(lj_b, li_b + 2) = Br1.z;
        Bs(lj_b, li_b + 3) = Br1.w;

        // synchronize threads in CTA to ensure that As and Bs are all loaded
        __syncthreads();

        #pragma unroll
        for (int lp = 0; lp < KS; lp++) {
            // Load one column (8x1) of As to registers
            VLOAD(Ar1, &As(li_c    , lp))
            VLOAD(Ar2, &As(li_c + 4, lp))
            // Load one row (1x8) of Bs to registers (Bs is stored in transposed form!)
            VLOAD(Br1, &Bs(lj_c    , lp))
            VLOAD(Br2, &Bs(lj_c + 4, lp))

            // compute the outer product Cr += Ar x Br with dim (8x8) = (8x1) x (1x8)
            AXPY(Cr[0] , Ar1, Br1.x)
            AXPY(Cr[1] , Ar2, Br1.x)
            AXPY(Cr[2] , Ar1, Br1.y)
            AXPY(Cr[3] , Ar2, Br1.y)
            AXPY(Cr[4] , Ar1, Br1.z)
            AXPY(Cr[5] , Ar2, Br1.z)
            AXPY(Cr[6] , Ar1, Br1.w)
            AXPY(Cr[7] , Ar2, Br1.w)
            AXPY(Cr[8] , Ar1, Br2.x)
            AXPY(Cr[9] , Ar2, Br2.x)
            AXPY(Cr[10], Ar1, Br2.y)
            AXPY(Cr[11], Ar2, Br2.y)
            AXPY(Cr[12], Ar1, Br2.z)
            AXPY(Cr[13], Ar2, Br2.z)
            AXPY(Cr[14], Ar1, Br2.w)
            AXPY(Cr[15], Ar2, Br2.w)            
        }

        // synchronize thread in CTA to ensure that work is done before
        // overwriting As and Bs with new data
        __syncthreads();
    }

    // compute global indices for each thread in matrix C
    const int i_c = bi + li_c;
    const int j_c = bj + lj_c;

    // store the 8x8 block of C by 16 VSTORE calls
    VSTORE(&C(i_c    , j_c    ), Cr[0])
    VSTORE(&C(i_c + 4, j_c    ), Cr[1])
    VSTORE(&C(i_c    , j_c + 1), Cr[2])
    VSTORE(&C(i_c + 4, j_c + 1), Cr[3])
    VSTORE(&C(i_c    , j_c + 2), Cr[4])
    VSTORE(&C(i_c + 4, j_c + 2), Cr[5])
    VSTORE(&C(i_c    , j_c + 3), Cr[6])
    VSTORE(&C(i_c + 4, j_c + 3), Cr[7])
    VSTORE(&C(i_c    , j_c + 4), Cr[8])
    VSTORE(&C(i_c + 4, j_c + 4), Cr[9])
    VSTORE(&C(i_c    , j_c + 5), Cr[10])
    VSTORE(&C(i_c + 4, j_c + 5), Cr[11])
    VSTORE(&C(i_c    , j_c + 6), Cr[12])
    VSTORE(&C(i_c + 4, j_c + 6), Cr[13])
    VSTORE(&C(i_c    , j_c + 7), Cr[14])
    VSTORE(&C(i_c + 4, j_c + 7), Cr[15])
}
