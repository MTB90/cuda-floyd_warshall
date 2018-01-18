
// #include <nvfunctional>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_apsp.cuh"

#define CMCPYHTD cudaMemcpyHostToDevice
#define CMCPYDTH cudaMemcpyDeviceToHost

// CONSTS for compute capability
#define THREAD_WIDTH 2
#define BLOCK_WIDTH 16

#define INF     1061109567 // 3F 3F 3F 3F
#define NONE    -1

/**
struct cugraphAPSPTopology {
    unsigned int nvertex; // number of vertex in graph
    unsigned int pitch;
    int* pred;  // predecessors matrix
    int* graph; // graph matrix
};
*/

/** Cuda handle error, if err is not success print error and line in code
*
* @param status CUDA Error types
*/
#define HANDLE_ERROR(err) \
{ \
    if (err != cudaSuccess) \
    { \
        fprintf(stderr, "%s failed  at line %d \nError message: %s \n", \
            __FILE__, __LINE__ ,cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

static __global__ void fw_kernel(const unsigned int u, const unsigned int n, int * const d, int * const p) {
        int v1 = blockDim.y * blockIdx.y + threadIdx.y;
        int v2 = blockDim.x * blockIdx.x + threadIdx.x;

        if (v1 < n && v2 < n) {
                int newPath = d[v1 * n + u] + d[u * n + v2];
                int oldPath = d[v1 * n + v2];
                if (oldPath > newPath) {
                        d[v1 * n + v2] = newPath;
                        p[v1 * n + v2] = p[u * n + v2];
                }
        }
}

static int _cudaMoveMemoryToDevice(const std::unique_ptr<graphAPSPTopology>& input, int *g, int* p, cudaStream_t& cpyStream) {
    int n = input->nvertex;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
    cudaStatus =  cudaMalloc((void**)&g, n * n * sizeof(int));
    HANDLE_ERROR(cudaStatus);
    cudaStatus =  cudaMalloc((void**)&p, n * n * sizeof(int));
    HANDLE_ERROR(cudaStatus);

    // Copy input from host memory to GPU buffers.
    cudaStatus = cudaMemcpyAsync(g, input->graph.get(), n * n * sizeof(int), CMCPYHTD, cpyStream);
    cudaStatus = cudaMemcpyAsync(p, input->pred.get(), n * n * sizeof(int), CMCPYHTD, cpyStream);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    cudaStatus = cudaDeviceSynchronize();
    HANDLE_ERROR(cudaStatus);
    return cudaStatus;
}

static int _cudaMoveMemoryToHost(const std::unique_ptr<graphAPSPTopology>& output, int *g, int* p, cudaStream_t& cpyStream) {
    int n = output->nvertex;
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(output->graph.get(), g, n * n * sizeof(int), CMCPYDTH);
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaMemcpy(output->pred.get(), p, n * n * sizeof(int), CMCPYDTH);
    HANDLE_ERROR(cudaStatus);
    cudaDeviceSynchronize();

    cudaStatus = cudaFree(g);
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaFree(p);
    HANDLE_ERROR(cudaStatus);
    return cudaStatus;
}

/**
 * Naive implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
int cudaNaiveFW(const std::unique_ptr<graphAPSPTopology>& dataHost) {
    int *g = 0;
    int *p = 0;
    int n = dataHost->nvertex;
    cudaError_t cudaStatus;
    cudaStream_t cpyStream;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    HANDLE_ERROR(cudaStatus);

    // Create new stream to copy data
    cudaStatus = cudaStreamCreate(&cpyStream);
    _cudaMoveMemoryToDevice(dataHost, g, p, cpyStream);

    // Initialize the grid and block dimensions here
    dim3 dimGrid((n - 1) / BLOCK_WIDTH + 1, (n - 1) / BLOCK_WIDTH + 1, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    // TODO not fully implemented yet
    cudaFuncSetCacheConfig(fw_kernel, cudaFuncCachePreferL1);
    for(int u=0; u < n; ++u) {
        fw_kernel<<<dimGrid, dimBlock>>>(u, n, g, p);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    HANDLE_ERROR(cudaStatus);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    HANDLE_ERROR(cudaStatus);

    _cudaMoveMemoryToHost(dataHost, p, g, cpyStream);
    return cudaStatus;
}

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
int cudaBlockedFW(const std::unique_ptr<graphAPSPTopology>& dataHost) {
    int *d = 0;
    int *p = 0;
    int n = dataHost->nvertex;
    cudaStream_t cpyStream;
    cudaError_t cudaStatus;

    cudaStatus = cudaStreamCreate(&cpyStream);

    _cudaMoveMemoryToDevice(dataHost, d, p, cpyStream);
    // TODO not implemented yet
    _cudaMoveMemoryToHost(dataHost, p, d, cpyStream);
    return 0;
}
