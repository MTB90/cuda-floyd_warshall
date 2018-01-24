
// #include <nvfunctional>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_apsp.cuh"

// CONSTS for compute capability
#define THREAD_WIDTH 2
#define BLOCK_WIDTH 16

/* Default structure for graph in CUDA*/
struct cudaGraphAPSPTopology {
    int* pred;  // predecessors matrix
    int* graph; // graph matrix
    unsigned int nvertex; // number of vertex in graph
    size_t pitch;
};

/**
 * CUDA handle error, if error occurs print message and exit program
*
* @param error: CUDA error status
* @param file: Name of file where error occurs
* @param line: Number line where error occurs
*/
inline static
void _handleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n",
                cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}
/* Macro to handle CUDA error */
#define HANDLE_ERROR(err) (_handleError( err, __FILE__, __LINE__ ))

/**
 * Naive CUDA kernel implementation algorithm Floyd Wharshall for APSP
 *
 * @param dataDevice: graph data with allocated fields on device
 * @param u: Index of vertex u
 */
static __global__
void _naive_fw_kernel(const cudaGraphAPSPTopology *dataDevice, const unsigned int u) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y < dataDevice->nvertex && x < dataDevice->nvertex) {
        int indexYX = y * dataDevice->pitch + x;
        int indexUX = u * dataDevice->pitch + x;

        int path = dataDevice->graph[y * dataDevice->pitch + u] +
                dataDevice->graph[indexUX];
        if (dataDevice->graph[indexYX] > path) {
            dataDevice->graph[indexYX] = path;
            dataDevice->pred[indexYX] = dataDevice->pred[indexUX];
            }
        }
}

static
cudaGraphAPSPTopology *_cudaMoveMemoryToDevice(const std::unique_ptr<graphAPSPTopology>& input) {
    cudaGraphAPSPTopology *dataDevice = NULL;
    int size = input->nvertex * input->nvertex * sizeof(int);

    int *graph;
    int *pred;

    // Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
    HANDLE_ERROR(cudaMalloc((void**)&graph, size));
    HANDLE_ERROR(cudaMalloc((void**)&pred, size));

    // Copy input from host memory to GPU buffers
    HANDLE_ERROR(cudaMemcpyAsync(graph, input->graph.get(), size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyAsync(pred, input->pred.get(), size, cudaMemcpyHostToDevice));

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    HANDLE_ERROR(cudaDeviceSynchronize());
    return dataDevice;
}

static
void _cudaMoveMemoryToHost(const cudaGraphAPSPTopology *dataDevice, const std::unique_ptr<graphAPSPTopology>& output) {
    /*int n = output->nvertex;
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(output->graph.get(), g, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaMemcpy(output->pred.get(), p, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(cudaStatus);
    cudaDeviceSynchronize();

    cudaStatus = cudaFree(g);
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaFree(p);
    HANDLE_ERROR(cudaStatus);
    return cudaStatus; */
}

/**
 * Naive implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
void cudaNaiveFW(const std::unique_ptr<graphAPSPTopology>& dataHost) {
    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));

    // Initialize the grid and block dimensions here
    dim3 dimGrid((dataHost->nvertex - 1) / BLOCK_WIDTH + 1,
            (dataHost->nvertex - 1) / BLOCK_WIDTH + 1, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // Move data from host do device
    cudaGraphAPSPTopology *dataDevice = _cudaMoveMemoryToDevice(dataHost);

    cudaFuncSetCacheConfig(_naive_fw_kernel, cudaFuncCachePreferL1);
    for(int u=0; u < dataHost->nvertex; ++u) {
        _naive_fw_kernel<<<dimGrid, dimBlock>>>(dataDevice, u);
    }

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Move data from device to host
    _cudaMoveMemoryToHost(dataDevice, dataHost);
}

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
void cudaBlockedFW(const std::unique_ptr<graphAPSPTopology>& dataHost) {
    // TODO not implemented yet
}
