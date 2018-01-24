#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_apsp.cuh"

// CONSTS for compute capability
#define THREAD_WIDTH 2
#define BLOCK_WIDTH 16


/**
 * CUDA handle error, if error occurs print message and exit program
*
* @param error: CUDA error status
*/
#define HANDLE_ERROR(error) { \
    if (error != cudaSuccess) { \
        fprintf(stderr, "%s in %s at line %d\n", \
                cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} \

/**
 * Naive CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * check if path from vertex x -> y will be short using vertex u x -> u -> y
 * for all vertices in graph
 *
 * @param u: Index of vertex u
 * @param nvertex: Number of all vertex in graph
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _naive_fw_kernel(const int u, const int nvertex, int* const graph, int* const pred) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y < nvertex && x < nvertex) {
        int indexYX = y * nvertex + x;
        int indexUX = u * nvertex + x;

        int newPath = graph[y * nvertex + u] + graph[indexUX];
        int oldPath = graph[indexYX];
        if (oldPath > newPath) {
            graph[indexYX] = newPath;
            pred[indexYX] = pred[indexUX];
        }
    }
}

/**
 * Allocate memory on device and copy memory from host to device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param graphDevice: Pointer to array of graph with distance between vertex on device
 * @param predDevice: Pointer to array of predecessors for a graph on device
 */
static
void _cudaMoveMemoryToDevice(const std::unique_ptr<graphAPSPTopology>& hostData, int **graphDevice, int **predDevice) {
    int size = hostData->nvertex * hostData->nvertex * sizeof(int);

    // Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
    HANDLE_ERROR(cudaMalloc((void**)graphDevice, size));
    HANDLE_ERROR(cudaMalloc((void**)predDevice, size));

    // Copy input from host memory to GPU buffers and
    HANDLE_ERROR(cudaMemcpy(*graphDevice, hostData->graph.get(), size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(*predDevice, hostData->pred.get(), size, cudaMemcpyHostToDevice));
}

/**
 * Copy memory from device to host and free device memory
 *
 * @param graphDevice: Array of graph with distance between vertex on device
 * @param predDevice: Array of predecessors for a graph on device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 */
static
void _cudaMoveMemoryToHost(int *graphDevice, int *predDevice, const std::unique_ptr<graphAPSPTopology>& dataHost) {
    int size = dataHost->nvertex * dataHost->nvertex * sizeof(int);
    HANDLE_ERROR(cudaMemcpy(dataHost->pred.get(), predDevice, size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(dataHost->graph.get(), graphDevice, size, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(predDevice));
    HANDLE_ERROR(cudaFree(graphDevice));
}

/**
 * Naive implementation of Floyd Warshall algorithm in CUDA
 *
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 */
void cudaNaiveFW(const std::unique_ptr<graphAPSPTopology>& dataHost) {
    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));
    int nvertex = dataHost->nvertex;

    // Initialize the grid and block dimensions here
    dim3 dimGrid((nvertex - 1) / BLOCK_WIDTH + 1, (nvertex - 1) / BLOCK_WIDTH + 1, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // Move data from host do device
    int *graphDevice;
    int *predDevice;
    _cudaMoveMemoryToDevice(dataHost, &graphDevice, &predDevice);

    cudaFuncSetCacheConfig(_naive_fw_kernel, cudaFuncCachePreferL1);
    for(int u=0; u < nvertex; ++u) {
        _naive_fw_kernel<<<dimGrid, dimBlock>>>(u, nvertex, graphDevice, predDevice);
    }

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Move data from device to host
    _cudaMoveMemoryToHost(graphDevice, predDevice, dataHost);
}

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
void cudaBlockedFW(const std::unique_ptr<graphAPSPTopology>& dataHost) {
    // TODO not implemented yet
}
