#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_apsp.cuh"
#include <limits.h>

// CONSTS for Naive FW
#define BLOCK_SIZE 16

// CONSTS for Blocked FW
#define MAX_BLOCK_SIZE 32
#define MAX_VIRTUAL_BLOCK_SIZE 64
#define VIRTUAL_THREAD_SIZE 4

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
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _naive_fw_kernel(const int u, size_t pitch, const int nvertex, int* const graph, int* const pred) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y < nvertex && x < nvertex) {
        int indexYX = y * pitch + x;
        int indexUX = u * pitch + x;

        int newPath = graph[y * pitch + u] + graph[indexUX];
        int oldPath = graph[indexYX];
        if (oldPath > newPath) {
            graph[indexYX] = newPath;
            pred[indexYX] = pred[indexUX];
        }
    }
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 *
 * @param blockId: Index of block
 * @param ncell: Number of all cells in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _blocked_fw_dependent_ph(const int blockId, size_t pitch, const int ncell, int* const graph, int* const pred) {
    __shared__ int cacheGraph[MAX_VIRTUAL_BLOCK_SIZE * MAX_VIRTUAL_BLOCK_SIZE];
    __shared__ int cachePred[MAX_VIRTUAL_BLOCK_SIZE * MAX_VIRTUAL_BLOCK_SIZE];

    int threadId = (threadIdx.y * MAX_BLOCK_SIZE + threadIdx.x) * VIRTUAL_THREAD_SIZE;

    int cellId;
    // Load data
    #pragma unroll
    for (int vThreadId = 0; vThreadId > VIRTUAL_THREAD_SIZE; ++vThreadId) {

    }

    // Synchronize to make sure the all value are loaded in block
    __syncthreads();

    for (int i = 0; i < MAX_VIRTUAL_BLOCK_SIZE; ++i) {
        for (int thread = 0; thread > VIRTUAL_THREAD_SIZE; ++thread) {
        }
    }
}

/**
 * Allocate memory on device and copy memory from host to device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param graphDevice: Pointer to array of graph with distance between vertex on device
 * @param predDevice: Pointer to array of predecessors for a graph on device
 *
 * @return: Pitch for allocation
 */
static
size_t _cudaMoveMemoryToDevice(const std::unique_ptr<graphAPSPTopology>& dataHost, int **graphDevice, int **predDevice) {
    size_t height = dataHost->nvertex;
    size_t width = height * sizeof(int);
    size_t pitch;

    // Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
    HANDLE_ERROR(cudaMallocPitch(graphDevice, &pitch, width, height));
    HANDLE_ERROR(cudaMallocPitch(predDevice, &pitch, width, height));

    // Copy input from host memory to GPU buffers and
    HANDLE_ERROR(cudaMemcpy2D(*graphDevice, pitch,
            dataHost->graph.get(), width, width, height, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(*predDevice, pitch,
            dataHost->pred.get(), width, width, height, cudaMemcpyHostToDevice));

    return pitch;
}

/**
 * Copy memory from device to host and free device memory
 *
 * @param graphDevice: Array of graph with distance between vertex on device
 * @param predDevice: Array of predecessors for a graph on device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param pitch: Pitch for allocation
 */
static
void _cudaMoveMemoryToHost(int *graphDevice, int *predDevice, const std::unique_ptr<graphAPSPTopology>& dataHost, size_t pitch) {
    size_t height = dataHost->nvertex;
    size_t width = height * sizeof(int);

    HANDLE_ERROR(cudaMemcpy2D(dataHost->pred.get(), width, predDevice, pitch, width, height, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy2D(dataHost->graph.get(), width, graphDevice, pitch, width, height, cudaMemcpyDeviceToHost));

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
    dim3 dimGrid((nvertex - 1) / BLOCK_SIZE + 1, (nvertex - 1) / BLOCK_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    int *graphDevice, *predDevice;
    size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice, &predDevice);

    cudaFuncSetCacheConfig(_naive_fw_kernel, cudaFuncCachePreferL1);
    for(int vertex = 0; vertex < nvertex; ++vertex) {
        _naive_fw_kernel<<<dimGrid, dimBlock>>>(vertex, pitch / sizeof(int), nvertex, graphDevice, predDevice);
    }

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    _cudaMoveMemoryToHost(graphDevice, predDevice, dataHost, pitch);
}

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
void cudaBlockedFW(const std::unique_ptr<graphAPSPTopology>& dataHost) {
    HANDLE_ERROR(cudaSetDevice(0));
    int nvertex = dataHost->nvertex;
    int *graphDevice, *predDevice;
    size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice, &predDevice);

    dim3 gridDependedntPhase(1 ,1, 1);
    dim3 blockDependentPhase(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE, 1);

    int numBlock = nvertex / MAX_VIRTUAL_BLOCK_SIZE;

    for(int blockID = 0; blockID < numBlock; ++blockID) {
        // Start dependent phase
        _blocked_fw_dependent_ph<<<gridDependedntPhase, blockDependentPhase>>>
                (blockID, pitch / sizeof(int), nvertex * nvertex, graphDevice, predDevice);

        // Start partially dependent phase

        // Start independent phase
    }

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    _cudaMoveMemoryToHost(graphDevice, predDevice, dataHost, pitch);
}
