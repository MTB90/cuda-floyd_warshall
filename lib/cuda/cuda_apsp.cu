
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
    int size = input->nvertex * input->nvertex * sizeof(int);
    int *graph;
    int *pred;

    // Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
    HANDLE_ERROR(cudaMalloc((void**)&graph, size));
    HANDLE_ERROR(cudaMalloc((void**)&pred, size));

    // Copy input from host memory to GPU buffers and
    HANDLE_ERROR(cudaMemcpy(graph, input->graph.get(), size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pred, input->pred.get(), size, cudaMemcpyHostToDevice));

    cudaGraphAPSPTopology temp;
    temp.nvertex = input->nvertex;
    temp.pitch = temp.nvertex;
    temp.graph = graph;
    temp.pred = pred;

    // Copy structure from host to device
    cudaGraphAPSPTopology *dataDevice;
    HANDLE_ERROR(cudaMalloc((void**) &dataDevice, sizeof(cudaGraphAPSPTopology)));
    HANDLE_ERROR(cudaMemcpy(dataDevice, &temp,
            sizeof(cudaGraphAPSPTopology),
            cudaMemcpyHostToDevice));
    return dataDevice;
}

static
void _cudaMoveMemoryToHost(cudaGraphAPSPTopology *dataDevice, const std::unique_ptr<graphAPSPTopology>& output) {
    // Copy structure form device to host
    cudaGraphAPSPTopology temp;
    HANDLE_ERROR(cudaMemcpy(&temp, dataDevice,
            sizeof(cudaGraphAPSPTopology),
            cudaMemcpyDeviceToHost));

    int size = temp.nvertex * temp.nvertex * sizeof(int);
    HANDLE_ERROR(cudaMemcpy(output->pred.get(), temp.pred, size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(output->graph.get(), temp.graph, size, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(temp.pred));
    HANDLE_ERROR(cudaFree(temp.graph));
    HANDLE_ERROR(cudaFree(dataDevice));
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
