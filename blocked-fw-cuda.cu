#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

// CUDA Headers
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Helper definition
#define VAR(v, i) __typeof(i) v=(i)
#define FOR(i, j, k) for (int i = (j); i <= (k); ++i)
#define FORD(i, j, k)for (int i=(j); i >= (k); --i)
#define FORE(i, c) for(VAR(i, (c).begin()); i != (c).end(); ++i)
#define REP(i, n) for(int i = 0;i <(n); ++i)

// CONSTS
#define INF 	1061109567 // 3F 3F 3F 3F
#define CHARINF 63	   // 3F	
#define CHARBIT 8
#define CMCPYHTD cudaMemcpyHostToDevice
#define CMCPYDTH cudaMemcpyDeviceToHost

// CONSTS for compute capability 2.0
#define BLOCK_WIDTH 16
#define WARP 	    32

const bool PRINT = true; 	// print graf d or not

// MIN FUNCTION
#define MIN(x, y) y + ((x - y) & ((x - y) >> (sizeof(int) * CHARBIT - 1)))

/** Cuda handle error, if err is not success print error and line in code
*
* @param status CUDA Error types
*/
#define HANDLE_ERROR(status) \
{ \
	if (status != cudaSuccess) \
	{ \
		fprintf(stderr, "%s failed  at line %d \nError message: %s \n", \
			__FILE__, __LINE__ ,cudaGetErrorString(status)); \
		exit(EXIT_FAILURE); \
	} \
}

/**Kernel for wake gpu
*
* @param reps dummy variable only to perform some action
*/
__global__ void wake_gpu_kernel(int reps) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= reps)return;
}

/**Kernel for parallel Floyd Warshall algorithm on gpu compute of independent block
* 
* @param b number block of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
*/
template <int BLOCK_SIZE> __global__ void fw_kernel_phase_one(const unsigned int block, const unsigned int n, int *d)
{
	int newPath;

	int tx = threadIdx.x; 
	int ty = threadIdx.y;
	
	int v1 = BLOCK_SIZE * block + ty;
	int v2 = BLOCK_SIZE * block + tx;  

	__shared__ int primary[BLOCK_SIZE][BLOCK_SIZE];

	primary[ty][tx] = (v1 < n && v2 < n) ? d[v1 * n + v2] : INF;
	
	// Synchronize to make sure the all value are loaded in block
	__syncthreads();
	
	#pragma unroll
	FOR(i, 0, BLOCK_SIZE - 1)
	{
		newPath = primary[ty][i] + primary[i][tx];
		primary[ty][tx] = primary[ty][tx] > newPath ? newPath : primary[ty][tx];
		
		// Synchronize to make sure that all value are current
		__syncthreads();
	}

	if (v1 < n && v2 < n) 
	{
		d[v1 * n + v2] = primary[ty][tx];
	}
}

/**Kernel for parallel Floyd Warshall algorithm on gpu compute of singly dependent block depend on the independent block
  *
  * @param b number block of which is performed relaxation paths [v1, v2]
  * @param n number of vertices in the graph G:=(V,E), n := |V(G)|
  * @param d matrix of shortest paths d(G)
  */
template <int BLOCK_SIZE> __global__ void fw_kernel_phase_two(const unsigned int block, const unsigned int n, int *d)
{
	if (blockIdx.x == block) return;

        int newPath;
	int v1, v2;

	int tx = threadIdx.x;
        int ty = threadIdx.y;

	int pv1 = BLOCK_SIZE * block + ty;
        int pv2 = BLOCK_SIZE * block + tx;

	__shared__ int primary[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int current[BLOCK_SIZE][BLOCK_SIZE];

	primary[ty][tx] = (pv1 < n && pv2 < n) ? d[pv1 * n + pv2] : INF;
	
	// Load i-aligned singly dependent blocks
	if (blockIdx.y == 0)
	{
		v1 = BLOCK_SIZE * block + ty;
		v2 = BLOCK_SIZE * blockIdx.x + tx;
	}
	// Load j-aligned singly dependent blocks
	else 
	{
		v1 = BLOCK_SIZE * blockIdx.x + ty;
		v2 = BLOCK_SIZE * block + tx;
	}
	
	current[ty][tx] = (v1 < n && v2 < n) ? d[v1 * n + v2] : INF;

	// Synchronize to make sure the all value are loaded in block
	__syncthreads();

	// Compute i-aligned singly dependent blocks
	if (blockIdx.y == 0)
	{
		#pragma unroll
		FOR(i, 0, BLOCK_SIZE - 1)
		{
			newPath = primary[ty][i] + current[i][tx];
			current[ty][tx] = current[ty][tx] > newPath ? newPath : current[ty][tx];
			
			// Synchronize to make sure that all value are current in block
			__syncthreads();
		}
	}
	// Compute j-aligned singly dependent blocks
	else
	{
        	#pragma unroll
		FOR(j, 0, BLOCK_SIZE - 1)
		{
			newPath = current[ty][j] + primary[j][tx];
			current[ty][tx] = current[ty][tx] > newPath ? newPath : current[ty][tx];
		        
			// Synchronize to make sure that all value are current in block
			__syncthreads();
		}
	}

	if (v1 < n && v2 < n)
        {
        	d[v1 * n + v2] = current[ty][tx];
        }

}

/**Kernel for parallel Floyd Warshall algorithm on gpu compute of dependent block depend on the singly dependent blocks
*
* @param b number block of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
*/

template <int BLOCK_SIZE> __global__ void fw_kernel_phase_three(const unsigned int block, const unsigned int n, int *d)
{
	if (blockIdx.x == block || blockIdx.y == block) return ;
	int newPath;
	int path;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int v1 = blockDim.y * blockIdx.y + ty;
	int v2 = blockDim.x * blockIdx.x + tx;

	int v1Row = BLOCK_SIZE * block + ty;
	int v2Row = v2;
	
	int v1Col = v1;
	int v2Col = BLOCK_SIZE * block + tx;
	
	__shared__ int primaryRow[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int primaryCol[BLOCK_SIZE][BLOCK_SIZE];

	path = (v1 < n && v2 < n)? d[v1 * n + v2] : INF;

	primaryCol[ty][tx] = (v1Col < n && v2Col < n) ? d[v1Col * n + v2Col] : INF;
	primaryRow[ty][tx] = (v1Row < n && v2Row < n) ? d[v1Row * n + v2Row] : INF;

	 // Synchronize to make sure the all value are loaded in block
	__syncthreads();

	#pragma unroll
	FOR (i, 0, BLOCK_SIZE - 1)
	{
		newPath = primaryCol[ty][i] + primaryRow[i][tx];
		path = path > newPath ? newPath : path;
	}

	if (v1 < n && v2 < n)
	{
		d[v1 * n + v2] = path;
	}
}

/** Parallel Floyd Warshall algorithm using gpu
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
* @param d matrix of shortest paths d(G)
*/
cudaError_t fw_gpu(const unsigned int n, int *G, int *d)
{
	int *dev_d = 0;
	int numOfBlock = (n - 1) / BLOCK_WIDTH;
	cudaError_t cudaStatus;
	cudaStream_t cpyStream;

	// Choose which GPU to run on, change this on a multi-GPU system.
    	cudaStatus = cudaSetDevice(0);
	HANDLE_ERROR(cudaStatus);

	// Initialize the grid and block dimensions here
	dim3 dimGridP2((n - 1) / BLOCK_WIDTH + 1, 2 , 1);
       	dim3 dimGridP3((n - 1) / BLOCK_WIDTH + 1, (n - 1) / BLOCK_WIDTH + 1, 1);

	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	#ifdef DEBUG
		printf("|V| %d\n", n);
		printf("Dim Grid:\nx - %d\ny -%d\nz - %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
		printf("Dim Block::\nx - %d\ny -%d\nz - %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	#endif
	
	// Wake up gpu 
 	wake_gpu_kernel<<<1, dimBlock>>>(32);
  
	// Create new stream to copy data	
	cudaStatus = cudaStreamCreate(&cpyStream);
	HANDLE_ERROR(cudaStatus);

	// Allocate GPU buffers for matrix of shortest paths d(G)
	cudaStatus =  cudaMalloc((void**)&dev_d, n * n * sizeof(int));
	HANDLE_ERROR(cudaStatus);

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpyAsync(dev_d, G, n * n * sizeof(int), CMCPYHTD, cpyStream);
        HANDLE_ERROR(cudaStatus);

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        cudaStatus = cudaDeviceSynchronize();
        HANDLE_ERROR(cudaStatus);

	FOR(block, 0, numOfBlock) 
	{
		fw_kernel_phase_one<BLOCK_WIDTH><<<1, dimBlock>>>(block, n, dev_d);
		fw_kernel_phase_two<BLOCK_WIDTH><<<dimGridP2, dimBlock>>>(block, n, dev_d);
		fw_kernel_phase_three<BLOCK_WIDTH><<<dimGridP3, dimBlock>>>(block, n, dev_d);
	}

	// Check for any errors launching the kernel
    	cudaStatus = cudaGetLastError();
	HANDLE_ERROR(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    	// any errors encountered during the launch.
    	cudaStatus = cudaDeviceSynchronize();
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaMemcpy(d, dev_d, n * n * sizeof(int), CMCPYDTH);
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaFree(dev_d);
	return cudaStatus;
}

/**
* Print graf G as a matrix
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
*/
void print_graf(const unsigned int n, const int *G)
{
	FOR(v1, 0, n - 1)
	{
		FOR(v2, 0, n - 1) 
		{	
			if (G[v1 * n + v2] < INF)
				printf("%d ", G[v1 * n + v2]);
			else
				printf("INF ");
		}
		printf("\n");
	}
}

int main(int argc, char **argv)
{
	unsigned int V;
	unsigned int E;
	unsigned int v1, v2, w; 
	
	// Load number vertices of the graph |V(G)| and number edges of the graph |E(G)|
	scanf("%d %d", &V, &E);
		
	// Alloc host data for G - graf, d - matrix of shortest paths
	unsigned int size = V * V;
	
	int *G = (int *) malloc (sizeof(int) * size);
	int *d = (int *) malloc (sizeof(int) * size);
	
	// Init Data for the graf G
	memset(G, CHARINF, sizeof(int) * V * V);
	
	#ifdef DEBUG
		print_graf(V, G);
	#endif

	// Load weight of the edges of the graph E(G)
	REP(e, E)
	{
		scanf("%d %d %d", &v1, &v2, &w);
		G[v1 * V + v2] = w;
	}

	FOR (v, 0, V - 1)
		G[v * V + v] = 0;

	#ifdef DEBUG
		print_graf(V, G);
	#endif

	fw_gpu(V, G, d);
	
	if (PRINT) print_graf(V, d);

	// Delete allocated memory 
	free(G);
	free(d);

	return 0;
}
