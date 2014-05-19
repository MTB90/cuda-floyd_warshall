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
#define NONE	-1
#define CMCPYHTD cudaMemcpyHostToDevice
#define CMCPYDTH cudaMemcpyDeviceToHost

// CONSTS for compute capability 2.0
#define BLOCK_WIDTH 16
#define WARP 	    32

bool gPrint = false; 	// print graph d or not
bool gDebug = false;	// print more deatails to debug

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
	if (idx >= reps) return;
}

/**Kernel for parallel Floyd Warshall algorithm on gpu compute of independent block
* 
* @param b number block of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/
template <int BLOCK_SIZE> __global__ void fw_kernel_phase_one(const unsigned int block, const unsigned int n, int * const d, int * const p)
{
	int newPath;
	int newPred;

	int tx = threadIdx.x; 
	int ty = threadIdx.y;
	
	int v1 = BLOCK_SIZE * block + ty;
	int v2 = BLOCK_SIZE * block + tx;  

	__shared__ int primary_d[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int primary_p[BLOCK_SIZE][BLOCK_SIZE];

	if (v1 < n && v2 < n) 
	{
		 primary_d[ty][tx] = d[v1 * n + v2];
		 primary_p[ty][tx] = p[v1 * n + v2];
		 newPred = primary_p[ty][tx];
	}
	else 
	{
                 primary_d[ty][tx] = INF;
		 primary_p[ty][tx] = NONE;
	}

	// Synchronize to make sure the all value are loaded in block
	__syncthreads();

		
	#pragma unroll
	FOR(i, 0, BLOCK_SIZE - 1)
	{
		newPath = primary_d[ty][i] + primary_d[i][tx];
		
		__syncthreads();
		if (newPath < primary_d[ty][tx] )
		{
			primary_d[ty][tx] = newPath;
			newPred = primary_p[i][tx];
		}
		
		// Synchronize to make sure that all value are current
		__syncthreads();
		primary_p[ty][tx] = newPred;
	}

	if (v1 < n && v2 < n) 
	{
		d[v1 * n + v2] = primary_d[ty][tx];
		p[v1 * n + v2] = primary_p[ty][tx];
	}
}

/**Kernel for parallel Floyd Warshall algorithm on gpu compute of singly dependent block depend on the independent block
*
* @param b number block of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/
template <int BLOCK_SIZE> __global__ void fw_kernel_phase_two(const unsigned int block, const unsigned int n, int * const d, int * const p)
{
	if (blockIdx.x == block) return;

        int newPath;
	int newPred;

	int v1, v2;

	int tx = threadIdx.x;
        int ty = threadIdx.y;

	int pv1 = BLOCK_SIZE * block + ty;
        int pv2 = BLOCK_SIZE * block + tx;

	__shared__ int primary_d[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int current_d[BLOCK_SIZE][BLOCK_SIZE];
	
	__shared__ int primary_p[BLOCK_SIZE][BLOCK_SIZE]; 
	__shared__ int current_p[BLOCK_SIZE][BLOCK_SIZE];

	if (pv1 < n && pv2 < n)
	{
		primary_d[ty][tx] = d[pv1 * n + pv2];
		primary_p[ty][tx] = p[pv1 * n + pv2];
	}
	else
	{
                primary_d[ty][tx] = INF;
		primary_p[ty][tx] = NONE;
	}
	
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

	if (v1 < n && v2 < n)
	{
		current_d[ty][tx] = d[v1 * n + v2];
		current_p[ty][tx] = p[v1 * n + v2];
		newPred = current_p[ty][tx];
	}
	else
	{
                current_d[ty][tx] = INF;
		current_p[ty][tx] = NONE;
	}	

	// Synchronize to make sure the all value are loaded in block
	__syncthreads();

	// Compute i-aligned singly dependent blocks
	if (blockIdx.y == 0)
	{
		#pragma unroll
		FOR(i, 0, BLOCK_SIZE - 1)
		{
			newPath = primary_d[ty][i] + current_d[i][tx];
			
			__syncthreads();
			if (newPath < current_d[ty][tx])
			{
				current_d[ty][tx] = newPath;
				newPred = current_p[i][tx];
			}
			
			// Synchronize to make sure that all value are current in block
			__syncthreads();
			current_p[ty][tx] = newPred;
		}
	}
	// Compute j-aligned singly dependent blocks
	else
	{
        	#pragma unroll
		FOR(j, 0, BLOCK_SIZE - 1)
		{
			newPath = current_d[ty][j] + primary_d[j][tx];
			
			__syncthreads();
			if (newPath < current_d[ty][tx])
			{
				current_d[ty][tx] = newPath;
				current_p[ty][tx] = primary_p[j][tx];
			}
		        
			// Synchronize to make sure that all value are current in block
			__syncthreads();
		}
	}

	if (v1 < n && v2 < n)
        {
        	d[v1 * n + v2] = current_d[ty][tx];
		p[v1 * n + v2] = current_p[ty][tx];
        }
}

/**Kernel for parallel Floyd Warshall algorithm on gpu compute of dependent block depend on the singly dependent blocks
*
* @param b number block of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/

template <int BLOCK_SIZE> __global__ void fw_kernel_phase_three(const unsigned int block, const unsigned int n, int * const d, int * const p)
{
	if (blockIdx.x == block || blockIdx.y == block) return ;
	int newPath;
	int path;
	int predecessor;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int v1 = blockDim.y * blockIdx.y + ty;
	int v2 = blockDim.x * blockIdx.x + tx;

	int v1Row = BLOCK_SIZE * block + ty;
	int v2Row = v2;
	
	int v1Col = v1;
	int v2Col = BLOCK_SIZE * block + tx;
	
	__shared__ int primaryRow_d[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int primaryCol_d[BLOCK_SIZE][BLOCK_SIZE];
        
	__shared__ int primaryRow_p[BLOCK_SIZE][BLOCK_SIZE];
	
	if (v1 < n && v2 < n)
	{
		path = d[v1 * n + v2];
		predecessor = p[v1 * n + v2];
	}
	else
	{
		path = INF;
		predecessor = NONE;
	}


	if (v1Row < n && v2Row < n)
	{
 		primaryRow_d[ty][tx] = d[v1Row * n + v2Row];
		primaryRow_p[ty][tx] = p[v1Row * n + v2Row];
	}
	else
	{
		primaryRow_d[ty][tx] = INF;
		primaryRow_p[ty][tx] = NONE;
	}

	primaryCol_d[ty][tx] = (v1Col < n && v2Col < n) ? d[v1Col * n + v2Col] : INF;
	

	 // Synchronize to make sure the all value are loaded in block
	__syncthreads();

	#pragma unroll
	FOR (i, 0, BLOCK_SIZE - 1)
	{
		newPath = primaryCol_d[ty][i] + primaryRow_d[i][tx];
		if (path > newPath)
		{
			path = newPath;
			predecessor = primaryRow_p[i][tx];
		}
	}

	if (v1 < n && v2 < n)
	{
		d[v1 * n + v2] = path;
		p[v1 * n + v2] = predecessor;
	}
}

/** Parallel Floyd Warshall algorithm using gpu
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/
void fw_gpu(const unsigned int n, const int * const G, int * const d, int * const p)
{
	int *dev_d = 0;
	int *dev_p = 0;
	
	cudaError_t cudaStatus;
	cudaStream_t cpyStream;

        // Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	HANDLE_ERROR(cudaStatus);

        // Initialize the grid and block dimensions here
	dim3 dimGridP2((n - 1) / BLOCK_WIDTH + 1, 2 , 1);
	dim3 dimGridP3((n - 1) / BLOCK_WIDTH + 1, (n - 1) / BLOCK_WIDTH + 1, 1);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	int numOfBlock = (n - 1) / BLOCK_WIDTH;

	if (gDebug) 
	{
		printf("|V| %d\n", n);
		printf("Dim Grid:\nx - %d\ny - %d\nz - %d\n", dimGridP3.x, dimGridP3.y, dimGridP3.z);
		printf("Dim Block::\nx - %d\ny - %d\nz - %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	}
	
	// Create new stream to copy data	
	cudaStatus = cudaStreamCreate(&cpyStream);
	HANDLE_ERROR(cudaStatus);

	// Allocate GPU buffers for matrix of shortest paths d(G)
	cudaStatus =  cudaMalloc((void**)&dev_d, n * n * sizeof(int));
	HANDLE_ERROR(cudaStatus);
	cudaStatus =  cudaMalloc((void**)&dev_p, n * n * sizeof(int));
	HANDLE_ERROR(cudaStatus);

        // Wake up gpu
	wake_gpu_kernel<<<1, dimBlock>>>(32);
	
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpyAsync(dev_d, G, n * n * sizeof(int), CMCPYHTD, cpyStream);
	cudaStatus = cudaMemcpyAsync(dev_p, p, n * n * sizeof(int), CMCPYHTD, cpyStream);
        
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
        cudaStatus = cudaDeviceSynchronize();
        HANDLE_ERROR(cudaStatus);

	FOR(block, 0, numOfBlock) 
	{
		fw_kernel_phase_one<BLOCK_WIDTH><<<1, dimBlock>>>(block, n, dev_d, dev_p);
		fw_kernel_phase_two<BLOCK_WIDTH><<<dimGridP2, dimBlock>>>(block, n, dev_d, dev_p);
		fw_kernel_phase_three<BLOCK_WIDTH><<<dimGridP3, dimBlock>>>(block, n, dev_d, dev_p);
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
	
	cudaStatus = cudaMemcpy(p, dev_p, n * n * sizeof(int), CMCPYDTH);
	HANDLE_ERROR(cudaStatus);
	
	cudaStatus = cudaFree(dev_d);
	HANDLE_ERROR(cudaStatus);
	
	cudaStatus = cudaFree(dev_p);
	HANDLE_ERROR(cudaStatus);

	return;
}

/**
* Print graph G as a matrix
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
*/
void print_graph(const unsigned int n, const int * const G)
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
	printf("\n");
}

int main(int argc, char **argv)
{
	unsigned int V;
	unsigned int E;
	unsigned int v1, v2, w; 
	int opt;

	while ((opt = getopt (argc, argv, "pd")) != -1)
	{
		switch(opt)
		{
			case 'p':
				gPrint = true;
				break;
			case 'd':
				gDebug = true;
				break;
			case '?':
				fprintf (stderr, "Unknown option character `\\x%x'.\n", opt);
				return 1;
			default:
			        abort ();
		}
	}

	// Load number vertices of the graph |V(G)| and number edges of the graph |E(G)|
	scanf("%d %d", &V, &E);
		
	// Alloc host data for G - graph, d - matrix of shortest paths
	unsigned int size = V * V;
	
	int *G = (int *) malloc (sizeof(int) * size);
	int *d = (int *) malloc (sizeof(int) * size);
	int *p = (int *) malloc (sizeof(int) * size);

	// Init Data for the graph G
	memset(G, CHARINF, sizeof(int) * size);
	memset(p, NONE, sizeof(int) * size);

	if (gDebug)
	{
		fprintf(stdout, "\nInit data:\n");
	       	print_graph(V, G);
	}

	// Load weight of the edges of the graph E(G)
	REP(e, E)
	{
		scanf("%d %d %d", &v1, &v2, &w);
		G[v1 * V + v2] = w;
		if (v1 != v2)
			p[v1 * V + v2] = v1;
	}

	FOR (v, 0, V - 1)
		G[v * V + v] = 0;

	if (gDebug)
	{	
		fprintf(stdout, "\nLoaded data:\n");
		print_graph(V, G);
	}

	// Initialize CUDA Event
	cudaEvent_t start,stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	fw_gpu(V, G, d, p);
	
	// Finish recording
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	// Calculate elasped time
	cudaEventElapsedTime(&elapsedTime,start,stop);

	if (gPrint) 
	{
		fprintf(stdout, "\nResult short path:\n");
		print_graph(V, d);
	}
	
	if (gPrint) 
	{
		fprintf(stdout, "\nResult predecessors:\n");
		print_graph(V, p);
	}

	elapsedTime /= 1000;
	printf ("Time : %f s\n", elapsedTime);
	
	// Delete allocated memory 
	free(G);
	free(d);
	free(p);

	return 0;
}
