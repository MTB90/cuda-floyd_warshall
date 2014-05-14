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

bool gPrint = false; 	// print graf d or not
bool gDebug = false;	// print more deatails to debug

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

/**Kernel for wake gpu
*
* @param reps dummy variable only to perform some action
*/
__global__ void wake_gpu_kernel(int reps) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= reps) return;
}

/**Kernel for parallel Floyd Warshall algorithm on gpu
* 
* @param u number vertex of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
*/
__global__ void fw_kernel(const unsigned int u, const unsigned int n, int *d)
{
	int v1 = blockDim.y * blockIdx.y + threadIdx.y;
	int v2 = blockDim.x * blockIdx.x + threadIdx.x;

	if (v1 < n && v2 < n) 
	{
		d[v1 * n + v2] = (d[v1 * n + v2] >  d[v1 * n + u] + d[u * n + v2]) ?  
				 d[v1 * n + u] + d[u * n + v2] : d[v1 * n + v2];
	}
}

/** Parallel Floyd Warshall algorithm using gpu
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
* @param d matrix of shortest paths d(G)
* @param opt define what version should be run
*/
cudaError_t fw_gpu(const unsigned int n, const int *G, int *d)
{
	int *dev_d = 0;
	cudaError_t cudaStatus;
	cudaStream_t cpyStream;

	// Choose which GPU to run on, change this on a multi-GPU system.
    	cudaStatus = cudaSetDevice(0);
	HANDLE_ERROR(cudaStatus);

	// Initialize the grid and block dimensions here
	dim3 dimGrid((n - 1) / BLOCK_WIDTH + 1, (n - 1) / BLOCK_WIDTH + 1, 1); 
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	if (gDebug) 
	{
		printf("|V| %d\n", n);
		printf("Dim Grid:\nx - %d\ny - %d\nz - %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
		printf("Dim Block::\nx - %d\ny - %d\nz - %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	}

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

	cudaFuncSetCacheConfig(fw_kernel, cudaFuncCachePreferL1 );
	FOR(u, 0, n - 1) 
	{
		fw_kernel<<<dimGrid, dimBlock>>>(u, n, dev_d);
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
		
	// Alloc host data for G - graf, d - matrix of shortest paths
	unsigned int size = V * V;
	
	int *G = (int *) malloc (sizeof(int) * size);
	int *d = (int *) malloc (sizeof(int) * size);
	
	// Init Data for the graf G
	memset(G, CHARINF, sizeof(int) * V * V);
	
	if (gDebug)
	{
		fprintf(stdout, "\nInit data:\n");
	       	print_graf(V, G);
	}

	// Load weight of the edges of the graph E(G)
	REP(e, E)
	{
		scanf("%d %d %d", &v1, &v2, &w);
		G[v1 * V + v2] = w;
	}

	FOR (v, 0, V - 1)
		G[v * V + v] = 0;

	if (gPrint)
	{	
		fprintf(stdout, "\nLoaded data:\n");
		print_graf(V, G);
	}

	// Initialize CUDA Event
	cudaEvent_t start,stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	fw_gpu(V, G, d);
	
	// Finish recording
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	// Calculate elasped time
	cudaEventElapsedTime(&elapsedTime,start,stop);

	if (gPrint) 
	{
		fprintf(stdout, "\nResult:\n");
		print_graf(V, d);
	}
	
	printf ("Time : %f ms\n", elapsedTime);
	
	// Delete allocated memory 
	free(G);
	free(d);

	return 0;
}
