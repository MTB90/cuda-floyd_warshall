#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

// CUDA Headers
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Helper definition
#define VAR(v, i) __typeof(i) v=(i)
#define FOR(i, j, k) for (i = (j); i <= (k); ++i)
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
#define THREAD_WIDTH 2
#define BLOCK_WIDTH 16
#define WARP 	    32

bool gPrint = false; 	// print graph d or not
bool gDebug = false;	// print more deatails to debug

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
template <int BLOCK_SIZE> __global__ void fw_kernel_phase_one(const unsigned int block, const unsigned int n, const size_t pitch, int * const d, int * const p)
{
	int i;
	int newPath;
	int newPred;

	const int tx = threadIdx.x; 
	const int ty = threadIdx.y;
	
	const int v1 = BLOCK_SIZE * block + ty;
	const int v2 = BLOCK_SIZE * block + tx;  

	const int cell = v1 * pitch + v2;

	__shared__ int primary_d[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int primary_p[BLOCK_SIZE][BLOCK_SIZE];

	if (v1 < n && v2 < n) 
	{
		 primary_d[ty][tx] = d[cell];
		 primary_p[ty][tx] = p[cell];
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
		d[cell] = primary_d[ty][tx];
		p[cell] = primary_p[ty][tx];
	}
}

/**Kernel for parallel Floyd Warshall algorithm on gpu compute of singly dependent block depend on the independent block
*
* @param b number block of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/
template <int BLOCK_SIZE> __global__ void fw_kernel_phase_two(const unsigned int block, const unsigned int n, const size_t pitch, int * const d, int * const p)
{
	if (blockIdx.x == block) return;
	int i;
        int newPath;
	int newPred;

	int tx = threadIdx.x;
        int ty = threadIdx.y;

	int v1 = BLOCK_SIZE * block + ty;
        int v2 = BLOCK_SIZE * block + tx;
	
	__shared__ int primary_d[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int current_d[BLOCK_SIZE][BLOCK_SIZE];
	
	__shared__ int primary_p[BLOCK_SIZE][BLOCK_SIZE]; 
	__shared__ int current_p[BLOCK_SIZE][BLOCK_SIZE];

	const int cell_primary = v1 * pitch + v2;
	if (v1 < n && v2 < n)
	{
		primary_d[ty][tx] = d[cell_primary];
		primary_p[ty][tx] = p[cell_primary];
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
	
	const int cell_current = v1 * pitch + v2;
	if (v1 < n && v2 < n)
	{
		current_d[ty][tx] = d[cell_current];
		current_p[ty][tx] = p[cell_current];
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
		FOR(i, 0, BLOCK_SIZE - 1)
		{
			newPath = current_d[ty][i] + primary_d[i][tx];
			
			__syncthreads();
			if (newPath < current_d[ty][tx])
			{
				current_d[ty][tx] = newPath;
				current_p[ty][tx] = primary_p[i][tx];
			}
		        
			// Synchronize to make sure that all value are current in block
			__syncthreads();
		}
	}

	if (v1 < n && v2 < n)
        {
        	d[cell_current] = current_d[ty][tx];
		p[cell_current] = current_p[ty][tx];
        }
}

/**Kernel for parallel Floyd Warshall algorithm on gpu compute of dependent block depend on the singly dependent blocks
*
* @param b number block of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/

template <int BLOCK_SIZE, int THREAD_SIZE> __global__ void fw_kernel_phase_three(unsigned int block, const unsigned int n, const size_t pitch, int * const d, int * const p)
{
	if (blockIdx.x == block || blockIdx.y == block) return;
	int i, j, k;	
	int newPath;
	int predecessor;
	int path;

	const int tx = threadIdx.x * THREAD_SIZE;
	const int ty = threadIdx.y * THREAD_SIZE;

	const int v1 = blockDim.y * blockIdx.y * THREAD_SIZE + ty;
	const int v2 = blockDim.x * blockIdx.x * THREAD_SIZE + tx;

	int idx, idy;
	
	__shared__ int primaryRow_d[BLOCK_SIZE * THREAD_SIZE][BLOCK_SIZE * THREAD_SIZE];
	__shared__ int primaryCol_d[BLOCK_SIZE * THREAD_SIZE][BLOCK_SIZE * THREAD_SIZE];
	__shared__ int primaryRow_p[BLOCK_SIZE * THREAD_SIZE][BLOCK_SIZE * THREAD_SIZE];
	
	int v1Row = BLOCK_SIZE * block * THREAD_SIZE + ty;
	int v2Col = BLOCK_SIZE * block * THREAD_SIZE + tx;

	// Load data for virtual block
	#pragma unroll
	FOR (i, 0, THREAD_SIZE - 1)
	{
		#pragma unroll
		FOR(j, 0, THREAD_SIZE -1)
		{
			idx = tx + j;
			idy = ty + i;
		
			if (v1Row + i < n && v2 + j < n)
			{
				block = (v1Row + i) * pitch + v2 + j;
 		
				primaryRow_d[idy][idx] = d[block];
				primaryRow_p[idy][idx] = p[block];
			}
			else
			{
				primaryRow_d[idy][idx] = INF;
				primaryRow_p[idy][idx] = NONE;
			}
		
			if (v1 + i  < n && v2Col + j < n)
			{
				block = (v1 + i) * pitch + v2Col + j;
				primaryCol_d[idy][idx] = d[block];
			}
			else
			{
				primaryCol_d[idy][idx] = INF;
			}
		}
	}
	
	 // Synchronize to make sure the all value are loaded in virtual block
	__syncthreads();

        // Compute data for virtual block
        #pragma unroll
        FOR (i, 0, THREAD_SIZE - 1)
        {
                #pragma unroll
                FOR(j, 0, THREAD_SIZE -1)
		{
			if (v1 + i < n && v2 + j < n )
			{
				block = (v1 + i) * pitch + v2 + j;
		                
				path = d[block];
		                predecessor = p[block];
				
				idy = ty + i;
				idx = tx + j;

				#pragma unroll
				FOR (k, 0, BLOCK_SIZE * THREAD_SIZE - 1)
				{
					newPath = primaryCol_d[idy][k] + primaryRow_d[k][idx];
					if (path > newPath)
					{
						path = newPath;
						predecessor = primaryRow_p[k][idx];
					}
				}
                		d[block] = path;
		                p[block] = predecessor;
			}
		}
	}
}

/** Parallel Floyd Warshall algorithm using gpu
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/
template <int BLOCK_SIZE, int THREAD_SIZE> void fw_gpu(const unsigned int n, const int * const G, int * const d, int * const p)
{
	int *dev_d = 0;
	int *dev_p = 0;

	size_t pitch;
	size_t pitch_int;

	// Size of virtual block
	const int VIRTUAL_BLOCK_SIZE = BLOCK_SIZE * THREAD_SIZE;

	cudaError_t cudaStatus;
	cudaStream_t cpyStream;

        // Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	HANDLE_ERROR(cudaStatus);

        // Initialize the grid and block dimensions here
	dim3 dimGridP1(1, 1, 1);
	dim3 dimGridP2((n - 1) / VIRTUAL_BLOCK_SIZE + 1, 2 , 1);
	dim3 dimGridP3((n - 1) / VIRTUAL_BLOCK_SIZE + 1, (n - 1) / VIRTUAL_BLOCK_SIZE + 1, 1);

	dim3 dimBlockP1(VIRTUAL_BLOCK_SIZE, VIRTUAL_BLOCK_SIZE, 1);
	dim3 dimBlockP2(VIRTUAL_BLOCK_SIZE, VIRTUAL_BLOCK_SIZE, 1);
	dim3 dimBlockP3(BLOCK_SIZE, BLOCK_SIZE, 1);
	
	int numOfBlock = (n - 1) / VIRTUAL_BLOCK_SIZE;

	if (gDebug) 
	{
		printf("|V| %d\n", n);

		printf("Phase 1\n");
		printf("Dim Grid:\nx - %d\ny - %d\nz - %d\n", dimGridP1.x, dimGridP1.y, dimGridP1.z);
		printf("Dim Block::\nx - %d\ny - %d\nz - %d\n", dimBlockP1.x, dimBlockP1.y, dimBlockP1.z);

		printf("\nPhase 2\n");
                printf("Dim Grid:\nx - %d\ny - %d\nz - %d\n", dimGridP2.x, dimGridP2.y, dimGridP2.z);
                printf("Dim Block::\nx - %d\ny - %d\nz - %d\n", dimBlockP2.x, dimBlockP2.y, dimBlockP2.z);

                printf("Phase 3\n");
                printf("Dim Grid:\nx - %d\ny - %d\nz - %d\n", dimGridP3.x, dimGridP3.y, dimGridP3.z);
                printf("Dim Block::\nx - %d\ny - %d\nz - %d\n", dimBlockP3.x, dimBlockP3.y, dimBlockP3.z);
	}
	
	// Create new stream to copy data	
	cudaStatus = cudaStreamCreate(&cpyStream);
	HANDLE_ERROR(cudaStatus);

	// Allocate GPU buffers for matrix of shortest paths d(G)
	cudaStatus = cudaMallocPitch(&dev_d, &pitch, n * sizeof(int), n);
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaMallocPitch(&dev_p, &pitch, n * sizeof(int), n);
	HANDLE_ERROR(cudaStatus);

	pitch_int = pitch / sizeof(int);

        // Wake up gpu
	wake_gpu_kernel<<<1, dimBlockP1>>>(32);
	
        // Copy input vectors from host memory to GPU buffers.
        cudaMemcpy2DAsync(dev_d, pitch, G, n * sizeof(int), n * sizeof(int), n, CMCPYHTD, cpyStream);
	cudaMemcpy2DAsync(dev_p, pitch, p, n * sizeof(int), n * sizeof(int), n, CMCPYHTD, cpyStream);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
        cudaStatus = cudaDeviceSynchronize();
        HANDLE_ERROR(cudaStatus);

	int block;
	FOR(block, 0, numOfBlock) 
	{
		fw_kernel_phase_one<VIRTUAL_BLOCK_SIZE><<<1, dimBlockP1>>>(block, n, pitch_int, dev_d, dev_p);
		fw_kernel_phase_two<VIRTUAL_BLOCK_SIZE><<<dimGridP2, dimBlockP2>>>(block, n, pitch_int, dev_d, dev_p);
		fw_kernel_phase_three<BLOCK_SIZE, THREAD_SIZE><<<dimGridP3, dimBlockP3>>>(block, n, pitch_int, dev_d, dev_p);
	}

	// Check for any errors launching the kernel
    	cudaStatus = cudaGetLastError();
	HANDLE_ERROR(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    	// any errors encountered during the launch.
    	cudaStatus = cudaDeviceSynchronize();
	HANDLE_ERROR(cudaStatus);

	cudaStatus = cudaMemcpy2D(d, n * sizeof(int), dev_d, pitch, n * sizeof(int), n, CMCPYDTH);
	HANDLE_ERROR(cudaStatus);
	
	cudaStatus = cudaMemcpy2D(p, n * sizeof(int), dev_p, pitch, n * sizeof(int), n, CMCPYDTH);
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
	int v1, v2;
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

/**
* Reconstruct Path
*
* @param i, j id vertex 
* @param G is a the graph G:=(V,E)
* @param p matrix of predecessors p(G)
*/
int reconstruct_path(unsigned int n, unsigned int i, unsigned int j, const int * const p, const int * const G)
{
	if (i == j )
		return 0;
	else if ( p[i * n + j] == NONE)
		return INF;
	else
	{
		int path = reconstruct_path(n, i, p[i * n + j], p, G);
		if (path == INF) 
			return INF;
		else
			return path + G[ p [i * n + j] * n + j];
	}
}

/**
* Check paths
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors p(G)
*/
bool check_paths(const unsigned int n, const int * const G, const int * const d, const int * const p)
{
	int i, j;
	FOR (i, 0, n - 1)
	{
		FOR (j, 0, n - 1)
		{
			int path = reconstruct_path(n, i, j, p, G);
			if (gDebug)
				printf("%d %d %d == %d \n", i, j, path, d[i * n + j]);
			if (path != d[i * n + j])
				return false;
		}
	}

	return true;
}

int main(int argc, char **argv)
{
	unsigned int V;
	unsigned int E;
	unsigned int v1, v2, w; 
	int check = false;
	int opt;

	while ((opt = getopt (argc, argv, "pdc")) != -1)
	{
		switch(opt)
		{
			case 'p':
				gPrint = true;
				break;
			case 'd':
				gDebug = true;
				break;
			case 'c':
				check = true;
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

	int v;
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
	
	fw_gpu<BLOCK_WIDTH, THREAD_WIDTH>(V, G, d, p);
	
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
	
	if (check)
	{
		if (check_paths(V, G, d, p))
		{
			 fprintf(stdout, "\nResult are correct:\n");
		}
		else
		{
			 fprintf(stdout, "\nResult are incorrect:\n");
		}
	}

	// Delete allocated memory 
	free(G);
	free(d);
	free(p);

	return 0;
}
