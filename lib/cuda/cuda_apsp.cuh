/* Simple CUDA library for APSP problem
 *
 * Author: Matuesz Bojanowski
 *  Email: bojanowski.mateusz@gmail.com
 */

#ifndef _CUDA_APSP_
#define _CUDA_APSP_

#include "../apsp.h"

// CONSTS for Naive FW
#define BLOCK_SIZE 16

// CONSTS for Blocked FW
#define MAX_BLOCK_SIZE 32
#define MAX_VIRTUAL_BLOCK_SIZE 64
#define VIRTUAL_THREAD_SIZE 4

/**
 * Naive implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
void cudaNaiveFW(const std::unique_ptr<graphAPSPTopology>& dataHost);

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
void cudaBlockedFW(const std::unique_ptr<graphAPSPTopology>& dataHost);


#endif /* _APSP_ */
