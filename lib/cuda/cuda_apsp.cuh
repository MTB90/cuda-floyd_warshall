/* Simple CUDA library for APSP problem
 *
 * Author: Matuesz Bojanowski
 *  Email: bojanowski.mateusz@gmail.com
 */

#ifndef _CUDA_APSP_
#define _CUDA_APSP_

#include "../apsp.h"

// CONSTS for CUDA FW
#define BLOCK_SIZE 16

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
