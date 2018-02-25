/* Simple library for APSP problem
 *
 * Author: Matuesz Bojanowski
 *  Email: bojanowski.mateusz@gmail.com
 */

#ifndef _APSP_
#define _APSP_

#include <memory>

/* Maximum distance value for path form
 * vertex x to vertex y means no path v1 -> v2.
 * This value should be MAX_INT / 2 - 1
 * because we should be able to compare path v1 -> v2 with
 * path v1 -> u -> v2 so adding to value of paths v1 -> u and
 * u -> v2 should be smaller than maximum int value
 */
#define MAX_DISTANCE 1 << 30 - 1

/* ASPS algorithms types */
typedef enum {
    NAIVE_FW = 0,
    CUDA_NAIVE_FW = 1,
    CUDA_BLOCKED_FW = 2

} graphAPSPAlgorithm;

/* Default structure for graph */
struct graphAPSPTopology {
    unsigned int nvertex; // number of vertex in graph
    std::unique_ptr<int[]> pred; // predecessors matrix
    std::unique_ptr<int[]> graph; // graph matrix

    /* Constructor for init fields */
    graphAPSPTopology(int nvertex): nvertex(nvertex) {
        int size = nvertex * nvertex;
        pred = std::unique_ptr<int[]>(new int[size]);
        graph = std::unique_ptr<int[]>(new int[size]);
    }
};

/* APSP API to compute all pairs shortest paths in graph,
 * init graph matrix should be point by graph in data, results will be
 * store in prep (predecessors) and in graph (value for shortest paths)
 *
 * @param data: unique ptr to graph data with allocated fields
 * @param algorithm: algorithm type for APSP
 */
void apsp(const std::unique_ptr<graphAPSPTopology>& data, graphAPSPAlgorithm algorithm);

#endif /* _APSP_ */
