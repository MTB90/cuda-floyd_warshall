#include<stdio.h>
#include<stdlib.h>
#include<string.h>

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

const bool PRINT = true; 	// print graf d or not

/** Floyd Warshall algorithm
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
* @param d matrix of shortest paths d(G)
*/
void fw(const unsigned int n, int *G, int *d)
{
	memcpy (d, G, n * n * sizeof(int));
	FOR(u, 0, n - 1)
		FOR(v1, 0, n - 1)
			FOR(v2, 0, n - 1)
			{
				int newPath = d[v1 * n + u] + d[u * n + v2];
				int oldPath = d[v1 * n + v2];
				d[v1 * n + v2] = ( oldPath > newPath) ? newPath : oldPath;
			}
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
	
	// Run Floyd Warshall
  	fw(V, G, d);

	if (PRINT) print_graf(V, d);

	// Delete allocated memory 
	free(G);
	free(d);

	return 0;
}
