#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<time.h>

// Helper definition
#define VAR(v, i) __typeof(i) v=(i)
#define FOR(i, j, k) for (int i = (j); i <= (k); ++i)
#define FORD(i, j, k)for (int i=(j); i >= (k); --i)
#define FORE(i, c) for(VAR(i, (c).begin()); i != (c).end(); ++i)
#define REP(i, n) for(int i = 0;i <(n); ++i)

// CONSTS
#define INF 	1061109567 // 3F 3F 3F 3F
#define NONE 	-1
#define CHARINF 63	   // 3F	
#define CHARBIT 8

bool gPrint = false; 	// print graph d and p or not
bool gDebug = false;	// print more deatails to gDebug

/** Floyd Warshall algorithm
*
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param G is a the graph G:=(V,E)
* @param d matrix of shortest paths d(G)
* @param p matrix of predecessors  p(G)
*/
void fw(const unsigned int n, const int * const G, int * const d, int * const p)
{
	int newPath = 0;
	int oldPath = 0;

	FOR(u, 0, n - 1)
		FOR(v1, 0, n - 1)
			FOR(v2, 0, n - 1)
			{
				newPath = d[v1 * n + u] + d[u * n + v2];
				oldPath = d[v1 * n + v2];
				if (oldPath > newPath) 
				{
					d[v1 * n + v2] = newPath;
					p[v1 * n + v2] = p[u * n + v2];
				}
			}
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
				fprintf (stderr, "Unknown option character \n");
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

	// Init Data for the graph G and p
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
		p[v1 * V + v2] = v1 != v2 ? v1: NONE;
	}

	FOR (v, 0, V - 1)
		G[v * V + v] = 0;

	// Copy data
	memcpy (d, G, size * sizeof(int));
	
	if (gDebug)
	{	
		fprintf(stdout, "\nLoaded data:\n");
		print_graph(V, G);
	}

  	clock_t begin = clock();
        
	// Run Floyd Warshall
	fw(V, G, d, p);
	
      	clock_t end = clock();
        double elapsedTime  = double(end - begin) / CLOCKS_PER_SEC;
	
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

	return 0;
}
