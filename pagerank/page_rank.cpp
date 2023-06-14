#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <vector>
#include <omp.h>
#include <utility>
#include <algorithm>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

int inline GetOutEdgeNum(Graph g, int node)
{
  int start_edge = g->outgoing_starts[node];
  int end_edge = (node == g->num_nodes - 1)
                     ? g->num_edges
                     : g->outgoing_starts[node + 1];
  int ret = end_edge - start_edge;
}

int inline GetInEdgeNum(Graph g, int node)
{
  int start_edge = g->incoming_starts[node];
  int end_edge = (node == g->num_nodes - 1)
                     ? g->num_edges
                     : g->incoming_starts[node + 1];
  int ret = end_edge - start_edge;
}

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  // #pragma omp parallel for
  // for (int i = 0; i < numNodes; ++i)
  // {
  //   solution[i] = equal_prob;
  // }

  std::vector<double> score_old(numNodes);
  std::vector<double> score_new(numNodes);
  // std::vector<int> not_out_edge(numNodes);
  std::vector<int> has_out_edge(numNodes);
  std::vector<int> has_in_edge(numNodes);

  // std::fill(score_old.begin(), score_old.end(), equal_prob);
  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i)
  {
    score_old[i] = equal_prob;
  }
  
  #pragma omp parallel for
  for (int i = 0; i < has_out_edge.size(); i++)
  {
    has_out_edge[i] = outgoing_size(g, i);
  }
  #pragma omp parallel for
  for (int i = 0; i < has_in_edge.size(); i++)
  {
    has_in_edge[i] = incoming_size(g, i);
  }
  bool converged = false;

  while (!converged)
  {

    // step 1;
    #pragma omp parallel for
    for (int i = 0; i < score_new.size(); i++)
    {
      double sum = 0.0;
      #pragma omp parallel for reduction(+:sum)
      for (auto j = incoming_begin(g, i); j != incoming_end(g, i); j++)
      {
        sum += score_old[*j] / has_in_edge[*j];
      }
      score_new[i] = sum;
    }

    // step 2
    // std::for_each(score_new.begin(), score_new.end(), [&](double &vi)
    //               { vi = damping * vi + (1.f - damping) / numNodes; });
    #pragma omp parallel for
    for (int i  = 0 ; i < score_new.size(); i ++)
    {
      score_new[i] = damping * score_new[i] + (1.f - damping) / numNodes;
    }
    // step 3
    #pragma omp parallel for
    for (int i = 0; i < score_new.size(); i++)
    {
      auto&vi = score_new[i];
      #pragma omp parallel for reduction(+:vi)
      for (int j = 0; j < score_new.size(); j++)
      {
        if (!has_out_edge[j])
        {
          vi += damping * score_old[j] / numNodes;
        }
      }
    }
    double global_diff = 0.0;
    #pragma omp parallel for reduction(+:global_diff)
    for (int i = 0; i < score_new.size(); i++)
    {
      global_diff += std::abs(score_new[i] - score_old[i]);
    }

    #pragma omp parallel for 
    for (int i = 0; i < score_new.size(); i ++)
    {
      score_old[i] = score_new[i];
    }
    converged = (global_diff < convergence);
    // if (global_diff >= convergence)
    // {
    //   converged = true;
    //   std::transform(score_new.begin(), score_new.end(), solution, [](double &x)
    //                  { return x; });
    // }
    // else
    // {
    //   score_old = score_new;
    // }
  }
  std::transform(score_new.begin(), score_new.end(), solution, [](double &x)
                 { return x; });
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
