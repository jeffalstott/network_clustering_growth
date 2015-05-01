# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import networkx as nx
import community 
import random
from math import floor
import clusterrewire
%pylab inline

# <codecell>

def sbm(cmtysize, pin, pout):
    graphs = []
    for i in range(0, len(cmtysize)):
        graphs.append(nx.gnp_random_graph(cmtysize[i], pin))
    G=nx.disjoint_union_all(graphs)

    s=[]
    s.append(0)
    for i in range(0, len(cmtysize)):
        s.append(s[i-1]+cmtysize[i])

    for i in range(0, len(cmtysize)):
        for n in range(s[i], s[i+1]):
            for m in range(s[i+1], G.number_of_nodes()):
                if rand()<pout:
                        G.add_edge(n, m)
    return G;

# <codecell>

g=sbm([20, 20, 20], .64, .5)
partition_before_rewire = community.best_partition(g)
sets_before_rewire = [[]]
for i in range(0, len(partition_before_rewire)):
    s = partition_before_rewire[i]
    if s>len(sets_before_rewire)-1:
        sets_before_rewire.append([])
    sets_before_rewire[s].append(i)

print(sets_before_rewire)

# <codecell>

imshow(nx.to_numpy_matrix(g))
title("Network before rewiring")

# <codecell>

A = nx.to_numpy_matrix(g)
A = clusterrewire.cluster_rewire_graph(A,
                                       percent_of_edges_to_rewire=1,
                                       verbose=False,
                                       property_functions=None)
rewired_graph = nx.Graph(A)

# <codecell>

partition_after_rewire = community.best_partition(rewired_graph)
sets_after_rewire = [[]]
for i in range(0, len(partition_after_rewire)):
    s = partition_after_rewire[i]
    if s>len(sets_after_rewire)-1:
        sets_after_rewire.append([])
    sets_after_rewire[s].append(i)

print(sets_after_rewire)

imshow(nx.to_numpy_matrix(rewired_graph))
title("Adjacency after rewire")

