
# coding: utf-8

# Setup
# ====

# In[22]:

import seaborn
import pandas as pd
get_ipython().magic('pylab inline')


# In[23]:

import clusterrewire as cr
from clusterrewire import cluster_rewire_graph
import networkx as nx


# Create the initial graph
# ===

# In[34]:

n_nodes = 100
p = 1.5*log(n_nodes)/n_nodes
g = nx.erdos_renyi_graph(n=n_nodes, p=p)

try_count = 1
max_tries = 1000
while not nx.is_connected(g):
    g = nx.erdos_renyi_graph(n=n_nodes, p=p)
    try_count += 1
    if try_count>max_tries:
        print("Can't make a connected graph. Tried %i times."%max_tries)
        break

original_graph = g.copy()

print("Average degree: %.2f"%mean(list(g.degree().values())))


# Cluster up
# ====

# In[7]:

g = original_graph.copy()
A = nx.adjacency_matrix(g).todense()
find_best_A, (triangles_completed, triangles_possible) = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best)
find_best_clustering = array(triangles_completed)/array(triangles_possible)


# In[5]:

g = original_graph.copy()
A = nx.adjacency_matrix(g).todense()
improve_worst_A, (triangles_completed, triangles_possible) = cluster_rewire_graph(A, rewire_function=cr.one_move_improve_worst)
improve_worst_clustering = array(triangles_completed)/array(triangles_possible)


# In[6]:

plot(improve_worst_clustering/improve_worst_clustering[0], color='b', label="Improve Worst")
plot(find_best_clustering/find_best_clustering[0], color='r', label="Find Best")
ylabel("Clustering Increase From Initial")
title("Clustering Goes Up with Both Algorithms")
xlabel("Number of Rewires")

legend(loc=4)

figure()
hist(sum(find_best_A,axis=1), alpha=.7, label='Rewired (Find Best)', normed=True)
hist(sum(improve_worst_A,axis=1), alpha=.7, label='Rewired (Improve Worst)', normed=True)
hist(sum(nx.adjacency_matrix(g).todense(),axis=1), alpha=.7, label='Original Network', normed=True)
ylabel("P(Degree)")
xlabel("Node Degree")
legend(loc=1)


# In[ ]:

g = original_graph.copy()
A = nx.adjacency_matrix(g).todense()
find_best_A, (triangles_completed, triangles_possible) = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best,
                                                                             preserve_degrees=True)
find_best_clustering = array(triangles_completed)/array(triangles_possible)

g = original_graph.copy()
A = nx.adjacency_matrix(g).todense()
improve_worst_A, (triangles_completed, triangles_possible) = cluster_rewire_graph(A, rewire_function=cr.one_move_improve_worst,
                                                                                 preserve_degrees=True)
improve_worst_clustering = array(triangles_completed)/array(triangles_possible)

plot(improve_worst_clustering/improve_worst_clustering[0], color='b', label="Improve Worst")
plot(find_best_clustering/find_best_clustering[0], color='r', label="Find Best")
ylabel("Clustering Increase From Initial")
title("Clustering Goes Up with Both Algorithms")
xlabel("Number of Rewires")

legend(loc=4)

figure()
hist(sum(find_best_A,axis=1), alpha=.7, label='Rewired (Find Best)', normed=True)
hist(sum(improve_worst_A,axis=1), alpha=.7, label='Rewired (Improve Worst)', normed=True)
hist(sum(nx.adjacency_matrix(g).todense(),axis=1), alpha=.7, label='Original Network', normed=True)
ylabel("P(Degree)")
xlabel("Node Degree")
legend(loc=1)


# In[36]:

g = original_graph.copy()
A = nx.adjacency_matrix(g).todense()
find_best_A = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best,
                                                                             preserve_degrees=True,
                                                                             property_functions=None)

g = original_graph.copy()
A = nx.adjacency_matrix(g).todense()
improve_worst_A = cluster_rewire_graph(A, rewire_function=cr.one_move_improve_worst,
                                                                                 preserve_degrees=True,
                                                                                 property_functions=None)
figure()
hist(sum(find_best_A,axis=1), alpha=.7, label='Rewired (Find Best)', normed=True)
hist(sum(improve_worst_A,axis=1), alpha=.7, label='Rewired (Improve Worst)', normed=True)
hist(sum(nx.adjacency_matrix(g).todense(),axis=1), alpha=.7, label='Original Network', normed=True)
ylabel("P(Degree)")
xlabel("Node Degree")
legend(loc=1)


# In[25]:

# g = original_graph.copy()
# A = nx.adjacency_matrix(g).todense()
# find_best_A, (triangles_completed, triangles_possible) = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best)
# find_best_clustering = array(triangles_completed)/array(triangles_possible)
# g = original_graph.copy()
# A = nx.adjacency_matrix(g).todense()
# find_best_A, (triangles_completed_test, triangles_possible_test) = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best,
#                                                                                         property_functions = [cr.number_of_triangles,cr.number_of_possible_triangles])
# scatter(diff(triangles_completed),diff(triangles_completed_test))
# plot((0,7),(0,7))
# ylabel("True")
# xlabel("My Code")
# scatter(diff(triangles_possible),diff(triangles_possible_test))
# ylabel("True")
# xlabel("My Code")
# plot(triangles_possible,triangles_possible_test)

