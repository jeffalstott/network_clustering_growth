
# coding: utf-8

# Setup
# ====

# In[1]:

import seaborn
import pandas as pd
get_ipython().magic('pylab inline')


# In[2]:

import clusterrewire as cr
from clusterrewire import cluster_rewire_graph
import networkx as nx


# Create the initial graph
# ===

# In[6]:

n_nodes = 500
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

get_ipython().run_cell_magic('time', '', 'g = original_graph.copy()\nA = nx.adjacency_matrix(g).todense()\nfind_best_A, (triangles_completed, triangles_possible) = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best)\nfind_best_clustering = array(triangles_completed)/array(triangles_possible)')


# In[5]:

plot(find_best_clustering)
min(diff(find_best_clustering))


# In[5]:

plot(find_best_clustering)
min(diff(find_best_clustering))


# In[6]:

get_ipython().run_cell_magic('time', '', 'g = original_graph.copy()\nA = nx.adjacency_matrix(g).todense()\nfind_best_A = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best,\n                                                                             property_functions=None)')


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


# In[7]:

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


# In[12]:

g = original_graph.copy()
A = nx.adjacency_matrix(g).todense()
find_best_A, (triangles_completed, triangles_possible) = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best)
find_best_clustering = array(triangles_completed)/array(triangles_possible)


# In[13]:

plot(array(triangles_completed)/array(triangles_possible))
min(diff(array(triangles_completed)/array(triangles_possible)))


# In[28]:

def number_of_triangles_update(nt, A, A2, hinge, doorstop, latch):
    #return nt + A2[hinge, latch] - A2[hinge,doorstop] #This isn't working for some reason and I don't know why
#     return nt + A2[hinge, latch] - A2[hinge,doorstop] + A[latch, doorstop]
    return nt + A2[hinge, latch] - A2[hinge,doorstop] + A[latch, doorstop]

def number_of_possible_triangles_update(np, A, A2, hinge, doorstop, latch):
#     return np + sum(A[latch]) - sum(A[doorstop]) -1 #This isn't working for some reason and I don't know why
    return np + (sum(A[latch])-1) - sum(A[doorstop])#This isn't working for some reason and I don't know why

g = original_graph.copy()
A = nx.adjacency_matrix(g).todense()
find_best_A, (triangles_completed_test, triangles_possible_test) = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best,
                                                                                        property_functions = [(cr.number_of_triangles, 
                                                                                                               number_of_triangles_update),
                                                                                                               (cr.number_of_possible_triangles,
                                                                                                                number_of_possible_triangles_update)])


# In[29]:

plot(array(triangles_completed_test)/array(triangles_possible_test))
min(diff(array(triangles_completed_test)/array(triangles_possible_test)))


# In[30]:

scatter(diff(triangles_completed),diff(triangles_completed_test))
plot((0,7),(0,7))
ylabel("True")
xlabel("My Code")
figure()
scatter(diff(triangles_possible),diff(triangles_possible_test))
ylabel("True")
xlabel("My Code")
plot((-2,4),(-2,4))
#plot(triangles_possible,triangles_possible_test)


# In[20]:

get_ipython().run_cell_magic('time', '', 'g = original_graph.copy()\nA = nx.adjacency_matrix(g).todense()\nfind_best_A, (triangles_completed_test, triangles_possible_test) = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best,\n                                                                                        property_functions = [(cr.number_of_triangles, \n                                                                                                               number_of_triangles_update),\n                                                                                                               (cr.number_of_possible_triangles,\n                                                                                                                number_of_possible_triangles_update)])')


# In[21]:

get_ipython().run_cell_magic('time', '', 'g = original_graph.copy()\nA = nx.adjacency_matrix(g).todense()\nfind_best_A, (triangles_completed, triangles_possible) = cluster_rewire_graph(A, rewire_function=cr.one_move_find_best)')

