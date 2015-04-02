# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Setup
# ====

# <codecell>

import seaborn
import pandas as pd
%pylab inline

# <markdowncell>

# Define the algorithm
# ====

# <codecell>

def one_move_find_best(A, A2, fixed):
    most_wedges_order = argsort(A2.ravel())[::-1]
    doorways = array(unravel_index(most_wedges_order, shape(A2)))
    for doorway_index in range(doorways.shape[1]):
        doorway = tuple(doorways[:,doorway_index])
        if A[doorway]: #There's already an edge there
            continue
        if doorway[0]==doorway[1]: #The proposed target is a self link
            continue
#         print(doorway)
        door = find_door(A, doorway, A2, fixed)
        if door:
            hinge, target, latch = door
            A, fixed = swing_door(A, hinge, target, latch, fixed)
            return A, hinge, target, latch
    return None
    
def find_door(A, sides_of_doorway, A2, fixed):
    
    wedges_across_doorway = A2[sides_of_doorway]    
    
    def find_door_helper(hinge, latch, A, A2, fixed, wedges_across_doorway):
        latch_degree = A[latch].sum()
        neighbors_of_hinge = A[hinge].astype('bool')
        neighbors_degrees = A.sum(axis=1)
        wedges_across_candidate_doors = A2[hinge]        
        candidate_doors_not_fixed = ~fixed[hinge]
        candidate_doors = (neighbors_of_hinge * 
                           (wedges_across_doorway > wedges_across_candidate_doors) *
                           (neighbors_degrees > latch_degree) * 
                           candidate_doors_not_fixed
                           )
        if any(candidate_doors):
            best_door_for_this_hinge = ma.masked_array(A2[hinge], mask=~candidate_doors.astype('bool')).argmin()
            return best_door_for_this_hinge
        else:
            return None

    best_doors = [find_door_helper(sides_of_doorway[0], sides_of_doorway[1], A, A2, fixed, wedges_across_doorway),
                  find_door_helper(sides_of_doorway[1], sides_of_doorway[0], A, A2, fixed, wedges_across_doorway)]
    if best_doors[0] is None and best_doors[1] is None:
        return None
    elif best_doors[0] is None:
        best_door_index = 1
    elif best_doors[1] is None:
        best_door_index = 0
    else:
        best_door_index = argmin(A2[sides_of_doorway, best_doors])

    hinge = sides_of_doorway[best_door_index]
    target_neighbor = best_doors[best_door_index]
    latch = sides_of_doorway[not(best_door_index)]

    return hinge, target_neighbor, latch            


def swing_door(A, hinge, target_neighbor, latch, fixed):
    A[hinge, target_neighbor] = 0
    A[target_neighbor, hinge] = 0
    A[hinge, latch] = 1
    A[latch, hinge] = 1

    fixed[hinge, latch] = True
    fixed[latch, hinge] = True

    return A, fixed


def one_move_improve_worst(A, A2, fixed):
    most_wedges_order = argsort(A2.ravel()) ##Sorted from least to most
    doors = array(unravel_index(most_wedges_order, shape(A2)))
    for door_index in range(doors.shape[1]):
        door = tuple(doors[:,door_index])
        if not A[door]: #There's not an edge there
            continue
        if fixed[door]: #If the edge there has already been moved
            continue
        if door[0]==door[1]: #The proposed target is a self link
            continue
#         print(door)

        doorway = find_doorway(A, door, A2, fixed)
        if doorway:
            hinge, door_stop, latch = doorway
            A, fixed = swing_door(A, hinge, door_stop, latch, fixed)
            return A, hinge, door_stop, latch
    return None

def find_doorway(A, sides_of_door, A2, fixed):
    
    wedges_across_door = A2[sides_of_door]    
    
    def find_doorway_helper(hinge, edge_of_door, A, A2, fixed, wedges_across_door):
        edge_of_door_degree = A[edge_of_door].sum()
        neighbors_of_hinge = A[hinge].astype('bool')
        neighbors_degrees = A.sum(axis=1)
        wedges_across_candidate_doorways = A2[hinge]        
        candidate_doorways = (~neighbors_of_hinge * 
                           (wedges_across_door < wedges_across_candidate_doorways) *
                           (neighbors_degrees < edge_of_door_degree)
                           )
        candidate_doorways[hinge] = False #Can't connect to itself!
        if any(candidate_doorways):
            best_doorway_for_this_hinge = ma.masked_array(A2[hinge], mask=~candidate_doorways.astype('bool')).argmax()
            return best_doorway_for_this_hinge
        else:
            return None

    best_doorways = [find_doorway_helper(sides_of_door[0], sides_of_door[1], A, A2, fixed, wedges_across_door),
                  find_doorway_helper(sides_of_door[1], sides_of_door[0], A, A2, fixed, wedges_across_door)]
#     print(best_doorways)
#     print(sides_of_door)
    if best_doorways[0] is None and best_doorways[1] is None:
        return None
    elif best_doorways[0] is None:
        best_doorway_index = 1
    elif best_doorways[1] is None:
        best_doorway_index = 0
    else:
        best_doorway_index = argmin(A2[sides_of_door, best_doorways])

    hinge = sides_of_door[best_doorway_index]
    latch = best_doorways[best_doorway_index]
    door_stop = sides_of_door[not(best_doorway_index)]

    return hinge, door_stop, latch            

# <codecell>

import networkx as nx
def nt_np(G):
    triangles=0 # 6 times number of triangles
    contri=0  # 2 times number of connected triples
    for v,d,t in nx.algorithms.cluster._triangles_and_degree_iter(G):
        contri += d*(d-1)
        triangles += t
    if triangles==0: # we had no triangles or possible triangles
        return 0.0, float(contri)
    else:
        return triangles/6.0, float(contri)/2.0

# <markdowncell>

# Create the initial graph
# ===

# <codecell>

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

# <markdowncell>

# Calculate with Find_Best
# ====
# Find the optimal doorway for a link to move into, and find an adjacent door to rotate into it.

# <codecell>

### Get initial measurements of the original graph

g = original_graph.copy()

nt, np = nt_np(g)

A = nx.adjacency_matrix(g).todense()
A2 = A**2

change_percentage = 1
n_trials = floor(change_percentage*g.number_of_edges())
nts = [nt]
nps = [np]
Cs = [nt/np]
C_locals = [nx.average_clustering(g)]
mean_k = [mean(list(g.degree().values()))]
pl = [nx.average_shortest_path_length(g)]

initial_degrees = g.degree().values()

### Rewire graph
fixed = zeros(shape(A), dtype=bool)
print("Attempting %i edge rewires, out of %i edges"%(n_trials, g.number_of_edges()))
for k in arange(n_trials):
    if not k%10:
        print("Rewiring %i out of %i"%(k,n_trials))

    outputs = one_move_find_best(array(A), array(A2), fixed)
    if not outputs:
        print("Couldn't make a move!")
        break
    else:
        A_new, hinge, door_stop, latch = outputs
    
    g = nx.from_numpy_matrix(A_new)
    mean_k.append(mean(list(g.degree().values())))
    nt, np = nt_np(g)
    nts.append(nt)
    nps.append(np)
    Cs.append(nt/np)
    if Cs[-1]<Cs[-2]:
        print("Clustering went down! That shouldn't happen!")
        print("Tried to move the link between %i to %i to %i and %i"%(hinge, door_stop, hinge, latch))
        break
    pl.append(nx.average_shortest_path_length(g))
    C_locals.append(nx.average_clustering(g))
    
    A = A_new
    A2 = matrix(A)**2 

print("Rewired %.1f percent of edges"%(100*float(k)/n_trials))
# end_degrees = g.degree().values()

# rewired_graph = g.copy()

# <codecell>

find_best_Cs = array(Cs)
find_best_pls = array(pl)

# <markdowncell>

# Calculate with Improve_Worst
# ====
# Find the worst door, a link that spans very few wedges. Find the best doorway next to it that the door can rotate into.

# <codecell>

### Get initial measurements of the original graph

g = original_graph.copy()

nt, np = nt_np(g)

A = nx.adjacency_matrix(g).todense()
A2 = A**2

change_percentage = 1
n_trials = floor(change_percentage*g.number_of_edges())
nts = [nt]
nps = [np]
Cs = [nt/np]
C_locals = [nx.average_clustering(g)]
mean_k = [mean(list(g.degree().values()))]
pl = [nx.average_shortest_path_length(g)]

initial_degrees = g.degree().values()

### Rewire graph
fixed = zeros(shape(A), dtype=bool)
print("Attempting %i edge rewires, out of %i edges"%(n_trials, g.number_of_edges()))
for k in arange(n_trials):
    if not k%10:
        print("Rewiring %i out of %i"%(k,n_trials))

    outputs = one_move_improve_worst(array(A), array(A2), fixed)
    if not outputs:
        print("Couldn't make a move!")
        break
    else:
        A_new, hinge, door_stop, latch = outputs
    
    g = nx.from_numpy_matrix(A_new)
    mean_k.append(mean(list(g.degree().values())))
    nt, np = nt_np(g)
    nts.append(nt)
    nps.append(np)
    Cs.append(nt/np)
    if Cs[-1]<Cs[-2]:
        print("Clustering went down! That shouldn't happen!")
        print("Tried to move the link between %i to %i to %i and %i"%(hinge, door_stop, hinge, latch))
        break
    pl.append(nx.average_shortest_path_length(g))
    C_locals.append(nx.average_clustering(g))
    
    A = A_new
    A2 = matrix(A)**2 

print("Rewired %.1f percent of edges"%(100*float(k)/n_trials))
# end_degrees = g.degree().values()

# rewired_graph = g.copy()

# <codecell>

improve_worst_Cs = array(Cs)
improve_worst_pls = array(pl)

# <markdowncell>

# Measure the graph's properties during rewiring
# ====

# <codecell>

plot(improve_worst_Cs/improve_worst_Cs[0], color='b', label="Improve Worst")
plot(find_best_Cs/find_best_Cs[0], color='r', label="Find Best")
ylabel("Clustering Increase From Initial")
title("Clustering Goes Up with Both Algorithms")
xlabel("Number of Rewires")

legend(loc=4)

# <codecell>

plot(improve_worst_pls/improve_worst_pls[0], color='b', label="Improve Worst")
plot(find_best_pls/find_best_pls[0], color='r', label="Find Best")
ylabel("Path Length Increase From Initial")
title("Path Length Goes Up with Both Algorithms")
xlabel("Number of Rewires")

legend(loc=4)

# <codecell>

improve_worst_swis = (improve_worst_Cs/improve_worst_Cs[0]) / (improve_worst_pls/improve_worst_pls[0])

find_best_swis = (find_best_Cs/find_best_Cs[0]) / (find_best_pls/find_best_pls[0])

plot(improve_worst_swis, color='b', label="Improve Worst")
plot(find_best_swis, color='r', label="Find Best")

ylabel("Small World Index")
xlabel("Number of Rewires")
title("Small World Index Grows with Rewiring, then Plateaus")

