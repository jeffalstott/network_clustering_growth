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

def one_move(A,A2, fixed, use_for_loops=True):
    most_wedges_order = argsort(A2.ravel())[::-1]
    doorways = array(unravel_index(most_wedges_order, shape(A2)))
    for doorway_index in range(doorways.shape[1]):
        doorway = tuple(doorways[:,doorway_index])
        if A[doorway]: #There's already an edge there
            continue
        if doorway[0]==doorway[1]: #The proposed target is a self link
            continue
#         print(doorway)
        door = find_door(A, doorway, A2, fixed, use_for_loops)
        if door:
            hinge, target, latch = door
            A, fixed = swing_door(A, hinge, target, latch, fixed)
            return A, hinge, target, latch
    return None
    
def find_door(A, sides_of_doorway, A2, fixed, use_for_loops=False):
    
    wedges_across_doorway = A2[sides_of_doorway]    
    
    if not use_for_loops:
        def find_door_helper(hinge, latch, A, A2, fixed, wedges_across_doorway):
            latch_degree = A[latch].sum()
            neighbors_of_hinge = A[hinge]
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

    else:
        neighbors_wedges = A2[array(sides_of_doorway)]
        neighbors_least_wedges_order = argsort(neighbors_wedges.ravel())
        candidate_doors = array(unravel_index(neighbors_least_wedges_order, shape(neighbors_wedges)))

        for candidate_door_index in range(candidate_doors.shape[1]):
            side_of_door_with_hinge, target_neighbor = candidate_doors[:,candidate_door_index]

            hinge = sides_of_doorway[side_of_door_with_hinge]
            latch = sides_of_doorway[not side_of_door_with_hinge]  

            wedges_across_candidate_door_position = A2[hinge, target_neighbor]
            
            #We want to complete more wedges than we open. So if the wedges across the doorway is less 
            #than those across the current door (all of which are currently triangles), then we don't
            #want to make this move. 
            if wedges_across_doorway<=wedges_across_candidate_door_position: 
                return None
                #Because the wedges_across_candidate_door_position is sorted from low to high, if 
                #we meet this condition once, then we know the rest of the candidate door positions will
                #be even worse, so we should stop
            if (A[hinge,target_neighbor] and 
                sum(A[target_neighbor]) > sum(A[latch]) and 
                not fixed[hinge,target_neighbor]):
                return hinge, target_neighbor, latch
        return None

def swing_door(A, hinge, target_neighbor, latch, fixed):
    A[hinge, target_neighbor] = 0
    A[target_neighbor, hinge] = 0
    A[hinge, latch] = 1
    A[latch, hinge] = 1

    fixed[hinge, latch] = True
    fixed[latch, hinge] = True

    return A, fixed


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

# <codecell>

g = original_graph.copy()

# <markdowncell>

# Get initial measurements of the original graph
# ====

# <codecell>

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

# <markdowncell>

# Rewire the graph
# ===

# <codecell>

fixed = zeros(shape(A), dtype=bool)
print("Attempting %i edge rewires, out of %i edges"%(n_trials, g.number_of_edges()))
for k in arange(n_trials):
    if not k%10:
        print("Rewiring %i out of %i"%(k,n_trials))

    outputs = one_move(array(A), array(A2), fixed)
    if not outputs:
        print("Couldn't make a move!")
        break
    else:
        A_new, hinge, target, latch = outputs
    
    g = nx.from_numpy_matrix(A_new)
    mean_k.append(mean(list(g.degree().values())))
    nt, np = nt_np(g)
    nts.append(nt)
    nps.append(np)
    Cs.append(nt/np)
    if Cs[-1]<Cs[-2]:
        print("Clustering went down! That shouldn't happen!")
        print("Tried to move the link between %i to %i to %i and %i"%(hinge, target, hinge, latch))
        break
    pl.append(nx.average_shortest_path_length(g))
    C_locals.append(nx.average_clustering(g))
    
    A = A_new
    A2 = matrix(A)**2 

print("Rewired %.1f percent of edges"%(100*float(k)/n_trials))
# end_degrees = g.degree().values()

# rewired_graph = g.copy()

# <markdowncell>

# Measure the graph's properties during rewiring
# ====

# <codecell>

Cs = array(Cs)
C_locals = array(C_locals)

plot(arange(k+1)/k,Cs/Cs[0], color='b', label="Total (Triange Density)")
plot(arange(k+1)/k, C_locals/C_locals[0], color='r', label="Avg. Local")
ylabel("Clustering Increase From Initial")
title("Clustering Goes Up, With Two Definitions")
xlabel("Percent of Rewirings")

lg = legend(loc=4)
lg.draw_frame(False)

ax = gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

