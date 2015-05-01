# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pylab import *

def one_move_find_best(A, A2, fixed):
    most_wedges_order = argsort(A2.ravel())[::-1]
    doorways = array(unravel_index(most_wedges_order, shape(A2)))
    for doorway_index in range(doorways.shape[1]):
        doorway = tuple(doorways[:,doorway_index])
        if A[doorway]: #There's already an edge there
            continue
        if doorway[0]==doorway[1]: #The proposed target is a self link
            continue
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


def move_edge_A2(A2, A, hinge, doorstop, latch):
    #Update the neighborhood of the hinge
    A2[hinge] = A2[hinge] + A[latch] - A[doorstop]
    A2[:,hinge] = A2[:, hinge] + A[:, latch] - A[:, doorstop]
    
    #Update the neighborhood of the doorstop
    A2[doorstop] -= A[hinge]
    A2[:,doorstop] -= A[:, hinge]

    #Update the neighborhood of the latch
    A2[latch] += A[hinge]
    A2[:,latch] += A[:, hinge]
    
    #Update the degrees of the hinge, doorstop and latch, compensating for the changes just made
    A2[hinge,hinge] -= 2*A[latch,hinge]
    A2[doorstop,doorstop] -= 1
    A2[latch,latch] -= A[hinge, latch]
    
    #Finally, we have accidentally messed up the entires Aˆ2(j,k) and Aˆ2(k,j) in the row/column updates, so we need to compensate for that:
    A2[doorstop, latch] += A[hinge, latch]
    A2[latch, doorstop] += A[latch, hinge]
    return A2

def number_of_triangles(A, A2):
    from networkx import triangles, Graph
    return sum(list(triangles(Graph(g)).values()))/3

def number_of_possible_triangles(A, A2):
    import networkx as nx
    contri=0  # 2 times number of connected triples
    for v,d,t in nx.algorithms.cluster._triangles_and_degree_iter(nx.Graph(A)):
        contri += d*(d-1)
    return float(contri)/2.0

def number_of_triangles_update(nt, A, A2, hinge, doorstop, latch):
    return nt + A2[hinge, latch] - A2[hinge,doorstop]

def number_of_possible_triangles_update(np, A, A2, hinge, doorstop, latch):
    return np + sum(A[latch])-sum(A[doorstop]) + 1
    
    
def cluster_rewire_graph(A, 
                 percent_of_edges_to_rewire = 1, 
                 n_trials = None,
                 rewire_function = one_move_find_best,
                 verbose = True,
                 verbose_count = 10,
                 property_functions = [number_of_triangles, 
                                       number_of_possible_triangles]):
    
    A2 = array(matrix(A)**2)
    A = array(A)
    n_edges = A.sum()/2
    fixed = zeros(shape(A), dtype=bool)
    
    if property_functions:
        n_properties = len(property_functions)
        properties = [[] for i in range(n_properties)]
        for nth_property in range(n_properties):
            prop_fun = property_functions[nth_property]
            if not callable(prop_fun):
                prop_fun = prop_fun[0]
            properties[nth_property].append(prop_fun(A,A2))
    
    if n_trials is None:
        n_trials = floor(percent_of_edges_to_rewire*n_edges)
    if verbose:
        print("Attempting %i edge rewires, out of %i edges"%(n_trials, n_edges))

    ### Rewire graph    
    for k in arange(n_trials):
        if not k%verbose_count:
            if verbose:
                print("Rewiring %i out of %i"%(k,n_trials))

        outputs = rewire_function(A, A2, fixed)
        if not outputs:
            if verbose:
                print("Couldn't make a move!")
            break
        else:
            A, hinge, doorstop, latch = outputs
            A2 = move_edge_A2(A2, A, hinge, doorstop, latch)
                
        if property_functions:
            for nth_property in range(n_properties):
                prop_fun = property_functions[nth_property]
                if callable(prop_fun):
                    updated_property = prop_fun(A,A2)
                else:
                    prop_fun = prop_fun[1]
                    previous_property = properties[nth_property][-1]
                    updated_property = property_update_functions[nth_property](previous_property,
                                                                              A,
                                                                              A2,
                                                                              hinge,
                                                                              doorstop,
                                                                              latch)
                properties[nth_property].append(updated_property)   
    
    if verbose:
        print("Rewired %.1f percent of edges"%(100*float(k)/n_trials))
    if property_functions:
        return A, properties
    else:
        return A

# <codecell>

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

