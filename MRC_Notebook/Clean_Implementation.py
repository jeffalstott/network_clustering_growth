# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Setup
# ====

# <codecell>

%matplotlib inline

# <codecell>

from pylab import *

# <markdowncell>

# Define the algorithm
# ====

# <codecell>

import networkx as nx

# def one_move(g,A2):
#     doorway = find_doorway(g,A2)
#     hinge, target, latch = find_door(g,A2,doorway)
#     swing_door(g, hinge, target, latch)
#     return g, hinge, target, latch

def one_move(g,A2):
    most_wedges_order = argsort(A2.ravel())[::-1]
    doorways = array(unravel_index(most_wedges_order, shape(A2)))
    for doorway_index in range(doorways.shape[1]):
        doorway = tuple(doorways[:,doorway_index])
        if g.has_edge(*doorway): #There's already an edge there
            continue
        if doorway[0]==doorway[1]: #The proposed target is a self link
            continue

        door = find_door(g, doorway, A2)
        if door:
            hinge, target, latch = door
            g = swing_door(g, hinge, target, latch)
            return g, hinge, target, latch
    return None
    
def find_door(g, doorway, A2):
    wedges_across_doorway = A2[doorway]    
    neighbors_wedges = A2[array(doorway)]
    neighbors_least_wedges_order = argsort(neighbors_wedges.ravel())
    candidate_doors = array(unravel_index(neighbors_least_wedges_order, shape(neighbors_wedges)))
    
    for candidate_door_index in range(candidate_doors.shape[1]):
        side_of_door_with_hinge, target_neighbor = candidate_doors[:,candidate_door_index]
        
        hinge = doorway[side_of_door_with_hinge]
        latch = doorway[not side_of_door_with_hinge]  

        wedges_across_candidate_door_position = A2[hinge, target_neighbor]
        
        #We want to complete more wedges than we open. So if the wedges across the doorway is less 
        #than those across the current door (all of which are currently triangles), then we don't
        #want to make this move. 
        if wedges_across_doorway<=wedges_across_candidate_door_position: 
            return None
            #Because the wedges_across_candidate_door_position is sorted from low to high, if 
            #we meet this condition once, then we know the rest of the candidate door positions will
            #be even worse, so we should stop
        
        if (g.has_edge(hinge,target_neighbor) and 
            g.degree(target_neighbor) > g.degree(latch) and 
            not g.edge[hinge][target_neighbor]['fixed']):
            return hinge, target_neighbor, latch
    return None

def swing_door(g, hinge, target_neighbor, latch):
    g.remove_edge(hinge, target_neighbor)
    g.add_edge(hinge, latch, fixed=True)
    return g

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

original_graph = g.copy()

try_count = 1
max_tries = 1000
while not nx.is_connected(g):
    g = nx.erdos_renyi_graph(n=n_nodes, p=p)
    try_count += 1
    if try_count>max_tries:
        print("Can't make a connected graph. Tried %i times."%max_tries)
        break
print("Average degree: %.2f"%mean(list(g.degree().values())))

# <markdowncell>

# Get initial measurements of the original graph
# ====

# <codecell>

nt, np = nt_np(g)

A = nx.adjacency_matrix(g).todense()
A2 = A**2

change_percentage = 1
n_trials = floor(change_percentage*g.number_of_edges())
Cs = [nt/np]
C_locals = [nx.average_clustering(g)]
mean_k = [mean(list(g.degree().values()))]
pl = [nx.average_shortest_path_length(g)]

initial_degrees = g.degree().values()

# <markdowncell>

# Rewire the graph
# ===

# <codecell>

for i,j in g.edges_iter():
    g[i][j]['fixed'] = False

print("Attempting %i edge rewires, out of %i edges"%(n_trials, g.number_of_edges()))
for k in arange(n_trials):
    #print k
    if not k%10:
        print("Rewiring %i out of %i"%(k,n_trials))

#     A2 = array(A2)
#     fill_diagonal(A2, 0)

    outputs = one_move(g, array(A2))
    if not outputs:
        print("Couldn't make a move!")
        break
    else:
        g, hinge, x, y = outputs
    
    w = A2[hinge, y]
    t = A2[hinge, x]
    #nt = nt + w - t
    #np = np + g.degree(y) - g.degree(x) + 1 
    nt, np = nt_np(g)
#     A = nx.adjacency_matrix(g).todense()
#     A2 = A**2 #To be accelerated by A2(new) = A2(old) + AN + NA + N**2 where N is A(new)-A(old)
    Anew = nx.adjacency_matrix(g).todense()
    N = Anew-A
    A2 = A2 + Anew*N + N*Anew + N**2
#     N = zeros(shape(A))
#     N[hinge,x] = N[x, hinge] = 1
#     N[hinge,y] = N[y, hinge] = -1
#     A2 = A2 + AN + NA + N**2
    mean_k.append(mean(list(g.degree().values())))
    Cs.append(nt/np)
    pl.append(nx.average_shortest_path_length(g))
    C_locals.append(nx.average_clustering(g))
print("Rewired %.1f percent of edges"%(100*float(k)/n_trials))
end_degrees = g.degree().values()

rewired_graph = g.copy()

# <markdowncell>

# Measure the graph's properties during rewiring
# ====

# <codecell>

Cs = array(Cs)
C_locals = array(C_locals)

plot(arange(k+1)/k,Cs/Cs[0], color='b', label="Clustering")
ylabel("Total Clustering", color='b')
twinx()
plot(arange(k+1)/k, C_locals/C_locals[0], color='r', label="Clustering Local")
ylabel("Average Local Clustering", color='r')
title("Clustering Goes Up, With Two Definitions")
xlabel("Percent of Rewirings")

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

savefig("Model-Clustering_Increases.pdf")

# <codecell>

Cs = array(Cs)
pl = array(pl)

plot(arange(k+1)/k,Cs/Cs[0], color='b', label="Clustering")
ylabel("Total Clustering", color='b')
xlabel("Percent of Rewirings")
ax = gca()
ax.spines['top'].set_visible(False)
ax.get_xaxis().tick_bottom()


twinx()
plot(arange(k+1)/k, pl/pl[0], color='r', label="Average Path Length")
ylabel("Average Path Length", color='r')

title("Path Length Also Increases, Though More Slowly")


ax = gca()
ax.spines['top'].set_visible(False)
ax.get_xaxis().tick_bottom()

savefig("Model-Clustering_vs_Path_Length.pdf")

# <codecell>

Gamma = Cs/Cs[0]
Lambda = pl/pl[0]

swi = Gamma/Lambda

f = figure()
ax = f.add_subplot(1,1,1)
x = arange(k+1)/k
plot(x,swi)
text(.7, .5, "Clustering / Path Length,\nCompared to Initial", transform=ax.transAxes, horizontalalignment='center')
ylabel("Small World Index")
xlabel("Percent of Rewirings")
title("Small World Index Grows with Rewiring, then Plateaus")

ax = gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

savefig("Model-Small_World_Index.pdf")

# <codecell>

hist(list(initial_degrees), label='Initial', normed=True)
hist(list(end_degrees), alpha=.8, label='After Rewiring', normed=True)
legend()
title("Degree Distribution Changes With Rewiring")
ylabel("p(Degree)")
xlabel("Degree")

lg = legend()
lg.draw_frame(False)

ax = gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

savefig("Model-Degree_Distribution.pdf")

# <markdowncell>

# Visualize the graph before and afterward
# ===

# <codecell>

figure()
nx.draw(original_graph)
title("Original Graph")

figure()
nx.draw(rewired_graph)
title("Rewired Graph")

# <markdowncell>

# Interactive visualization
# ===
# Run both of the cells below, and the graph will display between them. Yes, it's awkward right now.

# <codecell>

graph_to_be_visualized = rewired_graph

import json
from networkx.readwrite import json_graph

data = json_graph.node_link_data(graph_to_be_visualized)

with open('graph.json', 'w') as f:
    json.dump(data, f, indent=4)
    
from IPython.display import HTML
HTML(
"""<div id="d3-example"></div>
<style>
.node {stroke: #fff; stroke-width: 1.5px;}
.link {stroke: #999; stroke-opacity: .6;}
</style>"""
     )

# <codecell>

from IPython.display import Javascript
Javascript(
"""
// We load the d3.js library from the Web.
require.config({paths: {d3: "http://d3js.org/d3.v3.min"}});
require(["d3"], function(d3) {
    // The code in this block is executed when the 
    // d3.js library has been loaded.
    
    // First, we specify the size of the canvas containing
    // the visualization (size of the <div> element).
    var width = 800,
        height = 800;

    // We create a color scale.
    var color = d3.scale.category10();

    // We create a force-directed dynamic graph layout.
    var force = d3.layout.force()
        .charge(-120)
        .linkDistance(30)
        .size([width, height]);

    // In the <div> element, we create a <svg> graphic
    // that will contain our interactive visualization.
    var svg = d3.select("#d3-example").select("svg")
    if (svg.empty()) {
        svg = d3.select("#d3-example").append("svg")
                    .attr("width", width)
                    .attr("height", height);
    }
        
    // We load the JSON file.
    d3.json("graph.json", function(error, graph) {
        // In this block, the file has been loaded
        // and the 'graph' object contains our graph.
        
        // We load the nodes and links in the force-directed
        // graph.
        force.nodes(graph.nodes)
            .links(graph.links)
            .start();

        // We create a <line> SVG element for each link
        // in the graph.
        var link = svg.selectAll(".link")
            .data(graph.links)
            .enter().append("line")
            .attr("class", "link");

        // We create a <circle> SVG element for each node
        // in the graph, and we specify a few attributes.
        var node = svg.selectAll(".node")
            .data(graph.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", 5)  // radius
            .style("fill", function(d) {
                // The node color depends on the club.
                return color(d.club); 
            })
            .call(force.drag);

        // The name of each node is the node number.
        node.append("title")
            .text(function(d) { return d.name; });

        // We bind the positions of the SVG elements
        // to the positions of the dynamic force-directed graph,
        // at each time step.
        force.on("tick", function() {
            link.attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });

            node.attr("cx", function(d) { return d.x; })
                .attr("cy", function(d) { return d.y; });
        });
    });
});
"""
)

# <markdowncell>

# "When we execute this cell, the HTML object created in the previous cell is updated. The graph is animated and interactive: we can click on nodes, see their labels, and move them within the canvas."
# Code from http://nbviewer.ipython.org/github/dboyliao/cookbook-code/blob/master/notebooks/chapter06_viz/04_d3.ipynb

