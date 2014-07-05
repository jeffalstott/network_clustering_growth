# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

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

def find_door(g, hinges, A2):
    neighbors = A2[array(hinges)]
    neighbors_t_order = argsort(neighbors.flatten())
    
    for e in arange(g.number_of_nodes()*2):
        t_index = neighbors_t_order[e] 
        i,x = unravel_index(t_index, shape(neighbors))
        hinge = hinges[i]
        y = hinges[not i]
        
        w = A2[hinge, y]
        t = A2[hinge, x]
        
        if w<=t:
            break
    
        if g.has_edge(hinge,x) and g.degree(x) > g.degree(y) and not g.edge[hinge][x]['fixed']:
            return hinge, x, y
    return None, None, None

def swing_door(g, hinge, x, y):
    g.remove_edge(hinge, x)
    g.add_edge(hinge, y, fixed=True)
    return g

def one_move(g, A2):
    w_order = argsort(A2.flatten())[::-1]
    A = nx.adjacency_matrix(g)
    edge_list = array(A).flatten()
    for w_index in w_order:
        if edge_list[w_index]:
            continue

        hinges = unravel_index(w_index, shape(A2))

        hinge, x, y = find_door(g, hinges, A2)
        if hinge:
            g = swing_door(g, hinge, x, y)
            return g, hinge, x, y
    return None

# <codecell>

n_nodes = 100
p =.005
p = 1.5*log(n_nodes)/n_nodes
g = nx.erdos_renyi_graph(n=n_nodes, p=p)
#g = random_graphs.erdos_renyi_graph(n=n_nodes, p=p)
try_count = 1
max_tries = 1000
while not nx.is_connected(g):
    g = nx.erdos_renyi_graph(n=n_nodes, p=p)
    try_count += 1
    if try_count>max_tries:
        print("Can't make a connected graph. Tried %i times."%max_tries)
        break

# <codecell>

nt, np = nt_np(g)

A = nx.adjacency_matrix(g)
A2 = A**2

change_percentage = 1
n_trials = floor(change_percentage*g.number_of_edges())
Cs = [nt/np]
C_locals = [nx.average_clustering(g)]
mean_k = [mean(g.degree().values())]
pl = [nx.average_shortest_path_length(g)]

initial_degrees = g.degree().values()

# <codecell>

for i,j in g.edges_iter():
    g[i][j]['fixed'] = False

print("Attempting %i trials"%n_trials)
for k in arange(n_trials):
    #print k
    if not k%10:
        print("Rewiring %i out of %i-----------------------------------"%(k,n_trials))

    A2 = array(A2)
    fill_diagonal(A2, 0)

    outputs = one_move(g, A2)
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
    A = nx.adjacency_matrix(g)
    A2 = A**2 #To be accelerated by A2(new) = A2(old) + AN + NA + N**2 where N is A(new)-A(old)
#     N = zeros(shape(A))
#     N[hinge,x] = N[x, hinge] = 1
#     N[hinge,y] = N[y, hinge] = -1
#     A2 = A2 + AN + NA + N**2
    mean_k.append(mean(g.degree().values()))
    Cs.append(nt/np)
    pl.append(nx.average_shortest_path_length(g))
    C_locals.append(nx.average_clustering(g))
    #if not k%10:
        #figure()
        #nx.draw_spectral(g)

# <codecell>

print("Rewired %.1f percent of edges"%(100*float(k)/n_trials))

# <codecell>

end_degrees = g.degree().values()

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
#ylabel("Total Clustering", color='b')
plot(arange(k+1)/k, C_locals/C_locals[0], color='r', label="Avg. Local")
#ylabel("Average Local Clustering", color='r')
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
text(.7, .5, "Clustering / Path Length,\nCompared to Random", transform=ax.transAxes, horizontalalignment='center')
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

hist(initial_degrees, label='Initial', normed=True)
hist(end_degrees, alpha=.8, label='After Rewiring', normed=True)
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

# <codecell>

from IPython.html import widgets
from IPython.display import display
from eventful_graph import EventfulGraph, empty_eventfulgraph_hook
from widget_forcedirectedgraph import ForceDirectedGraphWidget, publish_js
publish_js()

# <codecell>

floating_container = widgets.PopupWidget(default_view_name='ModalView')
floating_container.description = "Dynamic D3 rendering of a NetworkX graph"
floating_container.button_text = "Render Window"
floating_container.set_css({
    'width': '420px',
    'height': '350px'}, selector='modal')

#G = EventfulGraph()
d3 = ForceDirectedGraphWidget(g)

floating_container.children = [d3]
display(floating_container)

# <codecell>

from networkx.generators import random_graphs
from networkx.generators import classic

# Add a listener to the eventful graph's construction method.
# If an eventful graph is created, build and show a widget
# for the graph.
def handle_graph(graph):
    print(graph.graph._sleep)
    popup = widgets.PopupWidget()
    popup.description = "NetworkX Graph"
    popup.button_text = "Render Window"
    popup.set_css({
        'width': '420px',
        'height': '350px'}, selector='modal')
    graph_widget = ForceDirectedGraphWidget(graph)
    popup.children = [graph_widget]
    display(popup)
EventfulGraph.on_constructed(handle_graph)

# Replace the empty graph of the networkx classic module with
# the eventful graph type.
random_graphs.empty_graph = empty_eventfulgraph_hook(sleep=0.2)

# <codecell>

import d3py
 
with d3py.NetworkXFigure(g, name="graph",width=1000, height=1000) as p:
    p += d3py.ForceLayout()
    p.css['.node'] = {'fill': 'blue', 'stroke': 'magenta'}
    p.save_to_files()
    p.show() 

