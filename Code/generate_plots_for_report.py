import neuroidal_model
import numpy as np

savepath = "../Figures/"
n = 30
d = 15
k = 3
r = 5
layout={i:np.random.rand(2) for i in range(n)}

print("*****CREATING VISUALIZATIONS FOR MODEL DESCRIPTION*****")
net = neuroidal_model.NeuroidalNet(n,d,k,r)
net.visualize_network(layout=layout, savename=savepath+"initialmodel")
net.create_item("A", disjoint=True)
net.visualize_network(layout=layout, savename=savepath+"itemrepresentation")

# n = 10
# neuron = 0
# p = 0.3
# a = np.random.choice([0.0, 1.0], (n,n), p=[1-p, p])
# np.fill_diagonal(a, 0)
# G = nx.Graph(a)
# nodelist = [n for n in G.neighbors(neuron)]
# pos = {i:np.random.rand(2) for i in range(n)}
# nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist, node_color="grey")
# nodes.set_edgecolor("yellow")
# nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=[neuron], node_color="blue")
# edgelist = [i for i in G.edges() if (i[0]== neuron)]
# nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist)


print("*****CREATING VISUALIZATIONS FOR JOIN*****")
net.create_item("B", disjoint=True)
net.visualize_network(layout=layout, savename=savepath+"beforeJOIN")
net.join('A', 'B', 'C')
net.visualize_network(layout=layout, savename=savepath+"afterJOIN")
print("Firing item A and B")
net.fire_item("A")
net.fire_item("B")
net.visualize_network(layout=layout, savename=savepath+"afterJOIN_fireAB")
net.run_firing_step()
net.visualize_network(layout=layout, savename=savepath+"afterJOIN_fireAB_stepone")
net.run_firing_step()
net.visualize_network(layout=layout, savename=savepath+"afterJOIN_fireAB_steptwo")
net.fire_item("B")
net.visualize_network(layout=layout, savename=savepath+"afterJOIN_fireB")
net.run_firing_step()
net.visualize_network(layout=layout, savename=savepath+"afterJOIN_fireB_stepone")

print("*****CREATING VISUALIZATIONS FOR LINK*****")
n = 30
d = 10
k = 2
r = 5
layout={i:np.random.rand(2) for i in range(n)}
net.reset_network(reset_synapse_strengths=True, reset_items=True)
net.create_item("A", disjoint=True)
net.create_item("B", disjoint=True)
net.visualize_network(layout=layout, savename=savepath+"beforeLINK")
net.link('A', 'B')
net.visualize_network(layout=layout, savename=savepath+"afterLINK")
net.fire_item('A')
net.visualize_network(layout=layout, savename=savepath+"afterLINK_fireA")
net.run_firing_step()
net.visualize_network(layout=layout, savename=savepath+"afterLINK_stepone")
net.run_firing_step()
net.visualize_network(layout=layout, savename=savepath+"afterLINK_steptwo")
