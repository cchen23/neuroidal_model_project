import neuroidal_model
import numpy as np

savepath = "../Figures/"
n = 30
d = 10
k = 2
r = 5
layout={i:np.random.rand(2) for i in range(n)}

print("Creating neuroidal model for n, d, k, r = %d, %d, %d, %d" % (n, d, k, r))
net = neuroidal_model.NeuroidalNet(n,d,k,r)
net.visualize_network(layout=layout, savename=savepath+"initialmodel")
net.create_item("A", disjoint=True)
net.visualize_network(layout=layout, savename=savepath+"itemrepresentation")


print("JOIN")
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

print("LINK")
net.reset_network(reset_synapse_strengths=True, reset_items=True)
net.create_item("A", disjoint=True)
net.create_item("B", disjoint=True)
net.link('A', 'B')
net.visualize_network(layout=layout, savename=savepath+"afterLINK")
net.fire_item('A')
net.visualize_network(layout=layout, savename=savepath+"afterLINK_fireA")
net.run_firing_step()
net.visualize_network(layout=layout, savename=savepath+"afterLINK_stepone")
net.run_firing_step()
net.visualize_network(layout=layout, savename=savepath+"afterLINK_steptwo")
