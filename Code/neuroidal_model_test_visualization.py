import neuroidal_model
import numpy as np

n = 30
d = 10
k = 2
r = 5
print("Creating neuroidal model for n, d, k, r = %d, %d, %d, %d" % (n, d, k, r))
net = neuroidal_model.NeuroidalNet(n,d,k,r)

print("\nCreating items")
net.create_item("A")
net.stored_items
net.create_item("B")
net.stored_items
#
# print("\nTest Firing")
# net.fire_item("A")
# net.neuron_firings
# net.neuron_memories
# print(net.get_firing_items())
# net.fire_item("B")
# net.neuron_firings
# net.neuron_memories
# net.visualize_network()
# print(net.get_firing_items())
# net.turn_off_all_firing()
# net.neuron_firings
# net.neuron_memories
# print(net.get_firing_items())


print("\nTest Joining")
print("Joining item A and B")
net.join('A', 'B', 'C')
layout=net.visualize_network()
print("Firing item A and B")
net.fire_item("A")
net.fire_item("B")
net.visualize_network(layout=layout)
print(layout)

print("Firing items: ")
print(net.get_firing_items())
print("Running update step:")
net.run_firing_step()
print("Firing items: ")
print(net.get_firing_items())
net.visualize_network(layout=layout)
