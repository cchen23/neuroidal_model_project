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

print("\nTest Joining")
print("Joining item A and B")
net.join('A', 'B', 'C')
print("Firing item A and B")
net.fire_item("A")
net.fire_item("B")
net.visualize_network(layout=layout)
print("Firing items: ")
print(net.get_firing_items())
print("Running update step:")
net.run_firing_step()
print("Firing items: ")
print(net.get_firing_items())
net.visualize_network(layout=layout)


net.reset_network(reset_synapse_strengths=True)
print("\nTest Linking")
net.turn_off_all_firing()
net.fire_item('A')
print('after firing A from scratch, firing items are: ', net.get_firing_items())
net.run_firing_step()
net.visualize_network(layout=layout)
print('after firing A from scratch, waiting 1 step, firing items are: ', net.get_firing_items())
net.visualize_network(layout=layout)
net.turn_off_all_firing()
net.fire_item('B')
print('after firing B from scratch, firing items are: ', net.get_firing_items())
net.visualize_network(layout=layout)
net.run_firing_step()
print('after firing B from scratch, waiting 1 step, firing items are: ', net.get_firing_items())
net.visualize_network(layout=layout)
net.turn_off_all_firing()
net.link('A', 'B')
net.fire_item('A')
print('after linking A-B, firing A from scratch, firing items are: ', net.get_firing_items())
net.visualize_network(layout=layout)
net.run_firing_step()
print('after linking A-B, firing A from scratch, waiting 1 step, firing items are: ', net.get_firing_items())
net.visualize_network(layout=layout)
net.run_firing_step()
print('after linking A-B, firing A from scratch, waiting 2 steps, firing items are: ', net.get_firing_items())
net.visualize_network(layout=layout)
