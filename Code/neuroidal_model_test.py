import neuroidal_model

n = 30
d = 10
k = 2
r = 5
print("Creating neuroidal model for n, d, k, r = %d, %d, %d, %d" % (n, d, k, r))
net = neuroidal_model.NeuroidalNet(n,d,k,r)

print("\ncreating items")
net.create_item("A")
net.stored_items
net.create_item("B")
net.stored_items

print("\nfiring")
net.fire_item("A")
net.neuron_firings
net.neuron_memories
print(net.get_firing_items())
net.fire_item("B")
net.neuron_firings
net.neuron_memories
print(net.get_firing_items())
net.turn_off_all_firing()
net.neuron_firings
net.neuron_memories
print(net.get_firing_items())

print("\njoining item A and B")
net.join('A', 'B', 'C')
print("Firing item A and B")
net.fire_item("A")
print("Firing item B")
net.fire_item("B")
print("Firing items: ")
print(net.get_firing_items())
print("Running update step:")
print(net.run_firing_step())
print("Firing items: ")
print(net.get_firing_items())

net.reset_network(reset_synapse_strengths=True)
print("\nlinking")
net.turn_off_all_firing()
net.fire_item('A')
print('after firing A from scratch, firing items are: ', net.get_firing_items())
net.run_firing_step()
print('after firing A from scratch, waiting 1 step, firing items are: ', net.get_firing_items())
net.turn_off_all_firing()
net.fire_item('B')
print('after firing B from scratch, firing items are: ', net.get_firing_items())
net.run_firing_step()
print('after firing B from scratch, waiting 1 step, firing items are: ', net.get_firing_items())
net.turn_off_all_firing()
net.link('A', 'B')
net.fire_item('A')
print('after linking A-B, firing A from scratch, firing items are: ', net.get_firing_items())
net.run_firing_step()
print('after linking A-B, firing A from scratch, waiting 1 step, firing items are: ', net.get_firing_items())
