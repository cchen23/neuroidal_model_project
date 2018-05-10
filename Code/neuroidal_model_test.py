import neuroidal_model

n = 30
d = 10
k = 2
r = 5
net = neuroidal_model.NeuroidalNet(n,d,k,r)

net.create_item("A")
net.stored_items
net.create_item("B")
net.stored_items

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

net.join('A', 'B', 'C')
net.fire_item("A")
net.fire_item("B")
net.get_firing_items()
net.run_firing_step()
