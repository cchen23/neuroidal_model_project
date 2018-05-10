import numpy as np
from enum import IntEnum

class Q(IntEnum):
    q1 = 0
    q2 = 1
    q3 = 2

class QQ(IntEnum):
    qq1 = 0
    qq2 = 1

class Firing(IntEnum):
    Off = 0
    On = 1

class NeuroidalNet:

    def __init__(self, n, d, k, r):
        """
        Args:
            n: number of neurons in network
            d: degree
            k: inverse synaptic strength
            r: number of neurons per item
        """
        self.THRESHOLD = 100
        self.num_neurons = n
        self.degree = d
        self.k = k
        self.r = r
        self.p = float(self.degree) / self.num_neurons

        self.stored_items = {}

        # Initialize neurons.
        self.neuron_firings = np.zeros(self.num_neurons)   # 0 if not firing, 1 if firing
        self.neuron_memories = np.zeros(self.num_neurons)

        # Initialize synapses.
        self.synapse_strengths = np.random.choice([0,1], (self.num_neurons, self.num_neurons), p=[1-self.p,self.p])
        self.synapse_memory_states = np.zeros([self.num_neurons, self.num_neurons])
        self.synapse_memory_values = np.empty([self.num_neurons, self.num_neurons])

    def create_item(self, item_name):
        """Randomly selects r neurons to represent an item."""
        self.stored_items[item_name] = np.random.choice(range(self.num_neurons), (self.r), replace=False)
        return

    """For JOIN"""
    def join(self, itemA_name, itemB_name, itemC_name):
        """Returns nodes with at least total synapse strength k to item nodes."""
        # Get potential JOIN nodes.
        itemA_neurons = self.stored_items[itemA_name]
        itemA_to_neuron_strengths = self.synapse_strengths[:,itemA_neurons]
        itemA_to_neuron_strengths_sums = np.sum(itemA_to_neuron_strengths, axis=1)
        potential_neuronsA = np.where(itemA_to_neuron_strengths_sums >= self.k)[0]

        # Set potential nodes and synapses.
        for potential_neuron in potential_neuronsA:
            self.neuron_memories[potential_neuron] = Q.q3
            x = np.count_nonzero(self.synapse_strengths[potential_neuron, itemA_neurons])
            for itemA_neuron in itemA_neurons:
                if self.synapse_strengths[potential_neuron, itemA_neuron] > 0:
                    self.synapse_memory_states[potential_neuron, itemA_neuron] = QQ.qq2
                    self.synapse_memory_values[potential_neuron, itemA_neuron] = self.THRESHOLD / float(x)

        # Get potential JOIN nodes.
        itemB_neurons = self.stored_items[itemB_name]
        itemB_to_neuron_strengths = self.synapse_strengths[:,itemB_neurons]
        itemB_to_neuron_strengths_sums = np.sum(itemB_to_neuron_strengths, axis=1)
        potential_neuronsB = np.where(itemB_to_neuron_strengths_sums >= self.k)[0]

        join_item_neurons = []

        for neuron in np.where(self.neuron_memories == Q.q3)[0]:
            if neuron in potential_neuronsB:
                print(neuron)
                y = np.count_nonzero(self.synapse_strengths[neuron, itemB_neurons])
                for source_neuron in range(self.num_neurons):
                    # Set synapses to item A
                    if self.synapse_memory_states[neuron, source_neuron] == QQ.qq2:
                        self.synapse_strengths[neuron, source_neuron] = self.synapse_memory_values[neuron, source_neuron]
                        self.synapse_memory_states[neuron, source_neuron] = QQ.qq1
                        self.neuron_memories[neuron] = Q.q2
                    # Set synapses to item B
                    elif source_neuron in itemB_neurons:
                        self.synapse_strengths[neuron, source_neuron] = self.THRESHOLD / float(y)
                        self.neuron_memories[neuron] = Q.q2
                    # Set other synapses.
                    else:
                        self.synapse_strengths[neuron, source_neuron] = 0
                join_item_neurons.append(neuron)
            else:
                self.neuron_memories[neuron] = Q.q1
                self.synapse_memory_states[neuron,:] = np.zeros([1,self.num_neurons])

        self.stored_items[itemC_name] = join_item_neurons
        return

    """For execution."""
    def fire_item(self, item_name):
        """Sets neurons associated with item to firing."""
        item_neurons = self.stored_items[item_name]
        for neuron in item_neurons:
            self.neuron_firings[neuron] = Firing.On
        return

    def run_firing_step(self):
        """Fires neurons where synapse-weighted sum of input firing is above threshold."""
        for neuron in range(self.num_neurons):
            weighted_input = sum(self.neuron_firings * self.synapse_strengths[neuron,:])
            if weighted_input > self.THRESHOLD:
                self.neuron_firings[neuron] = Firing.On

    def get_firing_items(self):
        """Returns a list of neuron firing."""
        firing_neurons = set(np.where(self.neuron_firings == Firing.On)[0])
        firing_items = []
        for item_name, item_neurons in self.stored_items.items():
            if len(firing_neurons.intersection(set(item_neurons))) > 0.5 * len(item_neurons): # Count as firing if > 50% of nodes firing.
                firing_items.append(item_name)
        return firing_items

    def turn_off_all_firing(self):
        """Sets all neurons to not firing."""
        self.neuron_firings = np.zeros(self.num_neurons)
