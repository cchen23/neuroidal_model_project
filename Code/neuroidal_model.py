####################################################################################################
# An implmenetation of Valiant's model of cortical learning, based on the description by
# Papadimitriuou and Vempala in their extension of it:
# http://proceedings.mlr.press/v40/Papadimitriou15.pdf
#
# May 2018
####################################################################################################

import numpy as np
from enum import IntEnum

class Q(IntEnum):
    """Neuron memories. """
    q1 = 0   # dismissed
    q2 = 1   # candidate
    q3 = 2   # operational

class QQ(IntEnum):
    """Synapse memories. """
    qq1 = 0   # temporarily connected to B?
    qq2 = 1   # temporarily connected to A?

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
        self.THRESHOLD = 100.0
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
        self.synapse_strengths = np.random.choice([0,1], (self.num_neurons, self.num_neurons),
                                                  p=[1-self.p,self.p])
        self.synapse_memory_states = np.zeros([self.num_neurons, self.num_neurons])
        self.synapse_memory_values = np.empty([self.num_neurons, self.num_neurons])

    def create_item(self, item_name):
        """Randomly selects r neurons to represent an item."""
        self.stored_items[item_name] = np.random.choice(range(self.num_neurons), (self.r),
                                                        replace=False)
        return


    """Learning memory operations."""

    def join(self, itemA_name, itemB_name, itemC_name):
        """Returns nodes with at least total synapse strength k to item nodes."""
        # Get potential JOIN nodes.
        itemA_neurons = self.stored_items[itemA_name]
        itemA_to_neuron_strengths = self.synapse_strengths[:,itemA_neurons]
        num_itemA_to_neuron_connections = np.count_nonzero(itemA_to_neuron_strengths, axis=1)
        potential_neuronsA = np.where(num_itemA_to_neuron_connections >= self.k)[0] # C neurons need at least k connections to A.

        # Set potential nodes and synapses.
        for potential_neuron in potential_neuronsA:
            self.neuron_memories[potential_neuron] = Q.q3
            x = np.count_nonzero(self.synapse_strengths[potential_neuron, itemA_neurons])
            if x == 0:
                print("JOIN failed. Insufficient neurons with strong enough connections to A.")
                self.reset_network()
                return
            for itemA_neuron in itemA_neurons:
                if self.synapse_strengths[potential_neuron, itemA_neuron] > 0:
                    self.synapse_memory_states[potential_neuron, itemA_neuron] = QQ.qq2
                    self.synapse_memory_values[potential_neuron, itemA_neuron] = \
                        self.THRESHOLD / float(x)

        # Get potential JOIN nodes.
        itemB_neurons = self.stored_items[itemB_name]
        itemB_to_neuron_strengths = self.synapse_strengths[:,itemB_neurons]
        num_itemB_to_neuron_connections = np.count_nonzero(itemB_to_neuron_strengths, axis=1)
        potential_neuronsB = np.where(num_itemB_to_neuron_connections >= self.k)[0]  # C neurons need at least k connections to B.

        join_item_neurons = []

        for neuron in np.where(self.neuron_memories == Q.q3)[0]:
            if neuron in potential_neuronsB:
                y = np.count_nonzero(self.synapse_strengths[neuron, itemB_neurons])
                if y == 0:
                    print("JOIN failed. Insufficient neurons with strong enough connections to B.")
                    self.reset_network()
                    return
                for source_neuron in range(self.num_neurons):
                    # Set synapses to item A
                    if self.synapse_memory_states[neuron, source_neuron] == QQ.qq2:
                        self.synapse_strengths[neuron, source_neuron] = \
                            self.synapse_memory_values[neuron, source_neuron]
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

    def link(self, itemA_name, itemB_name):
        """Updates some weights within item B neurons to make it so that any time A's neurons fire,
           all of B's neuron's will also fire. """

        itemB_neurons = self.stored_items[itemB_name]
        itemA_neurons = self.stored_items[itemA_name]
        itemA_to_neuron_strengths = self.synapse_strengths[:,itemA_neurons]
        itemA_to_neuron_strengths_sums = np.sum(itemA_to_neuron_strengths, axis=1)
        neurons_activated_by_A = np.where(itemA_to_neuron_strengths_sums >= self.THRESHOLD)[0]

        neurons_activated_by_A_to_B_strengths = self.synapse_strengths[:,neurons_activated_by_A]
        neurons_activated_by_A_to_B_strengths_sums = \
            np.sum(neurons_activated_by_A_to_B_strengths, axis=1)
        relay_neurons = np.where(neurons_activated_by_A_to_B_strengths_sums >= self.THRESHOLD)[0]
        n_relay_neurons = len(relay_neurons)

        if n_relay_neurons != 0:   # normal case described by Valiant
            for neuron in itemB_neurons:
                if neuron not in relay_neurons:
                    self.synapse_strengths[neuron, relay_neurons] = self.THRESHOLD / n_relay_neurons
        else:
            print("LINK failed. Insufficient relay neurons.")
            self.reset_network()
            return

        return


    """Executing Memory Operations."""

    def fire_item(self, item_name):
        """Sets neurons associated with item to firing."""
        item_neurons = self.stored_items[item_name]
        for neuron in item_neurons:
            self.neuron_firings[neuron] = Firing.On
        return

    def run_firing_step(self):
        """Fires neurons where synapse-weighted sum of input firing is above threshold."""
        new_neuron_firings = np.zeros_like(self.neuron_firings)
        for neuron in range(self.num_neurons):
            weighted_input = sum(self.neuron_firings * self.synapse_strengths[neuron,:])
            if weighted_input >= self.THRESHOLD:
                new_neuron_firings[neuron] = Firing.On
        self.neuron_firings = new_neuron_firings

    def get_firing_items(self):
        """Returns a list of neuron firing."""
        firing_neurons = set(np.where(self.neuron_firings == Firing.On)[0])
        firing_items = []
        for item_name, item_neurons in self.stored_items.items():
            # Count as firing if > 50% of nodes firing.
            if len(firing_neurons.intersection(set(item_neurons))) > 0.5 * len(item_neurons):
                firing_items.append(item_name)
        return firing_items

    def turn_off_all_firing(self):
        """Sets all neurons to not firing."""
        self.neuron_firings = np.zeros(self.num_neurons)

    def reset_network(self, reset_synapse_strengths=False):
        """Resets the network. If reset_synapse_strengths=False, maintains previously learned connections."""
        self.neuron_firings = np.zeros(self.num_neurons)
        self.neuron_memories = np.zeros(self.num_neurons)
        self.synapse_memory_states = np.zeros([self.num_neurons, self.num_neurons])
        self.synapse_memory_values = np.empty([self.num_neurons, self.num_neurons])
        if reset_synapse_strengths:
            self.synapse_strengths = np.random.choice([0,1], (self.num_neurons, self.num_neurons),
            p=[1-self.p,self.p])
