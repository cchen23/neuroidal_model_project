####################################################################################################
# An implmenetation of Valiant's model of cortical learning, based on the description by
# Papadimitriuou and Vempala in their extension of it:
# http://proceedings.mlr.press/v40/Papadimitriou15.pdf
#
# May 2018
####################################################################################################
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
import networkx as nx
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
        self.EPSILON = self.THRESHOLD / 10**10
        self.num_neurons = n
        self.degree = d
        self.k = k
        self.r = r
        self.p = float(self.degree) / self.num_neurons

        self.stored_items = {}

        # Initialize neurons.
        self.neuron_firings = np.zeros(self.num_neurons)   # 0 if not firing, 1 if firing.
        self.neuron_memories = np.zeros(self.num_neurons)

        # Initialize synapses.
        self.synapse_strengths = np.random.choice([0.0, 1.0], (self.num_neurons, self.num_neurons),
                                                  p=[1-self.p,self.p])
        np.fill_diagonal(self.synapse_strengths, 0) # Initialize without self loops.
        self.synapse_memory_states = np.zeros([self.num_neurons, self.num_neurons])
        self.synapse_memory_values = np.empty([self.num_neurons, self.num_neurons])

    def create_item(self, item_name, disjoint=False):
        """Randomly selects r neurons to represent an item."""
        if disjoint:
            neuron_choices = set(range(self.num_neurons))
            for item, neurons in self.stored_items.items():
                neuron_choices.difference_update(set(neurons))
            self.stored_items[item_name] = np.random.choice(list(neuron_choices), self.r, replace=False)
        else:
            self.stored_items[item_name] = np.random.choice(range(self.num_neurons), (self.r),
            replace=False)
        return


    """Learning memory operations."""
    def get_potentialneurons(self, item_neurons):
        """Returns indices of neurons with at least k connections to item_neurons.
        Does not include neurons in item_neurons.
        """
        item_to_neuron_strengths = self.synapse_strengths[:,item_neurons]
        num_item_to_neuron_connections = np.count_nonzero(item_to_neuron_strengths, axis=1)
        potential_neurons = set(np.where(num_item_to_neuron_connections >= self.k)[0]).difference(set(item_neurons))
        return potential_neurons

    def join(self, itemA_name, itemB_name, itemC_name):
        """Returns nodes with at least total synapse strength k to item nodes."""
        # Get potential JOIN nodes.
        itemA_neurons = self.stored_items[itemA_name]
        potential_neuronsA = self.get_potentialneurons(itemA_neurons)

        # Set potential nodes and synapses.
        if len(potential_neuronsA) == 0:
            print("JOIN failed. Insufficient neurons with strong enough connections to A.")
            self.reset_network()
            return
        for potential_neuron in potential_neuronsA:
            self.neuron_memories[potential_neuron] = Q.q3
            x = np.count_nonzero(self.synapse_strengths[potential_neuron, itemA_neurons])
            for itemA_neuron in itemA_neurons:
                if self.synapse_strengths[potential_neuron, itemA_neuron] > 0:
                    self.synapse_memory_states[potential_neuron, itemA_neuron] = QQ.qq2
                    self.synapse_memory_values[potential_neuron, itemA_neuron] = \
                        self.THRESHOLD / (2 * float(x))

        # Get potential JOIN nodes.
        itemB_neurons = self.stored_items[itemB_name]
        potential_neuronsB = self.get_potentialneurons(itemB_neurons)
        join_item_neurons = []

        if len(potential_neuronsB) == 0:
            print("JOIN failed. Insufficient neurons with strong enough connections to B.")
            self.reset_network()
            return
        for neuron in np.where(self.neuron_memories == Q.q3)[0]:
            if neuron in potential_neuronsB:
                y = np.count_nonzero(self.synapse_strengths[neuron, itemB_neurons])
                for source_neuron in range(self.num_neurons):
                    # Set synapses to item A
                    if self.synapse_memory_states[neuron, source_neuron] == QQ.qq2:
                        self.synapse_strengths[neuron, source_neuron] = \
                            self.synapse_memory_values[neuron, source_neuron]
                        self.synapse_memory_states[neuron, source_neuron] = QQ.qq1
                        self.neuron_memories[neuron] = Q.q2
                    # Set synapses to item B
                    elif source_neuron in itemB_neurons:
                        self.synapse_strengths[neuron, source_neuron] = self.THRESHOLD / (2 * float(y))
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
        potential_relay_neurons = self.get_potentialneurons(itemA_neurons)
        neurons_connectedtoB = set(np.where(np.count_nonzero(self.synapse_strengths[itemB_neurons,:], axis=0) > 0)[0]).difference(set(itemB_neurons)) # Relay neurons must be connected to B, and must be disjoint from B.
        relay_neurons = potential_relay_neurons.intersection(neurons_connectedtoB)
        n_relay_neurons = len(relay_neurons)

        if n_relay_neurons < self.k:   # Need at least k relay neurons.
            print("LINK failed. Insufficient relay neurons.")
            self.reset_network()
            return
        else:
            for neuron in relay_neurons:
                connected_neuronsA = list(set(np.where(self.synapse_strengths[neuron,:] > 0)[0]).intersection(set(itemA_neurons)))
                self.synapse_strengths[neuron, connected_neuronsA] = self.THRESHOLD / float(len(connected_neuronsA))
            for neuron in itemB_neurons:
                connected_neuronsrelay = list(set(np.where(self.synapse_strengths[neuron,:] > 0)[0]).intersection(set(relay_neurons)))
                num_connectedneurons = len(connected_neuronsrelay)
                if num_connectedneurons > 0:
                    self.synapse_strengths[neuron, connected_neuronsrelay] = self.THRESHOLD / float(num_connectedneurons)
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
            if weighted_input >= self.THRESHOLD - self.EPSILON:
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

    def get_firing_neurons(self):
        """Returns a list of neuron firing."""
        firing_neurons = set(np.where(self.neuron_firings == Firing.On)[0])
        return firing_neurons

    def turn_off_all_firing(self):
        """Sets all neurons to not firing."""
        self.neuron_firings = np.zeros(self.num_neurons)

    def reset_network(self, reset_synapse_strengths=False, reset_items=False):
        """Resets the network. If reset_synapse_strengths=False, maintains previously learned connections."""
        self.neuron_firings = np.zeros(self.num_neurons)
        self.neuron_memories = np.zeros(self.num_neurons)
        self.synapse_memory_states = np.zeros([self.num_neurons, self.num_neurons])
        self.synapse_memory_values = np.empty([self.num_neurons, self.num_neurons])
        if reset_synapse_strengths:
            self.synapse_strengths = np.random.choice([0,1], (self.num_neurons, self.num_neurons),
            p=[1-self.p,self.p]).astype(float)
        if reset_items:
            self.stored_items = {}

    """ Visualizing Network. """
    def visualize_network(self, color_by="firing", layout=None, savename=None):
        G = nx.from_numpy_matrix(self.synapse_strengths)

        # Label and color nodes.
        labels = {i:"" for i in range(self.num_neurons)}
        neuron_colors = ["grey" for i in range(self.num_neurons)]
        color_options = ["sandybrown", "darkcyan", "palevioletred", "mediumseagreen", "lightsteelblue", "teal"]
        num_color_options = len(color_options)
        num_items = len(self.stored_items.keys())
        colors_index = 0

        for item_name, item_neurons in self.stored_items.items():
            for neuron in item_neurons:
                labels[neuron] = item_name
                neuron_colors[neuron] = color_options[colors_index % num_color_options]
            colors_index += 1
        neuron_colors = np.array(neuron_colors)

        # Color edges according to weights.
        edgelist = G.edges(data=True)
        edge_cmap = plt.cm.Greys
        edge_vmax = self.THRESHOLD / self.k / 10
        edge_weights = [edge[2]['weight'] for edge in edgelist]
        if not layout:
            layout = nx.random_layout(G)
        on_neurons = [neuron for neuron in range(self.num_neurons) if self.neuron_firings[neuron] == Firing.On]
        off_neurons = [neuron for neuron in range(self.num_neurons) if self.neuron_firings[neuron] == Firing.Off]
        on_nodes = nx.draw_networkx_nodes(G, pos=layout, nodelist=on_neurons, labels=labels, node_color=neuron_colors[on_neurons], linewidths=2)
        off_nodes = nx.draw_networkx_nodes(G, pos=layout, nodelist=off_neurons, labels=labels, node_color=neuron_colors[off_neurons], linewidths=2)
        if on_nodes:
            on_nodes.set_edgecolor("yellow")
        nx.draw_networkx_labels(G, pos=layout, labels=labels)
        nx.draw_networkx_edges(G, pos=layout, edge_cmap=edge_cmap, edge_color=edge_weights, edge_vmin=0, edge_vmax=edge_vmax)
        plt.axis('off')
        if savename:
            plt.savefig(savename)
        plt.show()
