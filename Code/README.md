# Implementation of Algorithms on the Neuroidal Model
These files implement the two algorithms on the neuroidal model [1]. Our implementation of the neuroidal model and algorithms is primarily based on [1] and [2].
To learn the operation C=JOIN(A,B), the network updates its synapses according to the following steps:
  1. Find the neurons that have sufficiently many connections from A and are disjoint with A. "Sufficiently many" means that the neuron has at least k connections to neurons in A, where 1/k is the maximum synaptic strength. These are the potential C neurons, and there is some number x such neurons.
  2. Out of these neurons, choose the ones that also have sufficiently many connections from B and are disjoint with B. These are the neurons that represent C.
  3. Update the synapses between C and the neurons in A and B.

To learn the operation C=LINK(A,B), the network updates its synapses according to the following steps:
  1. Find the neurons that have sufficiently many connections from A.
  2. Out of these neurons, choose the ones that have sufficiently many connections to B. These are the relay neurons.
  3. Update the synapses from A to each relay neuron and from each relay neuron to B.

`neuroidal_model.py` contains modules for learning and executing JOIN and LINK operations, and `neuroidal_model_test.py` contains sample uses.

### References
1. L. G. Valiant, Circuits of the Mind. New York, NY, USA: Oxford University Press, Inc., 1994.
2. ----, "Memorization and association on a realistic neural model," Neural Computation,
vol. 17, no. 3, pp. 527{555, Mar. 2005.
