# Implementation of Algorithms on the Neuroidal Model
These files implement the two algorithms on the neuroidal model [1]. Our implementation of the neuroidal model and algorithms is primarily based on [1] and [2].

## Model
Our networks contain *n* total nodes, and when new items are created the network allocates *r* randomly selected nodes to represent the item. Each neuron fires if its incoming activity falls above some threshold *T*, and the maximum synaptic strength between any pair of neurons is <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7BT%7D%7Bk%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=arev&edit=0" align="center" border="0" alt="\frac{T}{k}" width="7" height="21" />. The network is initialized to connect each pair of neurons with some probability *p*.

## Execution
During execution, the network updates at discrete timesteps. At each timestep, neurons receive signals from incoming synapses. Neuron *i* receives signal <img src="http://latex.codecogs.com/gif.latex?\sum\limits_{i=1:n}w_{ji}f_i" title="\sum\limits_{i=1:n}w_{ji}f_i" align="center" border="0" width="50" height="30"/>, where *w<sub>ji</sub>* is the strength of the synapse from neuron *i* to neuron *j*, *n* is the total number of neurons in the network, and *f<sub>i</sub>* is an indicator variable for the firing of neuron *i*. Neuron *i* fires if and only if this signal falls above a threshold *T*.

## Learning
To learn the operation *C=JOIN(A,B)*, the network updates its synapses according to the following steps:
  1. Find the neurons that have sufficiently many connections from *A* and are disjoint with *A*. "Sufficiently many" means that the neuron has at least *k* connections to neurons in *A*, where <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7BT%7D%7Bk%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=arev&edit=0" align="center" border="0" alt="\frac{T}{k}" width="7" height="21" /> is the maximum synaptic strength. These are the potential neurons representing *C*.
  2. Out of these neurons, choose the ones that also have sufficiently many connections from *B* and are disjoint with *B*. These are the neurons that represent *C*.
  3. Update the synapses between *C* and the neurons in *A* and *B* as follows: for each neuron in *C* count the number *x* of synapses from *A* to that neuron, and set the strength of each of these synapses to <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7BT%7D%7Bx%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=arev&edit=0" align="center" border="0" alt="\frac{T}{k}" width="7" height="21" />. Do the same for the synapses from *B*.

To learn the operation *C=LINK(A,B)*, the network updates its synapses according to the following steps:
  1. Find the neurons that have sufficiently many connections from *A*, where "sufficiently many" means the same as in the *JOIN* algorithm. These are the potential relay neurons.
  2. Out of these neurons, choose the ones that have sufficiently many connections to *B*. These are the relay neurons.
  3. Update the synapses from *A* to each relay neuron and from each relay neuron to *B* as follows: for each relay neuron, count the number *x* of synapses from *A* to that neuron, and set the strength of each of these synapses to <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7BT%7D%7Bx%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=arev&edit=0" align="center" border="0" alt="\frac{T}{k}" width="7" height="21" />. Do the same for synapses to *B*. 

## Files
`neuroidal_model.py` contains modules for learning and executing JOIN and LINK operations, and `neuroidal_model_test.py` contains sample uses.

## References
1. L. G. Valiant, Circuits of the Mind. New York, NY, USA: Oxford University Press, Inc., 1994.
2. ----, "Memorization and association on a realistic neural model," Neural Computation,
vol. 17, no. 3, pp. 527{555, Mar. 2005.
