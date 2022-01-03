#!/bin/python
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.random import RandomDistribution, NumpyRNG
from pyNN.utility.plotting import Figure, Panel
from quantities import ms
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import neo
import time

### SIMULATOR CONFIGURATION ###

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--STDP", "Run network using the STDP implemented in PyNN", {"action": "store_true"}),
                             ("--dendritic-delay-fraction", "What fraction of the total transmission delay is due to dendritic propagation", {"default": 1}),
                             ("--debug", "Print debugging information"))


if options.debug:
    init_logging(None, debug=True)

if sim == "nest":
    from pyNN.nest import *

dt = 1/80000
sim.setup(timestep=0.01)

### INPUT DATA ###
time_data = 100000
interval = 100
duration = 5
num = time_data//interval
# The input should be at least 13*13 for a duration of 5 since we want to leave a marging of 4 neurons on the egdges when generating data
x_input = 13 
y_input = 13
filter_x = 5
filter_y = 5

# Dataset Generation
input_spiketrain = [[] for i in range(x_input*y_input)]
for t in range(time_data//interval):
	direction = randint(0,1)
	if direction==0:
		startx = randint(4, x_input-5-4)
		starty = randint(4, y_input-5-4)
		path = [[startx + i, starty + i] for i in range(duration)]
		pathid = [cell[0]+cell[1]*x_input for cell in path]
		for cell in range(len(pathid)):
			input_spiketrain[pathid[cell]].append(cell+t*interval+2)
	if direction==1:
		startx = randint(8, x_input-5)
		starty = randint(8, y_input-5)
		path = [[startx - i, starty - i] for i in range(duration)]
		pathid = [cell[0]+cell[1]*x_input for cell in path]
		for cell in range(len(pathid)):
			input_spiketrain[pathid[cell]].append(cell+t*interval+2)
	if direction==2:
		startx = randint(8, x_input-5)
		starty = randint(4, y_input-5-4)
		path = [[startx - i, starty + i] for i in range(duration)]
		pathid = [cell[0]+cell[1]*x_input for cell in path]
		for cell in range(len(pathid)):
			input_spiketrain[pathid[cell]].append(cell+t*interval+2)
	if direction==3:
		startx = randint(4, x_input-5-4)
		starty = randint(8, y_input-5)
		path = [[startx + i, starty - i] for i in range(duration)]
		pathid = [cell[0]+cell[1]*x_input for cell in path]
		for cell in range(len(pathid)):
			input_spiketrain[pathid[cell]].append(cell+t*interval+2)
ev = input_spiketrain


### NETWORK ###

# populations
Input = sim.Population(
    x_input*y_input, # le nombre d'entrées 
    sim.SpikeSourceArray(spike_times=ev), #on passe le data input ici
    label="Input"
)
Input.record("spikes") 

Convolutions_parameters = {
    'tau_m': 20.0,      # membrane time constant (in ms)   
    'tau_refrac': 30.0,  # duration of refractory period (in ms) 0.1 de base
    'v_reset': -70.0,   # reset potential after a spike (in mV) 
    'v_rest': -70.0,   # resting membrane potential (in mV)
    'v_thresh': -5.0,     # spike threshold (in mV) -5 de base
}

# The size of a convolution layer with a filter of size x*y is input_x-x+1 * input_y-y+1 
convolution1 = sim.Population(
  (x_input-filter_x+1)*(y_input-filter_y+1), 
  sim.IF_cond_exp(**Convolutions_parameters), 
)
convolution2 = sim.Population(
  (x_input-filter_x+1)*(y_input-filter_y+1),
  sim.IF_cond_exp(**Convolutions_parameters), 
)
convolution3 = sim.Population(
  (x_input-filter_x+1)*(y_input-filter_y+1),
  sim.IF_cond_exp(**Convolutions_parameters), 
)
convolution4 = sim.Population(
  (x_input-filter_x+1)*(y_input-filter_y+1),
  sim.IF_cond_exp(**Convolutions_parameters), 
)


convolution1.record(("spikes","v"))
convolution2.record(("spikes","v"))
convolution3.record(("spikes","v"))
convolution4.record(("spikes","v"))


# filters
weight_N = 0.35 
delays_N = 15.0 
weight_teta = 0.005 
delays_teta = 0.05 

# We initialize our filters 
filter1_w = np.array([[np.random.normal(weight_N,weight_teta) for j in range(filter_y)] for i in range(filter_x)])
filter2_w = np.array([[np.random.normal(weight_N,weight_teta) for j in range(filter_y)] for i in range(filter_x)])
filter3_w = np.array([[np.random.normal(weight_N,weight_teta) for j in range(filter_y)] for i in range(filter_x)])
filter4_w = np.array([[np.random.normal(weight_N,weight_teta) for j in range(filter_y)] for i in range(filter_x)])
filter1_d = np.array([[np.random.normal(delays_N,delays_teta) for j in range(filter_y)] for i in range(filter_x)])
filter2_d = np.array([[np.random.normal(delays_N,delays_teta) for j in range(filter_y)] for i in range(filter_x)])
filter3_d = np.array([[np.random.normal(delays_N,delays_teta) for j in range(filter_y)] for i in range(filter_x)])
filter4_d = np.array([[np.random.normal(delays_N,delays_teta) for j in range(filter_y)] for i in range(filter_x)])

# connections
# The weight and delays will be set based on the values of the filters created just before
syn_type = sim.StaticSynapse(weight=0, delay=50)
connections_proj1 = []
connections_proj2 = []
connections_proj3 = []
connections_proj4 = []


# we *sweep* the input layer with a 5*5 window
# window_x, window_y are the upper-left coordinates of our window
for window_x in range(0, x_input - (filter_x-1)):
  for window_y in range(0, y_input - (filter_y-1)):
    # we look at each input neuron in our window and create a connection
    # with the according neuron in each convolutionnal layer.
    
    for x in range(filter_x):
      for y in range(filter_y):
        input_neuron_id = window_x+x + (window_y+y)*x_input
        convo_neuron_id = window_x + window_y*(x_input-filter_x+1)
        # The connections we are creating are encoded like : [pre_synaptic_cell, post_synaptic_cell, weight, delay]
        connections_proj1.append( (input_neuron_id, convo_neuron_id, filter1_w[x][y], filter1_d[x][y]) )
        connections_proj2.append( (input_neuron_id, convo_neuron_id, filter2_w[x][y], filter2_d[x][y]) )
        connections_proj3.append( (input_neuron_id, convo_neuron_id, filter3_w[x][y], filter3_d[x][y]) )
        connections_proj4.append( (input_neuron_id, convo_neuron_id, filter4_w[x][y], filter4_d[x][y]) )


input2convolution1 = sim.Projection(
    Input, convolution1,
    connector=sim.FromListConnector(connections_proj1, column_names=["weight", "delay"]), 
    synapse_type=syn_type,
    receptor_type="excitatory",
    label="Connection input layer to convolutional layer 1"
)
input2convolution2 = sim.Projection(
    Input, convolution2,
    connector=sim.FromListConnector(connections_proj2, column_names=["weight", "delay"]), 
    synapse_type=syn_type,
    receptor_type="excitatory",
    label="Connection input layer to convolutional layer 2"
)
input2convolution3 = sim.Projection(
    Input, convolution3,
    connector=sim.FromListConnector(connections_proj3, column_names=["weight", "delay"]), 
    synapse_type=syn_type,
    receptor_type="excitatory",
    label="Connection input layer to convolutional layer 3"
)
input2convolution4 = sim.Projection(
    Input, convolution4,
    connector=sim.FromListConnector(connections_proj4, column_names=["weight", "delay"]), 
    synapse_type=syn_type,
    receptor_type="excitatory",
    label="Connection input layer to convolutional layer 4"
)


# Latheral Inhibition (WTA)
latheral_w = 50.0
latheralInhibition1_2 = sim.Projection(
  convolution1, convolution2,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 1 and 2"
)
latheralInhibition1_3 = sim.Projection(
  convolution1, convolution3,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 1 and 3"
)
latheralInhibition1_4 = sim.Projection(
  convolution1, convolution4,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 1 and 4"
)

latheralInhibition2_1 = sim.Projection(
  convolution2, convolution1,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 2 and 1"
)
latheralInhibition2_3 = sim.Projection(
  convolution2, convolution3,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 2 and 3"
)
latheralInhibition2_4 = sim.Projection(
  convolution2, convolution4,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 2 and 4"
)

latheralInhibition3_1 = sim.Projection(
  convolution3, convolution1,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 3 and 1"
)
latheralInhibition3_2 = sim.Projection(
  convolution3, convolution2,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 3 and 2"
)
latheralInhibition3_4 = sim.Projection(
  convolution3, convolution4,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 3 and 4"
)

latheralInhibition4_1 = sim.Projection(
  convolution4, convolution1,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 4 and 1"
)
latheralInhibition4_2 = sim.Projection(
  convolution4, convolution2,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 4 and 2"
)
latheralInhibition4_3 = sim.Projection(
  convolution4, convolution3,
  connector = sim.OneToOneConnector(),
  synapse_type=sim.StaticSynapse(weight=latheral_w, delay=0.01), 
  receptor_type="inhibitory",
  label="Latheral inhibition between convolutional layers 4 and 3"
)


# We will use this list to know wich convolution layer has reached its stop condition
full_stop_condition= [False, False, False, False]
# Each filter of each convolution layer will be put in this list and actualized at each stimulus
final_filters = [[], [], [], []]
# Sometimes, even with latheral inhibition, two neurons on the same location in different convolution 
# layer will both spike (due to the minimum delay on those connections). So we keep track of
# wich neurons in each layer has already spiked this stimulus. (Everything is put back to False at the end of the stimulus)
neuron_activity_tag = [ [False for cell in range((x_input-filter_x+1)*(y_input-filter_y+1))]for conv in range(len(full_stop_condition)) ]


### Run simulation
"""
*************************************************************************************************
From example "simple_STDP.py" on : http://neuralensemble.org/docs/PyNN/examples/simple_STDP.html
"""

class SimControl(object):
	"""
	Prints the step of the simulation and the filters at the end of each stimulus.
	Also checks if the stop condition of the simulation is reached.
	"""

	def __init__(self, sampling_interval, projection, print_time = False):
		self.interval = sampling_interval
		self.projection = projection
		self._weights = []
		self._delays = []
		self.print_time =print_time

	def __call__(self, t):
		if self.print_time:
			print("step : {}".format(t))
			if full_stop_condition[0] and full_stop_condition[1] and full_stop_condition[2] and full_stop_condition[3]:
				print("!!!! FINISHED LEARNING !!!!") 
				sim.end()
				self.print_final_filters()
				exit()
			if t>1 and int(t)%interval==0:
				self.print_final_filters()

		return t + self.interval


	def print_final_filters(self):
		filter1_d, filter1_w= final_filters[0][0], final_filters[0][1] 
		filter2_d, filter2_w= final_filters[1][0], final_filters[1][1] 
		filter3_d, filter3_w= final_filters[2][0], final_filters[2][1] 
		filter4_d, filter4_w= final_filters[3][0], final_filters[3][1] 

		print("Filtre 0 délais:")
		for x in filter1_d:
			for y in x:
				print("{}, ".format(y*ms), end='')
			print()
		print("Filtre 0 poids:")
		for x in filter1_w:
			for y in x:
				print("{}, ".format(y), end='')
			print()

		print("\n\n")
		print("Filtre 1 délais:")
		for x in filter2_d:
			for y in x:
				print("{}, ".format(y*ms), end='')
			print()
		print("Filtre 1 poids:")
		for x in filter2_w:
			for y in x:
				print("{}, ".format(y), end='')
			print()

		print("\n\n")
		print("Filtre 2 délais:")
		for x in filter3_d:
			for y in x:
				print("{}, ".format(y*ms), end='')
			print()
		print("Filtre 2 poids:")
		for x in filter3_w:
			for y in x:
				print("{}, ".format(y), end='')
			print()

		print("\n\n")
		print("Filtre 3 délais:")
		for x in filter4_d:
			for y in x:
				print("{}, ".format(y*ms), end='')
			print()
		print("Filtre 3 poids:")
		for x in filter4_w:
			for y in x:
				print("{}, ".format(y), end='')
			print()



class NeuronReset(object):
	"""	
	Resets neuron_activity_tag to False for all neurons in all layers.
	Also injects a negative amplitude pulse to all neurons at the end of each stimulus
	So that all membrane potentials are back to their resting values.
	"""

	def __init__(self, sampling_interval, pops):
		self.interval = sampling_interval
		self.populations = pops 

	def __call__(self, t):
		for conv in neuron_activity_tag:
			for cell in range(len(conv)):
				conv[cell] = False

		if t > 0:
			print("!!! RESET !!!")
			if type(self.populations)==list:
				for pop in self.populations:
					pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+10)
					pulse.inject_into(pop)
			else:
				pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+10)
				pulse.inject_into(self.populations)

			self.interval = interval
		return t + self.interval


class InputClear(object):
	"""
	When called, simply gets the data from the input with the 'clear' parameter set to True.
	By periodically clearing the data from the populations the simulation goes a lot faster.
	"""

	def __init__(self, sampling_interval, pops_to_clear_data):
		self.interval = sampling_interval
		self.pop_clear = pops_to_clear_data

	def __call__(self, t):
		if t > 0:
			print("!!! INPUT CLEAR !!!")
			try:
				input_spike_train = self.pop_clear.get_data("spikes", clear=True).segments[0].spiketrains 
			except:
				pass
			self.interval = interval
		return t + self.interval


class LearningMecanisms(object):
	def __init__(self, sampling_interval, proj, input_pop, output_pop, B_plus, B_minus, tau_plus, tau_minus, filter_d, A_plus, A_minus, teta_plus, teta_minus, filter_w, stop_condition, growth_factor, Rtarget=0.005, lamdad=0.002, lamdaw=0.00005, thresh_adapt=True, label=0):  #Rt= 0.01 lamd = 0.01 lamw=0.001
		self.interval = sampling_interval
		self.projection = proj
		self.input = input_pop
		self.output = output_pop
		self.input_last_spiking_times = [-1 for n in range(len(self.input))] # For ech neuron we keep its last time of spike
		self.output_last_spiking_times = [-1 for n in range(len(self.output))]
		self.B_plus = B_plus
		self.B_minus = B_minus
		self.tau_plus = tau_plus
		self.tau_minus = tau_minus
		self.max_delay = False # If set to False, we will find the maximum delay on first call.
		self.filter_d = filter_d
		self.filter_w = filter_w
		self.A_plus = A_plus
		self.A_minus = A_minus
		self.teta_plus = teta_plus
		self.teta_minus = teta_minus
		self.c = stop_condition
		self.growth_factor = growth_factor
		self.label = label
		self.thresh_adapt=thresh_adapt
		self.total_spike_count_per_neuron = [0 for cell in range(len(self.output))] # For each neuron, we count their number of spikes to compute their activation rate.
		self.call_count = 0 # Number of times this has been called.
		self.Rtarget = Rtarget
		self.lamdaw = lamdaw 
		self.lamdad = lamdad

	def __call__(self, t):
		self.call_count += 1
		final_filters[self.label] = [self.filter_d, self.filter_w]

		# We get the spiking times
		try:
			# We don't clear the data from input since the other layers will also need it. The NeuronReset
			# object will be called at the next timestep to clear the input data.
			input_spike_train = self.input.get_data("spikes").segments[0].spiketrains 
			output_spike_train = self.output.get_data("spikes", clear=True).segments[0].spiketrains
		except:
			print("No data")
			return t + self.interval

		# We get the current delays and current weights
		delays = self.projection.get("delay", format="array")
		weights = self.projection.get("weight", format="array")

		# The sum of all homeostasis delta_d and delta_t computed for each cell
		homeo_delays_total = 0
		homeo_weights_total = 0

		# Since we can't increase the delays past the maximum delay set at the beggining of the simulation,
		# we find the maximum delay during the first call
		if self.max_delay == False:
			self.max_delay = 0.01
			for x in delays:
				for y in x:
					if not np.isnan(y) and y > self.max_delay:
						self.max_delay = y

		for pre_neuron in range(len(input_spike_train)):
			if len(input_spike_train[pre_neuron]) > 0 and input_spike_train[pre_neuron][-1] > self.input_last_spiking_times[pre_neuron]:
				# We actualize the last time of spike for this neuron
				self.input_last_spiking_times[pre_neuron] = input_spike_train[pre_neuron][-1]
				print("PRE SPIKE {} : {}".format(pre_neuron, input_spike_train[pre_neuron][-1]))


		for post_neuron in range(len(output_spike_train)):

			if len(output_spike_train[post_neuron]) > 0 and self.check_activity_tags(post_neuron):
				neuron_activity_tag[self.label][post_neuron] = True
				print("***** STIMULUS {} *****".format(t//interval))

				self.total_spike_count_per_neuron[post_neuron] += 1

				# The neuron spiked during this stimulus and its threshold should be increased.
				# Since Nest Won't allow nerons with a treshold > 0 to spike, we decrease v_rest instead.
				current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
				if self.thresh_adapt:
					self.output.__getitem__(post_neuron).v_rest=current_rest-(1.0-self.Rtarget)
					self.output.__getitem__(post_neuron).v_reset=current_rest-(1.0-self.Rtarget)
				print("=== Neuron {} from layer {} spiked ! Whith rest = {} ===".format(post_neuron, self.label, current_rest))
				print("Total pikes of neuron {} from layer {} : {}".format(post_neuron, self.label, self.total_spike_count_per_neuron[post_neuron]))

				if output_spike_train[post_neuron][-1] > self.output_last_spiking_times[post_neuron] and not self.stop_condition(delays, post_neuron):
					# We actualize the last time of spike for this neuron
					self.output_last_spiking_times[post_neuron] = output_spike_train[post_neuron][-1]

					# We now compute a new delay for each of its connections using STDP
					for pre_neuron in range(len(delays)):
	
						# For each post synaptic neuron that has a connection with pre_neuron, we also check that both neurons
						# already spiked at least once.
						if not np.isnan(delays[pre_neuron][post_neuron]) and not np.isnan(weights[pre_neuron][post_neuron]) and self.input_last_spiking_times[pre_neuron] != -1 and self.output_last_spiking_times[post_neuron] != -1:

							# Some values here have a dimension in ms
							delta_t = self.output_last_spiking_times[post_neuron]/ms - self.input_last_spiking_times[pre_neuron]/ms - delays[pre_neuron][post_neuron]
							delta_d = self.G(delta_t)
							delta_w = self.F(delta_t)

							print("STDP from layer: {} with post_neuron: {} and pre_neuron: {} deltad: {}, deltat: {}".format(self.label, post_neuron, pre_neuron, delta_d*ms, delta_t*ms))
							print("TIME PRE {} : {} TIME POST 0: {} DELAY: {}".format(pre_neuron, self.input_last_spiking_times[pre_neuron], self.output_last_spiking_times[post_neuron], delays[pre_neuron][post_neuron]))
							self.actualize_filter(pre_neuron, post_neuron, delta_d, delta_w, delays, weights)
			else:
				# The neuron did not spike and its threshold should be lowered

				if self.thresh_adapt:
					current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
					self.output.__getitem__(post_neuron).v_rest=current_rest+self.Rtarget
					self.output.__getitem__(post_neuron).v_reset=current_rest+self.Rtarget

			# Homeostasis regulation per neuron
			Robserved = self.total_spike_count_per_neuron[post_neuron]/self.call_count
			K = (self.Rtarget - Robserved)/self.Rtarget
			delta_d = -self.lamdad*K
			delta_w = self.lamdaw*K
			homeo_delays_total += delta_d  
			homeo_weights_total += delta_w 
			print("Rate of neuron {} from layer {}: {}".format(post_neuron, self.label, Robserved))


		print("****** CONVO {} homeo_delays_total: {}, homeo_weights_total: {}".format(self.label, homeo_delays_total, homeo_weights_total))
		self.actualize_All_Filter( homeo_delays_total+self.growth_factor*self.interval, homeo_weights_total, delays, weights)

		# At last we give the new delays and weights to our projections
		self.projection.set(delay = delays)
		self.projection.set(weight = weights)

		# We update the list that tells if this layer has finished learning the delays and weights
		full_stop_condition[self.label] = self.full_stop_check(delays)
		return t + interval

	# Computes the delay delta by applying the STDP
	def G(self, delta_t):
		if delta_t >= 0:
			delta_d = -self.B_minus*np.exp(-delta_t/self.teta_minus)
		else:
			delta_d = self.B_plus*np.exp(delta_t/self.teta_plus)
		return delta_d

	# Computes the weight delta by applying the STDP
	def F(self, delta_t):
		if delta_t >= 0:
			delta_w = self.A_plus*np.exp(-delta_t/self.tau_plus)
		else:
			delta_w = -self.A_minus*np.exp(delta_t/self.tau_minus)
		return delta_w

	# Given a post synaptic cell, returns if that cell has reached its stop condition for learning
	def stop_condition(self, delays, post_neuron):
		for pre_neuron in range(len(delays)):
			if not np.isnan(delays[pre_neuron][post_neuron]) and delays[pre_neuron][post_neuron] <= self.c:
				return True
		return False

	# Checks if all cells have reached their stop condition
	def full_stop_check(self, delays):
		for post_neuron in range(self.output.size):
			if not self.stop_condition(delays, post_neuron):
				return False
		return True

	# Applies the current weights and delays of the filter to all the cells sharing those
	def actualize_filter(self, pre_neuron, post_neuron, delta_d, delta_w, delays, weights):
		# We now find the delay/weight to use by looking at the filter
		convo_coords = [post_neuron%(x_input-filter_x+1), post_neuron//(x_input-filter_x+1)]
		input_coords = [pre_neuron%x_input, pre_neuron//x_input]
		filter_coords = [input_coords[0]-convo_coords[0], input_coords[1]-convo_coords[1]]

		# And we actualize delay/weight of the filter after the STDP
		self.filter_d[filter_coords[0]][filter_coords[1]] = max(0.01, min(self.filter_d[filter_coords[0]][filter_coords[1]]+delta_d, self.max_delay))
		self.filter_w[filter_coords[0]][filter_coords[1]] = max(0.05, self.filter_w[filter_coords[0]][filter_coords[1]]+delta_w)

		# Finally we actualize the weghts and delays of all neurons that use the same filter
		for window_x in range(0, x_input - (filter_x-1)):
			for window_y in range(0, y_input - (filter_y-1)):
				input_neuron_id = window_x+filter_coords[0] + (window_y+filter_coords[1])*x_input
				convo_neuron_id = window_x + window_y*(x_input-filter_x+1)
				if not np.isnan(delays[input_neuron_id][convo_neuron_id]) and not np.isnan(weights[input_neuron_id][convo_neuron_id]):
					delays[input_neuron_id][convo_neuron_id] = self.filter_d[filter_coords[0]][filter_coords[1]]
					weights[input_neuron_id][convo_neuron_id] = self.filter_w[filter_coords[0]][filter_coords[1]]

	# Applies delta_d and delta_w to the whole filter 
	def actualize_All_Filter(self, delta_d, delta_w, delays, weights):

		for x in range(len(self.filter_d)):
			for y in range(len(self.filter_d[x])):
				self.filter_d[x][y] = max(0.01, min(self.filter_d[x][y]+delta_d, self.max_delay))
				self.filter_w[x][y] = max(0.05, self.filter_w[x][y]+delta_w)

		# Finally we actualize the weights and delays of all neurons that use the same filter
		for window_x in range(0, x_input - (filter_x-1)):
			for window_y in range(0, y_input - (filter_y-1)):
				for x in range(len(self.filter_d)):
					for y in range(len(self.filter_d[x])):
						input_neuron_id = window_x+x + (window_y+y)*x_input
						convo_neuron_id = window_x + window_y*(x_input-filter_x+1)
						if not np.isnan(delays[input_neuron_id][convo_neuron_id]) and not np.isnan(weights[input_neuron_id][convo_neuron_id]):
							delays[input_neuron_id][convo_neuron_id] = self.filter_d[x][y]
							weights[input_neuron_id][convo_neuron_id] = self.filter_w[x][y]

	def get_filters(self):
		return self.filter_d, self.filter_w

	def check_activity_tags(self, neuron_to_check):
		for conv in neuron_activity_tag:
			if conv[neuron_to_check]:
				return False
		return True








growth_factor = (0.001/interval)*duration # <- juste faire *duration dans STDP We increase each delay by this constant each step
c = 1.0 # Stop Condition
A_plus = 0.05 # A is for weight STDP
A_minus = 0.05
B_plus = 5.0 # B is for delay STDP #2.5 is good too
B_minus = 5.0
teta_plus= 1.0 # tetas are for delay STDP
teta_minus= 1.0
tau_plus= 1.0 # tau are for weights STDP
tau_minus= 1.0
STDP_sampling = interval


neuron_reset = NeuronReset(sampling_interval=interval-15, pops=[convolution1, convolution2, convolution3, convolution4])
input_clear = InputClear(sampling_interval=interval+1, pops_to_clear_data=Input)
#Test = TestClass(sampling_interval=1.0, pop=convolution1)

SimControl1 = SimControl(sampling_interval=1.0, projection=input2convolution1, print_time=True)

Learn1 = LearningMecanisms(sampling_interval=STDP_sampling, proj=input2convolution1, input_pop=Input, output_pop=convolution1, B_plus=B_plus, B_minus=B_minus, tau_plus=tau_plus, tau_minus=tau_minus, filter_d=filter1_d, A_plus=A_plus, A_minus=A_minus, teta_plus=teta_plus, teta_minus=teta_minus, filter_w=filter1_w , stop_condition=c, growth_factor=growth_factor, label=0)
Learn2 = LearningMecanisms(sampling_interval=STDP_sampling, proj=input2convolution2, input_pop=Input, output_pop=convolution2, B_plus=B_plus, B_minus=B_minus, tau_plus=tau_plus, tau_minus=tau_minus, filter_d=filter2_d, A_plus=A_plus, A_minus=A_minus, teta_plus=teta_plus, teta_minus=teta_minus, filter_w=filter2_w, stop_condition=c, growth_factor=growth_factor, label=1)
Learn3 = LearningMecanisms(sampling_interval=STDP_sampling, proj=input2convolution3, input_pop=Input, output_pop=convolution3, B_plus=B_plus, B_minus=B_minus, tau_plus=tau_plus, tau_minus=tau_minus, filter_d=filter3_d, A_plus=A_plus, A_minus=A_minus, teta_plus=teta_plus, teta_minus=teta_minus, filter_w=filter3_w, stop_condition=c, growth_factor=growth_factor, label=2)
Learn4 = LearningMecanisms(sampling_interval=STDP_sampling, proj=input2convolution4, input_pop=Input, output_pop=convolution4, B_plus=B_plus, B_minus=B_minus, tau_plus=tau_plus, tau_minus=tau_minus, filter_d=filter4_d, A_plus=A_plus, A_minus=A_minus, teta_plus=teta_plus, teta_minus=teta_minus, filter_w=filter4_w, stop_condition=c, growth_factor=growth_factor, label=3)


"""
**************************************************************************************************
"""

sim.run(time_data, callbacks=[SimControl1, Learn1, Learn2, Learn3, Learn4, neuron_reset, input_clear])

'''

### Simulation results



#Input_data = Input.get_data().segments[0]
Input_data = neuron_reset.get_pop_spike_times()
Output_data = convolution1.get_data().segments[0]
Output_data1 = convolution2.get_data().segments[0]
Output_data2 = convolution3.get_data().segments[0]
Output_data3 = convolution4.get_data().segments[0]
weights1 = weight_recorder1.get_weights()
weights2 = weight_recorder2.get_weights()
weights3 = weight_recorder3.get_weights()
weights4 = weight_recorder4.get_weights()
delays1 = weight_recorder1.get_delays()
delays2 = weight_recorder2.get_delays()
delays3 = weight_recorder3.get_delays()
delays4 = weight_recorder4.get_delays()
print(Output_data)

filename = normalized_filename("Results", "Saliency detection", "pkl", options.simulator)

if options.plot_figure:
    figure_filename = filename.replace("pkl", "png")
    Figure(
        # raster plot of the Input neurons' spike times
        Panel(Input_data.spiketrains, ylabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data)),
        # raster plot of the Outpout neurons' spike times
        Panel(Output_data.spiketrains, ylabel="Convolutional layer 1 spikes", yticks=True, markersize=0.2, xlim=(0, time_data)),
        Panel(Output_data1.spiketrains, ylabel="Convolutional layer 2 spikes", yticks=True, markersize=0.2, xlim=(0, time_data)),
        Panel(Output_data2.spiketrains, ylabel="Convolutional layer 3 spikes", yticks=True, markersize=0.2, xlim=(0, time_data)),
        Panel(Output_data3.spiketrains, ylabel="Convolutional layer 4 spikes", yticks=True, markersize=0.2, xlim=(0, time_data)),
        # membrane potential of the Output neurons
        Panel(Output_data.filter(name='v')[0], ylabel="Membrane potential (mV) input2convolution1", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        Panel(Output_data1.filter(name='v')[0], ylabel="Membrane potential (mV) input2convolution2", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        Panel(Output_data2.filter(name='v')[0], ylabel="Membrane potential (mV) input2convolution3", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        Panel(Output_data3.filter(name='v')[0], ylabel="Membrane potential (mV) input2convolution4", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        
        # evolution of the synaptic weights with time
        Panel(weights1, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Weights input2convolution1",
                legend=False, xlim=(0, time_data)),
        Panel(weights2, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Weights input2convolution2",
                legend=False, xlim=(0, time_data)),
        Panel(weights3, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Weights input2convolution3",
                legend=False, xlim=(0, time_data)),
        Panel(weights4, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Weights input2convolution4",
                legend=False, xlim=(0, time_data)),

        # evolution of the synaptic delays with time
        Panel(delays1, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Delays input2convolution1",
                legend=False, xlim=(0, time_data)),
        Panel(delays2, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Delays input2convolution2",
                legend=False, xlim=(0, time_data)),
        Panel(delays3, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Delays input2convolution3",
                legend=False, xlim=(0, time_data)),
        Panel(delays4, xticks=True, yticks=True, xlabel="Time (ms)", ylabel="Delays input2convolution4",
                legend=False, xlim=(0, time_data)),
        title="Simple 2 layers spiking neural network",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)


sim.end()
'''
