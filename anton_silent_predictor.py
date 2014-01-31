import pickle,sys,pprint,random,time,math
from collections import deque
import numpy as np
from numpy.random import RandomState
import pickle
from copy import copy,deepcopy
import argparse

import sys
import os

#FFNN supervised learning packages 
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer, LinearLayer
from pybrain.tools.validation import ModuleValidator
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection

#5th January 2014. Cumule Algorithm (Chrisantha Fernando)

PHASE_1_LENGTH = 100000

WEIGHT_DECAY=0.1
BACKTIME=10
PREDICTOR_MUTATION_PROBABILITY=0.8

parser = argparse.ArgumentParser()
parser.add_argument("--timelimit",default=50,type=int)
parser.add_argument("-n","--num_predictors",help="population size(default:50)",default=50,type=int)
parser.add_argument("--runs",help="number of runs(default:1)",default=1,type=int)
parser.add_argument("--epochs",help="number of epochs for each training(default:5)",default=5,type=int)
parser.add_argument("-ts","--test_set_length",help="test set length(default:50)",default=50,type=int)
parser.add_argument("-e","--evolution_period", help="evolution period(default:10)", type=int, default=10)
parser.add_argument("-a","--archive_threshold", help="threshold for getting into the archive(default: 0.0004)", type=float, default=0.0004)
parser.add_argument("-lr","--learning_rate", help="learning rate for predictors(default: 0.01)", type=float, default=0.01)
parser.add_argument("-r","--replication", help="enable weights replication(default: no)",action="store_true", default=False)
parser.add_argument("-lg","--logfile", help="log file name(default: prediction.log)",type=str, default="prediction.log")
parser.add_argument("-s","--statsname", help="stats file name(default: )",type=str, default="")
parser.add_argument("-i","--mutate_input", help="enable input mask mutation(default: yes)",action="store_true", default=True)
parser.add_argument("--episode_length", help="number of samples per episode(default: 50)", type=int, default=10)
parser.add_argument("--show_test_error", help="test archive and show the plot", action="store_true",default=False)
parser.add_argument("--show_plots", help="show live plots", action="store_true",default=False)
parser.add_argument("--sliding_training", help="use sliding window of examples", action="store_true",default=False)
parser.add_argument("--input_mutation_prob", help="input mutation probability per bit(default: 0.05)", type=float, default=0.05)
parser.add_argument("--output_mutation_prob", help="output mutation probability per mask(default: 0.9)", type=float, default=0.9)
parser.add_argument("--replication_prob", help="weight copy probability per weight(default: 0.1)", type=float, default=0.1)
parser.add_argument("--predictor_mutation_prob", help="tournament loser mutation probability(default: 1)", type=float, default=1.0)
parser.add_argument("--random_input_masks", help="initialise population with random input masks(default:False)", action="store_true",default=False)
parser.add_argument("--correct_input_masks", help="supply predictors with correct masks(default:False)", action="store_true",default=False)
parser.add_argument("--fixed_hidden_layer",default=None,type=int)
parser.add_argument("--max_hidden_units",default=50,type=int)
parser.add_argument("--max_hiden_layers",default=2,type=int)
parser.add_argument("--world_file", type=str, default=None)
parser.add_argument("--test_name",type=str)
parser.add_argument("--fixed_structures", action="store_true", default=False)
parser.add_argument("--outputdir", help="folder for log files (default: '')", type=str, default='')
parser.add_argument("--use_common_weights", help="only copy weights from the same inputs (default: False)", action="store_true", default=False)

from old_world import OldWorld
from new_world import NewWorld

FLAGS={}

WORLD_STATE_SIZE=0
WORLD_ACTION_SIZE=0

STRUCTURES = [(5,), (20,), (5, 5), (20, 20)]

def getAllInputWeights(net):
	return net.connections[net['in']][0].params.reshape(
		net['in'].dim, net['hidden0'].dim)

def copyInputWeights(selfNet, otherNet, selfInputMask, otherInputMask, use_common_weights=False):
	"""Copies input weights from otherNet to selfNet """
	all_weights = getAllInputWeights(otherNet)
	param_range = getInputMaskParamRange(selfNet)
	param_shape = (selfNet['in'].dim, selfNet['hidden0'].dim)
	if use_common_weights:
		common_weights = extractCommonInputWeights(all_weights, otherInputMask, selfInputMask)
		if len(common_weights) == 0:
			return selfNet
		weights = []
		for inp in xrange(len(selfInputMask)):
			if (selfInputMask[inp] == 1):
				weights.append(common_weights[inp])
		weights = np.array(weights)
	else:
		weights = all_weights
	trunc_weights = truncateParams(weights, (min(
				sum(selfInputMask), sum(otherInputMask)), min(
				selfNet['hidden0'].dim, otherNet['hidden0'].dim)))
	replaceNetParams(selfNet, param_range, param_shape, trunc_weights)

def extractCommonInputWeights(weights, selfInputMask, otherInputMask):
	common_inputs = [selfInputMask[i]*otherInputMask[i] for i in xrange(len(selfInputMask))]
	if sum(common_inputs) == 0:
		return np.zeros(0)
	weight_index = 0
	extracted_weights = []
	for inp in xrange(len(selfInputMask)):
		if selfInputMask[inp] == 0:
			extracted_weights.append(None)
			continue
		weight_array = weights[weight_index]
		weight_index += 1
		if common_inputs[inp] == 0:
			extracted_weights.append(None)
		else:
			extracted_weights.append(weight_array)
	return np.array(extracted_weights)

def getInputMaskParamRange(net):
	input_start = 0
	input_params_length = net.connections[net['in']][0].paramdim
	if net['bias'] is None:
		return (input_start, input_params_length)

	for conn in net.connections[net['bias']]:
		input_start += conn.paramdim
	return (input_start, input_start+input_params_length)

def truncateParams(params_array, trunc_size):
	new_array = np.zeros(trunc_size)
	for i in xrange(trunc_size[0]):
		if params_array[i] is not None:
			new_array[i] = np.copy(params_array[i][:trunc_size[1]])
		else:
			new_array[i] = None
	return new_array

def replaceNetParams(net, param_range, param_shape, new_params):
	net_params = np.copy(net.params)
	offset = param_range[0]
	for row in xrange(min(param_shape[0], new_params.shape[0])):
		if not math.isnan(new_params[row][0]):
			length = min(param_shape[1], new_params.shape[1])
			net_params[offset:offset+length] = new_params[row][:length]
		offset += param_shape[1]
	net._setParameters(net_params)

class Predictor():

	def __init__(self, structure, inputMask):

		self.learning_rate = FLAGS.learning_rate
		self.inputMask=inputMask
		self.inSize=self.inputMask.count(1)

		self.structure = tuple(structure)
		self.createStructure()

		self.mse = 100
		self.age=0

		self.error = 0
		self.errorHistory = []
		self.dErrorHistory = []
		self.slidingError = 0
		self.dError = 0
		self.fitness = 0
		self.previousData=[]
	
	def createStructure(self):
		self.hidden_neurons=sum(self.structure)
		self.hidden_layers=len(self.structure)

		self.ds = SupervisedDataSet(self.inSize, 1)

		args = [self.inSize]
		for hl in self.structure:
			args.append(hl)
		args.append(1)
		kwargs = {'hiddenclass': TanhLayer, 'bias': True}
		self.net = buildNetwork(*(tuple(args)), **kwargs)
		self.trainer = BackpropTrainer(self.net, self.ds, learningrate=self.learning_rate, verbose = False, weightdecay=WEIGHT_DECAY)

	def setProblem(self,problem):
		self.problem=problem

	def getPrediction(self, input):
		out = self.net.activate(input)
		return out

	def trainPredictor(self):

		self.age+=1

		new_ds=deepcopy(self.ds)

		if FLAGS.sliding_training:
			if len(self.previousData)!=0:
				for sample,target in self.previousData:
					new_ds.addSample(sample,target)
		

		self.trainer.setData(new_ds)
		for i in range(FLAGS.epochs):
			e = self.trainer.train()
		
		if FLAGS.sliding_training:
			self.previousData=deepcopy(self.ds)

		# self.error = e

		# if len(self.errorHistory) < 5:  
		# 	self.errorHistory.append(e)
		# else:
		# 	for i in range(len(self.errorHistory)-1):
		# 		self.errorHistory[i] = self.errorHistory[i+1]
		# 	self.errorHistory[-1] = e

		# #Sliding window error over appeox last 10 episodes characturistic time. 
		# self.slidingError = self.slidingError*0.9 + self.error
		# #Instantaneous difference in last er ror between episodes. 
		# if len(self.errorHistory) > 1:
		# 	self.dError = self.errorHistory[-1] - self.errorHistory[-2] 

		return e


	def getFitness(self):
		return self.fitness
	
	def setFitness(self,fitness):
		self.fitness_change=fitness-self.fitness
		self.fitness=fitness

	def storeDataPoint(self, inputA, targetA):
		assert(len(self.prepareInput(inputA))==self.net.modulesSorted[1].dim)
		self.ds.addSample(self.prepareInput(inputA), self.prepareTarget(targetA))

	def prepareTarget(self,targetA):
		return targetA[self.problem]


	def prepareInput(self,raw):
		inputA = []
		for j in range(len(raw)):
		    if self.inputMask[j]==1:
		        inputA.append(raw[j])
		return inputA

	def predict(self,inputA):
		return self.net.activate(self.prepareInput(inputA))[0]

class Agent(): 
		def __init__(self,world):
			self.predictors = []
			self.world=world
			self.archive = [None for i in xrange(self.world.state_size)]
			self.initialisePredictors()

		def archivePredictors(self):
			for problem, predictors in enumerate(self.problemsDistribution()):
				try:
					best = np.argmax([p.fitness for p in predictors])
					if predictors[best].fitness > FLAGS.archive_threshold:
						continue
					if (self.archive[problem] is None) or (self.archive[problem].fitness < predictors[best].fitness):
						self.archive[p.problem] = p
				except:
					pass

		def unitsDistribution(self,num,layers_num):
			per_layer=num/layers_num
			d=(num-per_layer*layers_num)
			distr=[]
			if d!=0:
				for i in range(d):
					distr.append(per_layer+1)
			else:
				distr=[per_layer]
			distr+=[per_layer for i in range(layers_num-len(distr))]
			return distr


		def initialisePredictors(self):
			input_size=WORLD_STATE_SIZE+WORLD_ACTION_SIZE
			if FLAGS.num_predictors<WORLD_STATE_SIZE:
				raise "number of predictors is less than number of problems!"

			problem_distribution=self.unitsDistribution(FLAGS.num_predictors,WORLD_STATE_SIZE)
			random.shuffle(problem_distribution)
			cur_problem=0
			
			for i in range(FLAGS.num_predictors):

				if FLAGS.correct_input_masks:
					mask=self.world.input_masks()[cur_problem]
					# in order to have at least one input to the predictor
					# (also to avoid annoying errors)
					if mask.count(1) == 0:
						on_bit=random.randint(1,input_size-1)
						mask[on_bit] = 1
				elif FLAGS.random_input_masks:
					on_bits=random.randint(1,input_size/2)
					mask=[1]*on_bits
					mask+=[0]*(input_size-on_bits)
					random.shuffle(mask)
				else:
					mask = [1 for i in range(input_size)]

				hidden_units=random.randint(FLAGS.max_hiden_layers,FLAGS.max_hidden_units)
				hidden_layers=random.randint(1,FLAGS.max_hiden_layers)

				if FLAGS.fixed_structures:
					p=Predictor(STRUCTURES[random.randint(0, len(STRUCTURES)-1)], mask)
				else:
					p=Predictor(self.unitsDistribution(hidden_units,hidden_layers),mask)

				p.setProblem(cur_problem)
				problem_distribution[cur_problem]-=1

				assert(p.net.modulesSorted[1].dim==p.inputMask.count(1))

				if problem_distribution[cur_problem]==0:
					cur_problem+=1;

				self.predictors.append(p)


		def problemsDistribution(self):
			r=[[] for i in range(WORLD_STATE_SIZE)]
			for predictor in self.predictors:
				r[predictor.problem].append(predictor)
			return r
		
		def getRandomMotor(self):
			return [random.uniform(0,1), random.uniform(0,1)]
	
		def storeDataPoint(self, inp, targ):
			for i in range(FLAGS.num_predictors):
				self.predictors[i].storeDataPoint(inp, targ)

		def trainPredictors(self):
			ep = []
			for i in range(FLAGS.num_predictors):
				e = self.predictors[i].trainPredictor()
				ep.append(e)
			return ep

		def clearPredictorsData(self):
			for i in range(FLAGS.num_predictors):
				self.predictors[i].ds.clear()

		def changeTournamentLoser(self, winner, loser):
			if FLAGS.fixed_structures:
				if self.archive[self.predictors[winner].problem] is not None:
					problem = self.predictors[loser].problem
					structure = self.predictors[winner].structure
					inMask = self.predictors[winner].inputMask
				else:
					problem = self.predictors[loser].problem
					structure = self.predictors[winner].structure
					inMask = self.predictors[loser].inputMask
				self.predictors[loser] = Predictor(structure, inMask)
				self.predictors[loser].setProblem(problem)

class Cumule():
		def __init__(self, world_instance=None, run_n=1):

			if not world_instance:
				self.world = World()
			else:
				self.world=deepcopy(world_instance)

			self.popFitHistory=np.ndarray((FLAGS.num_predictors,BACKTIME))*0
			self.timestep=0
			self.run_n=run_n
			self.predictor_statsfile=open(FLAGS.outputdir+FLAGS.statsname+'_predictors.csv','a',1)
			self.archive_statsfile=open(FLAGS.outputdir+FLAGS.statsname+'_archive.csv','a',1)
			self.solution_statsfile=open(FLAGS.outputdir+FLAGS.statsname+'_solution.csv','a',1)

		def generate_test_set(self,length):
			m = self.agent.getRandomMotor()
			s = self.world.updateState(m)
			test_set=[]
			for t in range(length):
				m = self.agent.getRandomMotor()
				stp1 = self.world.updateState(m)
				inp = np.concatenate((s,m), axis = 0)
				test_set.append([copy(inp),copy(stp1)])
				s = stp1
			return test_set


		def predictors_stats(self):
			for i,p in enumerate(self.agent.predictors):
				self.predictor_statsfile.write("{run};{timestep};{predictor_number};{problem};{error};'{inputmask}';{hiddenlayers};{hiddenneurons}\n".format(
						run=self.run_n, timestep=self.timestep, predictor_number=i, problem=p.problem, error=-p.fitness,
						inputmask="".join([str(k) for k in p.inputMask]), hiddenlayers=p.hidden_layers, hiddenneurons=p.hidden_neurons))
		def archive_stats(self):
			if self.timestep>0 and (self.timestep % 3)==0:
				self.archive_statsfile.write("{run};{timestep};{min_error}\n".format(
						run=self.run_n,timestep=self.timestep,min_error=self.testSolution(self.getSolution(),self.test_set)))

		def getSolution(self):
			sol=[None for i in range(WORLD_STATE_SIZE)]
			for p in self.agent.predictors:
				cur_solver=sol[p.problem]
				if cur_solver==None or cur_solver.fitness<p.fitness:
					sol[p.problem]=p
			return sol
		
		def partialSolution(self,predictor):
			solution=[None for i in range(WORLD_STATE_SIZE)]
			solution[predictor.problem]=predictor
			return solution

		def testSolution(self,solution,test_set,write=False):
			m = self.agent.getRandomMotor()
			s = self.world.updateState(m)
			err=0

			dims=[]
			for k,v in enumerate(solution):
				if v!=None:
					dims.append(k)


			for n,t in enumerate(test_set):
				inp=t[0]
				stp1=t[1]

				expected=stp1
	
				for i in dims:
					predicted=solution[i].predict(inp)
					err+=(predicted-expected[i])**2
					if write:
						self.solution_statsfile.write("{run};{problem};{expected};{predicted};{example}\n".format(
								run=self.run_n,problem=i,expected=expected[i],predicted=predicted,example=n))

			return 0.5*err/len(test_set)

		def setFitnesses(self, test_set):
			errs=[0.0]*FLAGS.num_predictors
			for sample in test_set:
				inp=sample[0]
				outp=sample[1]

				for num,p in enumerate(self.agent.predictors):
					predicted=p.predict(inp)
					errs[num]+=(predicted-outp[p.problem])**2

			errs=np.multiply(errs,-0.5/len(test_set))

			for num,p in enumerate(self.agent.predictors):
				p.setFitness(errs[num])


		def collect_training_data(self, initialState):
			s = initialState[:]
			for t in range(FLAGS.episode_length):
				m = self.agent.getRandomMotor()
				stp1 = self.world.updateState(m)
				inp = np.concatenate((s,m), axis = 0)
				self.agent.storeDataPoint(inp, stp1) 
				s = stp1

		def pick_predictors(self):
			p1 = random.randint(0, FLAGS.num_predictors-1)
			p2 = random.randint(0, FLAGS.num_predictors-1)
			while p1 == p2:
				p2 = random.randint(0, FLAGS.num_predictors-1)
			return p1, p2

		def tournament(self, p1, p2):
			if self.agent.predictors[p1].getFitness() > self.agent.predictors[p2].getFitness():
				winner = p1
				loser = p2
			else:
				winner = p2
				loser = p1
			return winner, loser

		def run(self): 
			self.timestep=0
			self.agent = Agent(self.world)

			logfile=open(FLAGS.outputdir+FLAGS.logfile,'w',1)
			errHis = []

			self.test_set=self.generate_test_set(FLAGS.test_set_length)


			m = self.agent.getRandomMotor()
			s = self.world.resetState(m)

			for i in range(PHASE_1_LENGTH):
				self.timestep+=1
				
				if self.timestep==FLAGS.timelimit+1 and FLAGS.timelimit!=-1:
					return self.testSolution(self.getSolution(),self.test_set,True)

				logfile.write("Timestep:"+str(i)+"\n")

				# training of predictors
				self.collect_training_data(s)
				self.agent.trainPredictors()
				self.setFitnesses(self.generate_test_set(FLAGS.episode_length))
				self.agent.archivePredictors()
				self.agent.clearPredictorsData()
				distr=self.agent.problemsDistribution()

				if i%FLAGS.evolution_period == 0:
					p1, p2 = self.pick_predictors()
					winner, loser = self.tournament(p1, p2)
					self.agent.changeTournamentLoser(winner, loser)

				self.predictors_stats()
				self.archive_stats()
			
			logfile.close()
			self.statsfile.close()


if __name__ == '__main__':
	FLAGS=parser.parse_args()

	if (FLAGS.outputdir != "") and (FLAGS.outputdir[-1] != "/"):
		FLAGS.outputdir += "/"

	try:
		os.mkdir(FLAGS.outputdir)
	except:
		pass

	st=open(FLAGS.outputdir+FLAGS.statsname+'_predictors.csv',"w")
	st.write("Run; Timestep; Predictor Number; Problem; Error; Input Mask; HiddenLayers; HiddenNeurons\n")
	st.close()

	st=open(FLAGS.outputdir+FLAGS.statsname+'_archive.csv',"w")
	st.write("Run; Timestep; MinError\n")
	st.close()

	st=open(FLAGS.outputdir+FLAGS.statsname+'_solution.csv',"w")
	st.write("Run; Problem; Expected; Predicted; Example\n")
	st.close()

	fl=vars(FLAGS)
	for k in sorted(fl.iterkeys()):
		print k+": "+str(fl[k])

	avg=0.0
	errs=[]

	# if FLAGS.world_file!=None:
	inp=open(FLAGS.world_file,'rb')
	world_instance=pickle.load(inp)
	inp.close()

	WORLD_STATE_SIZE=world_instance.__class__.state_size
	WORLD_ACTION_SIZE=world_instance.__class__.action_size

	# a=Agent()
	# a.initialisePredictors()
	# print [len(p) for p in a.problemsDistribution()]
	# exit()
	# else:
	# 	world_instance=None

	for i in range(FLAGS.runs):
		c = Cumule(world_instance,i)
		result=c.run()
		errs.append(result)
		print result
	print "Average: "+str(np.mean(errs))
	print "Standard deviation: "+str(np.std(errs))
	print "Min: "+str(np.min(errs))
	print "Max: "+str(np.max(errs))

	if FLAGS.show_test_error:
		# c.test_archive()
		raw_input("Press Enter to exit")

