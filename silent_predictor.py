import pylab,pickle,sys,pprint,random,time,math
from collections import deque, defaultdict
import numpy as np
from numpy.random import RandomState
import pickle
from copy import copy,deepcopy
from matplotlib.pyplot import *
import argparse
import shutil
import os

import sys
import atexit

#FFNN supervised learning packages 
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer, LinearLayer
from pybrain.tools.validation import ModuleValidator


#5th January 2014. Cumule Algorithm (Chrisantha Fernando)

PHASE_1_LENGTH = 10000

# EVOLUTION_PERIOD = 2 #Evolve predictors every 10 episodes. 
WEIGHT_DECAY=0.1
# MUTATE_MASK_PROBABILITY = 0.9
BACKTIME=10
PREDICTOR_MUTATION_PROBABILITY=0.8
# WEIGHT_COPY_PROBABILITY=0.05

INITIAL_INPUT_ALL_ONES = "ones"
INITIAL_INPUT_RANDOM = "random"
INITIAL_INPUT_CORRECT = "correct"

MAX_TEST_ERROR = 10

parser = argparse.ArgumentParser()
parser.add_argument("timelimit",default=50,type=int)
parser.add_argument("-n","--num_predictors",help="population size(default:50)",default=50,type=int)
parser.add_argument("--runs",help="number of runs(default:1)",default=1,type=int)
parser.add_argument("--epochs",help="number of epochs for each training(default:5)",default=5,type=int)
parser.add_argument("-ts","--test_set_length",help="test set length(default:50)",default=50,type=int)
parser.add_argument("-e","--evolution_period", help="evolution period(default:10)", type=int, default=10)
parser.add_argument("-a","--archive_threshold", help="threshold for getting into the archive(default: 0.02)", type=float, default=0.02)
parser.add_argument("-lr","--learning_rate", help="learning rate for predictors(default: 0.01)", type=float, default=0.01)
parser.add_argument("-r","--replication", help="enable weights replication(default: no)",action="store_true", default=False)
parser.add_argument("-lg","--logfile", help="log file name(default: prediction.log)",type=str, default="prediction.log")
parser.add_argument("-i","--disable_input_mutation", help="disable input mask mutation(default: false)",action="store_true", default=False)
parser.add_argument("--episode_length", help="number of samples per episode(default: 50)",type=int, default=50)
parser.add_argument("--show_test_error", help="test archive and show the plot", action="store_true",default=False)
parser.add_argument("--show_plots", help="show live plots", action="store_true",default=False)
parser.add_argument("--sliding_training", help="use sliding window of examples", action="store_true",default=False)
parser.add_argument("--input_mutation_prob", help="input mutation probability per bit(default: 0.05)", type=float, default=0.05)
parser.add_argument("--output_mutation_prob", help="output mutation probability per mask(default: 0.9)", type=float, default=0.9)
parser.add_argument("--replication_prob", help="weight copy probability per weight(default: 0.1)", type=float, default=0.1)
parser.add_argument("--predictor_mutation_prob", help="tournament loser mutation probability(default: 1)", type=float, default=1.0)
parser.add_argument("--punish_archive_factor", help="factor by which to multiply error for predictors already in archive (default: 15)", type=float, default=15)
parser.add_argument("--punish_population_factor", help="factor by which to encourage output mask mutation (default: 2)", type=float, default=2)
parser.add_argument("--outputdir", help="folder for log files (default: '')", type=str, default='')
parser.add_argument("--initial_input", help="input mask initialisation options (default: ones, other options: random, correct)", type=str, default='ones')
parser.add_argument("--punish_inputs_base", help="error = error*(base^(number of bits in input mask))/base^2.5 (default: 1.6)", type=float, default=1.6)
parser.add_argument("--recombination_prob", help="input masks recombination probability (default: 0.0)", type=float, default=0.0)
parser.add_argument("--population_test_length", help="number of time steps for which to test predictors in population (default: 10)", type=int, default=10)
parser.add_argument("--train_error", help="use training error instead of test error for archiving predictors (default: false)", action="store_true", default=False)
parser.add_argument("--old_world", help="use old 8-dimensional world (default: false)", action="store_true", default=False)
parser.add_argument("--check_input_mask", help="plot a graph of how many bits are wrong in the input masks of archived predictors", action="store_true", default=False)
parser.add_argument("--disable_evolution", help="do not use evolution - just train (default: False)", action="store_true", default=False)
parser.add_argument("--hidden_layer_size", help="size of the hidden layer (default: 10)", type=int, default=10)
parser.add_argument("--hidden_layer_number", help="number of hidden layers (default: 1)", type=int, default=1)
parser.add_argument("--relative_error", help="use error = (state-prediction)*abs(state) to take into account magnitude of state (default: False)",
					action="store_true", default=False)

from world import World as OldWorld
from new_world import World as NewWorld

World = None

def list_diff(list1, list2):
	list_out = list1[:]
	for i in xrange(len(list_out)):
		list_out[i] = abs(list_out[i] - list2[i])
	return list_out

def stringify_mask(mask):
	return "".join([str(b) for b in mask])

class Predictor(): 

	def __init__(self, inSize, outSize, LearningRate):

		self.learning_rate = LearningRate
		self.ds = SupervisedDataSet(inSize, outSize)
		args = [inSize]
		for hl in xrange(FLAGS.hidden_layer_number):
			args.append(FLAGS.hidden_layer_size)
		args.append(outSize)
		kwargs = {'hiddenclass': TanhLayer, 'bias': True}
		self.net = buildNetwork(*(tuple(args)), **kwargs)
		self.trainer = BackpropTrainer(self.net, self.ds, learningrate=self.learning_rate, verbose = False, weightdecay=WEIGHT_DECAY)
		self.prediction = [0] * outSize
		self.mse = 100
		self.age=0
		
#		self.outputMask = [random.randint(0, 1) for i in range(outSize)]
		self.outputMask = [0]*outSize
		r = random.randint(0,outSize-1)
		self.outputMask[r] = 1
		#Specific to Mai's code. Make input and output masks.
		if FLAGS.initial_input == INITIAL_INPUT_CORRECT:
			self.inputMask = World.correct_masks[r]
		elif FLAGS.initial_input == INITIAL_INPUT_RANDOM:
			self.inputMask = [random.randint(0, 1) for i in range(inSize)]
		elif FLAGS.initial_input == INITIAL_INPUT_ALL_ONES:
			self.inputMask = [1]*inSize
		self.trainError = 0
		self.testError = MAX_TEST_ERROR
		self.trainErrorHistory = []
		self.dErrorHistory = []
		self.slidingError = 0
		self.dError = 0
		self.fitness = 0
		self.problem=r
		self.previousData=[]

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

		#Update possible fitness indicators. 
		#Error now
		self.trainError = e
		#Entire error history
		if len(self.trainErrorHistory) < 5:  
			self.trainErrorHistory.append(e)
		else:
			for i in range(len(self.trainErrorHistory)-1):
				self.trainErrorHistory[i] = self.trainErrorHistory[i+1]
			self.trainErrorHistory[-1] = e

		#Sliding window error over appeox last 10 episodes characturistic time. 
		self.slidingError = self.slidingError*0.9 + self.trainError
		#Instantaneous difference in last er ror between episodes. 
		if len(self.trainErrorHistory) > 1:
			self.dError = self.trainErrorHistory[-1] - self.trainErrorHistory[-2] 

		return e

	def getTrainFitness(self, fitness_type, rewardMinimal=True):
		if rewardMinimal:
			# Trying to get a difference of between 10 and 20 for 8 bits set vs 2.5 bits set
			# 2.5 is roughly the average number of bits that the "correct" input mask would use in this case
			errMultiplier = FLAGS.punish_inputs_base**(sum(self.inputMask)-2.5)
		else:
			errMultiplier = 1

		fit = 0 
		#Fitness function 1 Chrisantha's attempt
		if fitness_type == 0:#SIMPLE MINIMIZE PREDICTION ERROR FITNESS FUNCTION FOR PREDICTORS.
#           fit = -self.dError/(1.0*self.error)
			fit = -self.trainError*errMultiplier
		elif fitness_type == 1:
			#Fitness function 2 Mai's attempt (probably need to use adaptive thresholds for this to be ok)
			if self.trainError > ERROR_THRESHOLD and self.dError > DERROR_THRESHOLD:
				fit = 0
			else:
				fit = 1

		self.fitness = fit
		return fit 

	def getTestFitness(self, fitness_type, test_length, test_agent, test_world, rewardMinimal=True):
		distr = [[] for i in xrange(test_world.state_size)]
		distr[self.problem] = [self]

		_, _, mse = test_distribution(distr, test_length, test_agent, test_world, [self.problem], plot_data=False, get_best=False)

		if rewardMinimal:
			# Trying to get a difference of between 10 and 20 for 8 bits set vs 2.5 bits set
			# 2.5 is roughly the average number of bits that the "correct" input mask would use in this case
			errMultiplier = FLAGS.punish_inputs_base**(sum(self.inputMask)-2.5)
		else:
			errMultiplier = 1

		fit = 0
		#Fitness function 1 Chrisantha's attempt 
		if fitness_type == 0:#SIMPLE MINIMIZE PREDICTION ERROR FITNESS FUNCTION FOR PREDICTORS. 
#           fit = -self.dError/(1.0*self.error)
			fit = -mse*errMultiplier
		elif fitness_type == 1:
			#Fitness function 2 Mai's attempt (probably need to use adaptive thresholds for this to be ok)
			if mse > ERROR_THRESHOLD and self.dError > DERROR_THRESHOLD:
				fit = 0
			else:
				fit = 1

		self.fitness = fit
		return fit

	def storeDataPoint(self, inputA, targetA):
		self.ds.addSample(inputA, targetA)

	def predict(self,inputA):
		return self.net.activate(inputA)

	def predict_masked(self,inputA):
		input_masked = []
		for i in xrange(len(inputA)):
			input_masked.append(inputA[i]*self.inputMask[i])
		return self.predict(input_masked)

class Agent(): 
		def __init__(self):

			#The agent has a population of M predictors. 
			self.predictors = []
			self.archive=[0 for i in range(World.state_size)]
			for i in range(FLAGS.num_predictors):
				p=Predictor(World.state_size + World.action_size,World.state_size, FLAGS.learning_rate)
				self.predictors.append(p)

		def problemsDistribution(self):
			r=[[] for i in range(World.state_size)]
			for predictor in self.predictors:
				r[predictor.problem].append(predictor)
			return r

		def averageErrors(self,distr):
			r=[]

			for problem, predictors in enumerate(distr):
				error=np.mean([p.trainError for p in predictors])
				r.append(error)	

			return r

		def minimumErrors(self, distr, use_train):
			r=[]

			for problem, predictors in enumerate(distr):
				if len(predictors)>0:
					if use_train:
						error=min([p.trainError for p in predictors])
					else:
						error=min([p.testError for p in predictors])
				else:
					error=5

				r.append(error)	


			return r

		def bestSolved(self, distr, use_train):
			min_error=10000000000
			best_solved=-1
			best_predictor=-1
			for problem, predictors in enumerate(distr):
				if len(predictors)!=0:
					if use_train:
						errors=[p.trainError for p in predictors]
					else:
						errors=[p.testError for p in predictors]
					best=np.argmin(errors)
					err=errors[best]
					if err<min_error:
						best_solved=problem
						min_error=err
						best_predictor=predictors[best]
			return (best_solved,min_error,best_predictor)


		def bestSolvingSpeed(self,distr):
			min_speed=100000000
			fastest=-1
			for problem, predictors in enumerate(distr):
				if len(predictors)!=0:
					speed=min([p.dError for p in predictors])
					if speed<min_speed and speed<0:
						fastest=problem
						min_speed=speed
			return (fastest,min_speed)

		# execute this AFTER storing into archive and BEFORE new training
		def problemsMutationProbabilities(self, distr, use_train):
			r=[]
			min_err=1000000
			max_err=-1000000

			for problem, predictors in enumerate(distr):
				if len(predictors)!=0:
					if use_train:
						err=np.mean([p.trainError for p in predictors])
					else:
						err=np.mean([p.testError for p in predictors])
				else:
					err=-1

				if self.archive[problem]!=0:
					err=err*FLAGS.punish_archive_factor # we discourage agent from generating predictors that solve already solved problems
				
				if err>0 and err<min_err:
					min_err=err

				if err>0 and err>max_err:
					max_err=err
			
				r.append(err)

			for k,v in enumerate(r):
				if self.archive[k]==0:
					r[k]=max_err*5
				if v<0:
					r[k]=max_err*3

			
			r=np.divide(r,sum(r))
			return r
		
		def minTrainErrors(self,distr,rewardMinimal=True):
			r=[]
			for problem, predictors in enumerate(distr):
				if len(predictors)!=0:
					if rewardMinimal:
						# Trying to get a difference of between 10 and 20 for 8 bits set vs 2.5 bits set
						# 2.5 is roughly the average number of bits that the "correct" input mask would use in this case
						errMultipliers = [(FLAGS.punish_inputs_base**(sum(p.inputMask)-2.5)) for p in predictors]
					else:
						errMultipliers = [1 for p in predictors]
					errors=[p.trainError for p in predictors]
					for e in xrange(len(errors)):
						errors[e] *= errMultipliers[e]
					best=np.argmin(errors)
					err=errors[best]

					r.append((problem,err,predictors[best]))	
			return r

		def minTestErrors(self, distr, test_length, test_world, itime, logfile):
			r=[]

			_, sum_errors, _ = test_distribution(distr, FLAGS.test_set_length, self, test_world,
												 range(test_world.state_size), plot_data=False, get_best=True)

			for problem, predictors in enumerate(distr):
				if len(predictors) != 0:
					logfile.write("time %s, predictor %s, errors %s\n" % (itime, problem, sum_errors[problem]))
					best = np.argmin(sum_errors[problem])
					err = sum_errors[problem][best]
					r.append((problem, err, predictors[best]))
			return r

		def problemsAllocation(self,distr):
			return [len(predictors) for predictors in distr]


		def getRandomMotor(self):
			return [random.uniform(0,1), random.uniform(0,1)]
	
		def storeDataPoint(self, inp, targ):

			for i in range(FLAGS.num_predictors):
				#APPLY INPUT AND OUTPUT MASKS BEFORE SENDING DATA TO PREDICTORS. 
				inputA = [0]*len(inp)
				for j in range(len(inp)):
					inputA[j] = inp[j]*self.predictors[i].inputMask[j]
				target = [0]*len(targ)
				for j in range(len(targ)):
					target[j] = targ[j]*self.predictors[i].outputMask[j]

				self.predictors[i].storeDataPoint(inputA, target)

		def trainPredictors(self):
			ep = []
			for i in range(FLAGS.num_predictors):
				e = self.predictors[i].trainPredictor()
				ep.append(e)
			return ep

		def createPredictor(self, problem, test_world):
			p=Predictor(World.state_size + World.action_size,World.state_size, FLAGS.learning_rate)
			p.problem=problem
			p.outputMask = [0]*World.state_size
			p.outputMask[problem]=1
			if (FLAGS.disable_evolution or FLAGS.disable_input_mutation) and (FLAGS.initial_input == INITIAL_INPUT_CORRECT):
				p.inputMask = test_world.correct_masks[problem]
			return p

		def clearPredictorsData(self):
			for i in range(FLAGS.num_predictors):
				self.predictors[i].ds.clear()

		def copyAndMutatePredictor(self, winner, loser,distribution):
			newLoser = deepcopy(self.predictors[winner])
			self.predictors[loser] = newLoser

			self.predictors[loser].learning_rate =  FLAGS.learning_rate
			self.predictors[loser].ds = SupervisedDataSet(World.state_size+World.action_size, World.state_size)
			args = [World.state_size+World.action_size]
			for hl in xrange(FLAGS.hidden_layer_number):
				args.append(FLAGS.hidden_layer_size)
			args.append(World.state_size)
			kwargs = {'hiddenclass': TanhLayer, 'bias': True}
			self.predictors[loser].net = buildNetwork(*(tuple(args)), **kwargs)
			self.predictors[loser].trainer = BackpropTrainer(self.predictors[loser].net, self.predictors[loser].ds, learningrate=self.predictors[loser].learning_rate, verbose = False, weightdecay=WEIGHT_DECAY)
			self.predictors[loser].outputMask = self.predictors[winner].outputMask

			if FLAGS.replication:
				for i in range(len(self.predictors[loser].net.params)):
					if random.uniform(0,1)<FLAGS.replication_prob:
						self.predictors[loser].net.params[i] = self.predictors[winner].net.params[i]
			
			#self.predictors[loser].net._setParameters(self.predictors[loser].net.params)

			if random.uniform(0,1) < FLAGS.recombination_prob:
				recombination_point = random.randint(0, len(loser.inputMask)-1)
				self.predictors[loser].inputMask = self.predictors[loser].inputMask[:recombination_point] + self.predictors[winner].inputMask[recombination_point:]
			else:
				self.predictors[loser].inputMask = self.predictors[winner].inputMask

			if not FLAGS.disable_input_mutation:
				for i in range(len(self.predictors[loser].inputMask)):
					if random.uniform(0,1) < FLAGS.input_mutation_prob:
						if self.predictors[loser].inputMask[i] == 0:
							self.predictors[loser].inputMask[i] = 1
						else:
							self.predictors[loser].inputMask[i] = 0

			allocation = [a*1.0/FLAGS.num_predictors for a in self.problemsAllocation(self.problemsDistribution())]
			current_problem = self.predictors[loser].outputMask.index(1)
			problem_fraction = allocation[current_problem]

			# if there are more predictors for this output, increase the probability of output mask mutation
			if random.uniform(0,1) < (FLAGS.output_mutation_prob * (1 + problem_fraction)**FLAGS.punish_population_factor):
				self.predictors[loser].outputMask = [0]*World.state_size
				r = np.random.choice(range(World.state_size),p=distribution)
				self.predictors[loser].outputMask[r] = 1
				self.predictors[loser].problem=r
				if (FLAGS.initial_input == INITIAL_INPUT_CORRECT):
					self.predictors[loser].inputMask = World.correct_masks[r]

class Cumule():
		def __init__(self):
			self.world = World()
			self.agent = Agent()
			self.popFitHistory=np.ndarray((FLAGS.num_predictors,BACKTIME))*0
			self.timestep=0

		def plot_fitness(self,fig):
			popFit = []
			for i in range(FLAGS.num_predictors):
				popFit.append(self.agent.predictors[i].fitness)
			
			self.popFitHistory=np.roll(self.popFitHistory,-1,axis=1)
			self.popFitHistory[:,BACKTIME-1]=popFit

			fig.clear()
			for i in range(FLAGS.num_predictors):
				fig.plot(self.popFitHistory[i,:])

			x=self.timestep-(self.timestep%BACKTIME)
			x=range(x,self.timestep+(self.timestep%BACKTIME))
			fig.xaxis.set_ticks(np.arange(0, BACKTIME, 2.0))
			# fig.xaxis.set_ticklabels(x)
			xlabel('generations')
			ylabel('fitness')

		def test_archive(self, itime, compare_input_bits=False):
			plots, _, mse = test_distribution([[p] for p in self.agent.archive], FLAGS.test_set_length, self.agent, self.world,
											  range(self.world.state_size), plot_data=True, get_best=False)

			# use as many figures as necessary containing 8 plots each
			num_figures = (World.state_size+7)/8

			for fig_num in xrange(num_figures):
				figure()
				for i in xrange(0, 8):
					j = (fig_num*8)+i
					if j >= World.state_size:
						break
					subplot(4, 2, i)
					title("Problem #"+str(j))
					plot(plots[j,:,:])
				savefig("%sarchive_saved_%s.png" % (FLAGS.outputdir, fig_num))
				shutil.move("%sarchive_saved_%s.png" % (FLAGS.outputdir, fig_num),
							"%sarchive_saved_%s_part%s_%s.png" % (FLAGS.outputdir, itime, fig_num, FLAGS.outputdir[:-1]))

			if compare_input_bits:
				figure()
				num_errors = []
				nonzero = [a for a in xrange(World.state_size) if self.agent.archive[a] != 0]

				#print nonzero
				for non_0 in nonzero:
					# fraction of incorrect bits in input mask over the total number of bits
					diff = sum(list_diff(World.correct_masks[non_0], self.agent.archive[non_0].inputMask))*1.0/len(self.agent.archive[non_0].inputMask)
					#print i, World.correct_masks[i], self.agent.archive[i].inputMask, diff
					num_errors.append(diff)
				bar(nonzero, num_errors)
				savefig("%sarchive_wrong_input_fractions.png" % FLAGS.outputdir)
				shutil.move("%sarchive_wrong_input_fractions.png" % FLAGS.outputdir, "%swrong_input_fractions_%s_%s.png" % (FLAGS.outputdir, itime, FLAGS.outputdir[:-1]))

			return mse

		def archive_error(self, test_length, dims):
			_, _, mse = test_distribution([[p] for p in self.agent.archive], FLAGS.test_set_length, self.agent, self.world,
										  dims, plot_data=False, get_best=False)
			return mse

		def run(self, run_number, try_number):
			logfile=open(FLAGS.outputdir+FLAGS.logfile.replace(".log", "_%s_%s_%s.log" % (run_number, try_number, FLAGS.outputdir[:-1])),'w',1)
			errHis = []

			m = self.agent.getRandomMotor()
			s = self.world.updateState(m)

			archive_changed=False

			if FLAGS.show_plots:
				f=figure(figsize=(15,10))

			min_archive_error=1000

			for itime in range(PHASE_1_LENGTH):
				self.timestep+=1
				
				if self.timestep==FLAGS.timelimit+1 and FLAGS.timelimit!=-1:
					if min_archive_error==1000:
						return -1
					else:
						return min_archive_error
				elif FLAGS.timelimit==-1 and min_archive_error!=1000:
					return min_archive_error 

				logfile.write("Timestep:"+str(itime)+"\n")
				
				m = self.agent.getRandomMotor() 
				s = self.world.resetState(m)

				# Archive evaluating
				if self.agent.archive.count(0) < World.state_size:
					if archive_changed==True:
						new_error=self.test_archive(itime, FLAGS.check_input_mask)
						if self.agent.archive.count(0) == 0:
							if min_archive_error>new_error:
								logfile.write("New achieved archive error: "+str(new_error)+"\n")
								min_archive_error=new_error
						archive_changed=False

				distr=self.agent.problemsDistribution()

				# Check if there's a candidate solution in population
				if itime!=0:
					if FLAGS.train_error:
						bestEfforts=self.agent.minTrainErrors(distr)
					else:
						bestEfforts=self.agent.minTestErrors(distr, FLAGS.population_test_length, self.world, itime, logfile)
					for problem, error, predictor in bestEfforts:
						if error<FLAGS.archive_threshold:
							if self.agent.archive[problem]==0:
								#logfile.write("Problem "+str(problem)+" was successfully solved. Error: "+str(round(error,4))+"\n")
								logfile.write("Problem %s was successfully solved. Error: %s, inputMask %s\n" % (
										problem, round(error, 4), stringify_mask(predictor.inputMask)))
								self.agent.archive[problem]=predictor
								self.agent.predictors[self.agent.predictors.index(predictor)]=self.agent.createPredictor(problem, self.world)
								archive_changed=True

							else: 
								old_err=self.archive_error(FLAGS.test_set_length,[problem])
								old_predictor=self.agent.archive[problem]

								self.agent.archive[problem]=predictor
								new_err=self.archive_error(FLAGS.test_set_length,[problem])
								if new_err<old_err:
									#logfile.write("Problem "+str(problem)+" has a better solution. Archived test error: "+str(round(old_err,4))+". Better solution: "+str(round(new_err,4))+"\n")
									logfile.write("Problem %s has a better solution. Archived test error: %s, input %s. Better solution: %s, input %s\n" % (
											problem, round(old_err,4), stringify_mask(old_predictor.inputMask), round(new_err,4), stringify_mask(predictor.inputMask)))
									self.agent.predictors[self.agent.predictors.index(predictor)]=self.agent.createPredictor(problem, self.world)
									archive_changed=True
								else:
									logfile.write("Solution for %s remains the same. Archived test error: %s, inputMask %s. Candidate error %s, inputMask %s\n" % (
											problem, round(error, 4), stringify_mask(old_predictor.inputMask), round(new_err,4), stringify_mask(predictor.inputMask)))
									self.agent.archive[problem]=old_predictor


				# training of predictors
				for t in range(FLAGS.episode_length):#*********************************************

					m = self.agent.getRandomMotor()
					stp1 = self.world.updateState(m)
					inp = np.concatenate((s,m), axis = 0)
					self.agent.storeDataPoint(inp, stp1) 

					s = stp1

				self.agent.trainPredictors()
				self.agent.clearPredictorsData()

				distr=self.agent.problemsDistribution()
				errHis.append(self.agent.minimumErrors(distr, FLAGS.train_error))

				# don't store too much error history, for performance reasons
				if len(errHis) > BACKTIME:
					errHis = errHis[-BACKTIME:]

				if FLAGS.show_plots:
					#Plot the raw errors of the predictors in the population 
					# fig=subplot(2,3,3)
					# fig.clear()
					# bar(np.arange(0,World.state_size),self.agent.problemsMutationProbabilities(self.agent.problemsDistribution()))
					# xlabel("problem number")
					# ylabel("mutation probabilty")
					
					fig=subplot(2,3,1)
					fig.clear()
					title('Minimum %s errors on outputs' % ('train' if FLAGS.train_error else 'test'))
					plot(errHis[-BACKTIME:])
					xlabel('episodes(last '+str(BACKTIME)+')')
					ylabel('errors')

					# fig=subplot(2,3,2)
					# fig.clear()
					# title('Minimum errors on outputs')
					# plot(errHis)
					# xlabel('episodes(all time)')
					# ylabel('errors')

					fig=subplot(2,3,4)
					fig.clear()
					title('Solved problems(blue means solved)')
					barlist=bar(np.arange(0,World.state_size),[1 for k in self.agent.archive])
					for k,v in enumerate(self.agent.archive):
						if v==0:
							color='r'
							barlist[k].set_color(color)
					xlabel('problem')
					ylabel('archive')

					fig=subplot(2,3,5)
					fig.clear()
					bar(np.arange(0,World.state_size),self.agent.problemsAllocation(self.agent.problemsDistribution()))
					xlabel("Problem number")
					ylabel("Predictors")

					savefig("%sfigure1.png" % FLAGS.outputdir)
					shutil.move("%sfigure1.png" % FLAGS.outputdir, "%sfigure1_%s_%s_%s.png" % (FLAGS.outputdir, run_number, try_number, FLAGS.outputdir[:-1]))

					#draw()

				if (not FLAGS.disable_evolution) and (itime%FLAGS.evolution_period == 0):
					a = random.randint(0, FLAGS.num_predictors-1)
					b = random.randint(0, FLAGS.num_predictors-1)
					while(a == b):
						b = random.randint(0, FLAGS.num_predictors-1)

					if FLAGS.train_error:
						fit1 = self.agent.predictors[a].getTrainFitness(0) #0 = Fitness type Chrisantha, 1 = Fitness type Mai
						fit2 = self.agent.predictors[b].getTrainFitness(0)
					else:
						fit1 = self.agent.predictors[a].getTestFitness(0, 5, self.agent, self.world, rewardMinimal=True)
						fit2 = self.agent.predictors[b].getTestFitness(0, 5, self.agent, self.world, rewardMinimal=True)

					winner = None
					loser = None

					if self.agent.archive[self.agent.predictors[a].problem] != 0:
						fit1 *= FLAGS.punish_archive_factor
					if self.agent.archive[self.agent.predictors[b].problem] != 0:
						fit2 *= FLAGS.punish_archive_factor

					if fit1 > fit2:
						winner = a
						loser = b
					else:
						winner = b
						loser = a 

					if random.uniform(0,1) < FLAGS.predictor_mutation_prob:
						self.agent.copyAndMutatePredictor(winner, loser, self.agent.problemsMutationProbabilities(
								self.agent.problemsDistribution(), FLAGS.train_error))

					# fig=subplot(5,2,1)
					# self.plot_fitness(fig)

					#Plot which outputs are being predicted by each predictor, and what the errors are. 
					# outputTypes = []
					# for i in range(FLAGS.num_predictors):
					# 	outputTypes.append(self.agent.predictors[i].outputMask)
					# fig=subplot(5,2,3)
					# fig.clear()
					# # imshow(np.array(outputTypes).T)
					# bar(np.arange(0,World.state_size),self.agent.problemsAllocation(self.agent.problemsDistribution()))
					# xlabel("Problem number")
					# ylabel("Predictors")


			
			logfile.close()

def test_distribution(distr, test_set_length, test_agent, test_world, dims, plot_data=False, get_best=False):
	#Generate random initial motor command between -1 and 1.
	m = test_agent.getRandomMotor()
	#Generate initial state for this motor command, and all else zero.
	s = test_world.updateState(m)
	err=0

	plots=np.ndarray((test_world.state_size, test_set_length, 2))*0
	sum_errors = defaultdict(list)

	test_distrib_file = open(FLAGS.outputdir + "test_distrib_%s.log" % FLAGS.outputdir[:-1], 'a')

	for t in range(FLAGS.test_set_length):

		m = test_agent.getRandomMotor()
		stp1 = test_world.updateState(m)
		inp = np.concatenate((s,m), axis = 0)
		s = stp1

		for problem, predictors in enumerate(distr):
			if (problem in dims) and (len(predictors) != 0) and (predictors[0] != 0):
				# this is used for the archive, where there is at most 1 predictor per output
				predicted=predictors[0].predict_masked(inp)
				# to get the mean squared error
				err+=(abs(predicted[problem]-stp1[problem])/(abs(stp1[problem]) if FLAGS.relative_error else 1))

				# collect data for plotting purposes
				if plot_data:
					plots[problem, t]=[predicted[problem], stp1[problem]]

				# collect data to get the best predictor for an output
				if get_best:
					predictions = [p.predict_masked(inp) for p in predictors]
					test_distrib_file.write("problem %s, state %s\npredictions %s\n\n" % (problem, stp1, predictions))
					errors = [abs(stp1[problem] - p.predict_masked(inp)[problem])/(abs(stp1[problem]) if FLAGS.relative_error else 1) for p in predictors]
					errors = [e*1.0/test_set_length for e in errors]
					if len(sum_errors[problem]) == 0:
						sum_errors[problem] = errors
					else:
						for k in xrange(len(predictors)):
							sum_errors[problem][k] += errors[k]
	# set test errors on predictors
	for problem in sum_errors:
		for k in xrange(len(sum_errors[problem])):
			distr[problem][k].testError = sum_errors[problem][k]


	test_distrib_file.close()
	return plots, sum_errors, 0.5*err/test_set_length

global_errs=[]
def print_global_errs():
	print "Average: "+str(np.mean(global_errs))
	print "Standard deviation: "+str(np.std(global_errs))
	print "Min: "+str(np.min(global_errs))
	print "Max: "+str(np.max(global_errs))

if __name__ == '__main__':
	#ion()

	FLAGS=parser.parse_args()
	if FLAGS.outputdir != "":
		try:
			os.mkdir(FLAGS.outputdir)
		except:
			pass
		FLAGS.outputdir += "/"

	if FLAGS.old_world:
		World = OldWorld
	else:
		World = NewWorld

	parameters = open(FLAGS.outputdir + "parameters.log", 'w')
	parameters.write(" ".join(sys.argv))
	parameters.write("\n")

	fl=vars(FLAGS)
	for k in sorted(fl.iterkeys()):
		print k+": "+str(fl[k])
		parameters.write("%s: %s\n" % (k, fl[k]))

	parameters.close()

	avg=0.0


	for i in range(FLAGS.runs):
		c = Cumule()
		result=c.run(i, 0)
		tries=0
		while result==-1 and tries<10:
			result=c.run(i, tries) # we need to get those N runs
			tries+=1
		
		if result==-1:
			print "Couldn't find a solution for one of the runs in "+str(tries)+" tries. Something is clearly wrong."
		else:
			global_errs.append(result)
			print result

	print_global_errs()
	# if FLAGS.show_test_error:
	# 	c.test_archive()
	# 	raw_input("Press Enter to exit")

atexit.register(print_global_errs)
