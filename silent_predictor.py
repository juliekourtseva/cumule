import pylab,pickle,sys,pprint,random,time,math
from collections import deque
import numpy as np
from numpy.random import RandomState
import pickle
from copy import copy,deepcopy
from matplotlib.pyplot import *
import argparse

import sys

#FFNN supervised learning packages 
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer, LinearLayer
from pybrain.tools.validation import ModuleValidator

#5th January 2014. Cumule Algorithm (Chrisantha Fernando)

NUM_DIMENSIONS = 8 
NUM_MOTORS = 2
PHASE_1_LENGTH = 100000

# EVOLUTION_PERIOD = 2 #Evolve predictors every 10 episodes. 
WEIGHT_DECAY=0.1
# MUTATE_MASK_PROBABILITY = 0.9
BACKTIME=10
PREDICTOR_MUTATION_PROBABILITY=0.8
# WEIGHT_COPY_PROBABILITY=0.05

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
parser.add_argument("-i","--mutate_input", help="enable input mask mutation(default: yes)",action="store_true", default=True)
parser.add_argument("--episode_length", help="number of samples per episode(default: 50)",action="store_true", default=10)
parser.add_argument("--show_test_error", help="test archive and show the plot", action="store_true",default=False)
parser.add_argument("--show_plots", help="show live plots", action="store_true",default=False)
parser.add_argument("--sliding_training", help="use sliding window of examples", action="store_true",default=False)
parser.add_argument("--input_mutation_prob", help="input mutation probability per bit(default: 0.05)", type=float, default=0.05)
parser.add_argument("--output_mutation_prob", help="output mutation probability per mask(default: 0.9)", type=float, default=0.9)
parser.add_argument("--replication_prob", help="weight copy probability per weight(default: 0.1)", type=float, default=0.1)
parser.add_argument("--predictor_mutation_prob", help="tournament loser mutation probability(default: 1)", type=float, default=1.0)



class World(): 
		def __init__(self):

			#Create world data structures 
			self.s = [0]*NUM_DIMENSIONS    #CURRENT STATE 
			self.stp1 = [0]*NUM_DIMENSIONS #TEMPORARY STATE. 

		def resetState(self, m):
			self.s = [0]*NUM_DIMENSIONS    #CURRENT STATE 
			s = self.updateState(m)
			return s

		def updateState(self, m):

			#Update each state in this weird and impenetrable manner. 
			self.stp1[0] = math.cos(self.s[0] + m[0])
			self.stp1[1] = math.cos(self.s[1] + m[1])
			p0 = pow(self.s[0],2)
			p1 = pow(self.s[1],2)
			p2 = pow(self.s[2],2)
			p3 = pow(self.s[3],2)
			p4 = pow(self.s[4],2)
			self.stp1[2] = math.cos(p1 + p3 + p4 + pow(m[1],2) ) #Is there a mistake here Mai?
			self.stp1[3] = math.cos(self.s[0] + self.s[1])
			self.stp1[4] = math.cos(m[0] + m[1])
			self.stp1[5] = p2 + p3 + p4
			self.stp1[6] = p0 + p1 + p2
			self.stp1[7] = pow(m[0], 2) + p3 + p4

			#Set s to s(t+1)
			for i in range(NUM_DIMENSIONS):
				self.s[i] = self.stp1[i]

			return self.s

		def getState(self):
			return self.s

		def getRandomMotor(self):
			return [random.uniform(0,1), random.uniform(0,1)]

class Predictor(): 

	def __init__(self, inSize, outSize, LearningRate):

		self.inputMask = [random.randint(0, 1) for i in range(inSize)]
		self.inSize=self.inputMask.count(1)

		if self.inSize==0:
			self.inputMask[random.randint(0,inSize)]=1
			self.inSize=1
		
		self.outSize=1

		self.learning_rate = LearningRate
		self.ds = SupervisedDataSet(self.inSize, self.outSize)
		self.net = buildNetwork(self.inSize, self.inSize+1, self.outSize, hiddenclass=TanhLayer, bias=True)
		self.trainer = BackpropTrainer(self.net, self.ds, learningrate=self.learning_rate, verbose = False, weightdecay=WEIGHT_DECAY)
		self.prediction = [0] * self.outSize
		self.mse = 100
		self.age=0

		#Specific to Mai's code. Make input and output masks.  
		
#		self.outputMask = [random.randint(0, 1) for i in range(outSize)]
		self.outputMask = [0]*outSize
		r = random.randint(0,outSize-1)
		self.outputMask[r] = 1

		self.error = 0
		self.errorHistory = []
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
		self.error = e
		#Entire error history
		if len(self.errorHistory) < 5:  
			self.errorHistory.append(e)
		else:
			for i in range(len(self.errorHistory)-1):
				self.errorHistory[i] = self.errorHistory[i+1]
			self.errorHistory[-1] = e

		#Sliding window error over appeox last 10 episodes characturistic time. 
		self.slidingError = self.slidingError*0.9 + self.error
		#Instantaneous difference in last er ror between episodes. 
		if len(self.errorHistory) > 1:
			self.dError = self.errorHistory[-1] - self.errorHistory[-2] 

		return e

	def getFitness(self, type):

		fit = 0 
		#Fitness function 1 Chrisantha's attempt 
		if type == 0:#SIMPLE MINIMIZE PREDICTION ERROR FITNESS FUNCTION FOR PREDICTORS. 
#           fit = -self.dError/(1.0*self.error)
			fit = -self.error
		elif type == 1:
			#Fitness function 2 Mai's attempt (probably need to use adaptive thresholds for this to be ok)
			if self.error > ERROR_THRESHOLD and self.dError > DERROR_THRESHOLD:
				fit = 0
			else:
				fit = 1

		self.fitness = fit
		return fit 

	def storeDataPoint(self, inputA, targetA):
		self.ds.addSample(inputA, targetA)

	def getInput(self,raw):
		inputA = []
		for j in range(len(raw)):
			if self.inputMask[j]==1:
				inputA.append(raw[j])
		return inputA

	def predict(self,inputA):
		r=[0]*len(self.inputMask)
		r[self.problem]=self.net.activate(self.getInput(inputA))
		return r


class Agent(): 
		def __init__(self):

			#The agent has a population of M predictors. 
			self.predictors = []
			self.archive=[0 for i in range(NUM_DIMENSIONS)]
			for i in range(FLAGS.num_predictors):
				p=Predictor(NUM_DIMENSIONS + NUM_MOTORS,NUM_DIMENSIONS, FLAGS.learning_rate)
				self.predictors.append(p)

		def problemsDistribution(self):
			r=[[] for i in range(NUM_DIMENSIONS)]
			for predictor in self.predictors:
				r[predictor.problem].append(predictor)
			return r

		def minimumErrors(self,distr):
			r=[]

			for problem, predictors in enumerate(distr):
				if len(predictors)>0:
					error=min([p.error for p in predictors])
				else:
					error=5

				r.append(error)	


			return r

		# execute this AFTER storing into archive and BEFORE new training
		def problemsMutationProbabilities(self,distr):
			r=[]
			min_err=1000000
			max_err=-1000000

			for problem, predictors in enumerate(distr):
				if len(predictors)!=0:
					err=np.mean([p.error for p in predictors])
				else:
					err=-1

				if self.archive[problem]!=0:
					err=err*2 # we discourage agent from generating predictors that solve already solved problems
				
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
										

		
		def minErrors(self,distr):
			r=[]
			for problem, predictors in enumerate(distr):
				if len(predictors)!=0:
					errors=[p.error for p in predictors]
					best=np.argmin(errors)
					err=errors[best]

					r.append((problem,err,predictors[best]))	
			return r

		def problemsAllocation(self,distr):
			return [len(predictors) for predictors in distr]


		def getRandomMotor(self):
			return [random.uniform(0,1), random.uniform(0,1)]
	
		def storeDataPoint(self, inp, targ):

			for i in range(FLAGS.num_predictors):
				#APPLY INPUT AND OUTPUT MASKS BEFORE SENDING DATA TO PREDICTORS. 
				inputA = []
				for j in range(len(inp)):
					if self.predictors[i].inputMask[j]==1:
						inputA.append(inp[j])
				# target = [0]*len(targ)
				# for j in range(len(targ)):
				# 	target[j] = targ[j]*self.predictors[i].outputMask[j]

				self.predictors[i].storeDataPoint(inputA, targ[self.predictors[i].problem])

		def trainPredictors(self):
			ep = []
			for i in range(FLAGS.num_predictors):
				e = self.predictors[i].trainPredictor()
				ep.append(e)
			return ep

		def createPredictor(self,hiddenLayerSize,problem):
			p=Predictor(NUM_DIMENSIONS + NUM_MOTORS,NUM_DIMENSIONS, FLAGS.learning_rate)
			p.problem=problem
			p.outputMask = [0]*NUM_DIMENSIONS
			p.outputMask[problem]=1

			return p



		def clearPredictorsData(self):
			for i in range(FLAGS.num_predictors):
				self.predictors[i].ds.clear()

		def copyAndMutatePredictor(self, winner, loser,distribution):
			newLoser = deepcopy(self.predictors[winner])

			if FLAGS.mutate_input:
				for i in range(len(newLoser.inputMask)):
					if random.uniform(0,1) < FLAGS.input_mutation_prob:
						if newLoser.inputMask[i] == 0:
							newLoser.inputMask[i] = 1
						else:
							newLoser.inputMask[i] = 0
				
				if newLoser.inputMask.count(1)==0:
					newLoser.inputMask[random.randint(0,len(newLoser.inputMask))]=1
				
				newLoser.inSize=newLoser.inputMask.count(1)



			newLoser.learning_rate =  FLAGS.learning_rate
			newLoser.ds = SupervisedDataSet(newLoser.inSize, 1)
			newLoser.net = buildNetwork(newLoser.inSize,newLoser.inSize+1,1, bias=True)
			newLoser.trainer = BackpropTrainer(newLoser.net, newLoser.ds, learningrate=newLoser.learning_rate, verbose = False, weightdecay=WEIGHT_DECAY)

			
			if FLAGS.replication:
				for i in range(len(newLoser.net.params)):
					if random.uniform(0,1)<FLAGS.replication_prob:
						newLoser.net.params[i] = self.predictors[winner].net.params[i]
			

			if random.uniform(0,1) < FLAGS.output_mutation_prob:
				newLoser.outputMask = [0]*NUM_DIMENSIONS
				r = np.random.choice(range(NUM_DIMENSIONS),p=distribution)
				newLoser.outputMask[r] = 1
				newLoser.problem=r

			self.predictors[loser]=newLoser

class Cumule():
		def __init__(self):

			self.world = World()
			self.agent = Agent()
			self.popFitHistory=np.ndarray((FLAGS.num_predictors,BACKTIME))*0
			self.timestep=0

		def test_archive(self):
			plots=np.ndarray((NUM_DIMENSIONS,FLAGS.test_set_length,2))*0

			#Generate random initial motor command between -1 and 1. 
			m = self.agent.getRandomMotor()
			#Geneate initial state for this motor command, and all else zero. 
			s = self.world.updateState(m)
			
			for t in range(FLAGS.test_set_length):#*********************************************

				m = self.agent.getRandomMotor()
				stp1 = self.world.updateState(m)
				inp = np.concatenate((s,m), axis = 0)
				s = stp1
	
				for i in range(NUM_DIMENSIONS):
					predicted=self.agent.archive[i].predict(inp)
					expected=stp1
					plots[i,t]=[predicted[i], expected[i]]
			
			figure()
			for i in range(NUM_DIMENSIONS):
				subplot(4,2,i)
				title("Problem #"+str(i))
				plot(plots[i,:,:])
			show()

		def archive_error(self,test_length,dims):
			m = self.agent.getRandomMotor()
			s = self.world.updateState(m)
			err=0


			for t in range(test_length):#*********************************************
				m = self.agent.getRandomMotor()
				stp1 = self.world.updateState(m)
				inp = np.concatenate((s,m), axis = 0)

				s = stp1

				predicted=np.ndarray(NUM_DIMENSIONS)
				expected=stp1
	
				for i in dims:
					predicted[i]=self.agent.archive[i].predict(inp)[i]
					err+=(predicted[i]-expected[i])**2

			return 0.5*err/test_length




		def run(self): 

			logfile=open("prediction.log",'w',1)
			errHis = []

			m = self.agent.getRandomMotor()
			s = self.world.updateState(m)

			archive_changed=False

			if FLAGS.show_plots:
				f=figure(figsize=(15,10))

			min_archive_error=1000

			for i in range(PHASE_1_LENGTH):
				self.timestep+=1
				
				if self.timestep==FLAGS.timelimit+1 and FLAGS.timelimit!=-1:
					if min_archive_error==1000:
						return -1
					else:
						return min_archive_error
				elif FLAGS.timelimit==-1 and min_archive_error!=1000:
					return min_archive_error 

				logfile.write("Timestep:"+str(i)+"\n")
				
				m = self.agent.getRandomMotor() 
				s = self.world.resetState(m)


				# Archive evaluating
				if self.agent.archive.count(0)==0:				
					if archive_changed==True:
						new_error=self.archive_error(FLAGS.test_set_length, range(NUM_DIMENSIONS))
						if min_archive_error>new_error:
							logfile.write("New achieved archive error: "+str(new_error)+"\n")
							logfile.write("Input sizes of new archive:"+", ".join([str(p.inSize) for p in self.agent.archive])+"\n")
							logfile.write("Input masks: \n")
							for p in self.agent.archive:
								logfile.write(",".join(map(str,p.inputMask))+"\n")
							min_archive_error=new_error
					
					archive_changed=False

				distr=self.agent.problemsDistribution()
				
				# Check if there's a candidate solution in population
				if i!=0:
					bestEfforts=self.agent.minErrors(distr)
					for problem, error, predictor in bestEfforts:
						if error<FLAGS.archive_threshold:
							if self.agent.archive[problem]==0:
								logfile.write("Problem "+str(problem)+" was successfully solved. Error: "+str(round(error,4))+"\n")
								self.agent.archive[problem]=predictor
								self.agent.predictors[self.agent.predictors.index(predictor)]=self.agent.createPredictor(10,problem)
								archive_changed=True

							else: 
								old_err=self.archive_error(FLAGS.test_set_length,[problem])
								old_predictor=self.agent.archive[problem]

								self.agent.archive[problem]=predictor
								new_err=self.archive_error(FLAGS.test_set_length,[problem])
								if new_err<old_err:
									logfile.write("Problem "+str(problem)+" has a better solution. Archived test error: "+str(round(old_err,4))+". Better solution: "+str(round(new_err,4))+"\n")
									self.agent.predictors[self.agent.predictors.index(predictor)]=self.agent.createPredictor(10,problem)
									archive_changed=True
								else:
									logfile.write("Solution for "+str(problem)+" remains the same. Archived test error: "+str(round(old_err,4))+". Candidate solution: "+str(round(new_err,4))+"\n")
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
				errHis.append(self.agent.minimumErrors(distr))


				if FLAGS.show_plots:
					#Plot the raw errors of the predictors in the population 
					# fig=subplot(2,3,3)
					# fig.clear()
					# bar(np.arange(0,NUM_DIMENSIONS),self.agent.problemsMutationProbabilities(self.agent.problemsDistribution()))
					# xlabel("problem number")
					# ylabel("mutation probabilty")
					
					fig=subplot(2,3,1)
					fig.clear()
					title('Minimum errors on outputs')
					plot(errHis[-BACKTIME:])
					xlabel('episodes(last '+str(BACKTIME)+')')
					ylabel('errors')

					fig=subplot(2,3,2)
					fig.clear()
					title('Minimum errors on outputs')
					plot(errHis)
					xlabel('episodes(all time)')
					ylabel('errors')

					fig=subplot(2,3,4)
					fig.clear()
					title('Solved problems(blue means solved)')
					barlist=bar(np.arange(0,NUM_DIMENSIONS),[1 for k in self.agent.archive])
					for k,v in enumerate(self.agent.archive):
						if v==0:
							color='r'
							barlist[k].set_color(color)
					xlabel('problem')
					ylabel('archive')

					fig=subplot(2,3,5)
					fig.clear()
					bar(np.arange(0,NUM_DIMENSIONS),self.agent.problemsAllocation(self.agent.problemsDistribution()))
					xlabel("Problem number")
					ylabel("Predictors")

					draw()

				if i%FLAGS.evolution_period == 0:
					a = random.randint(0, FLAGS.num_predictors-1)
					b = random.randint(0, FLAGS.num_predictors-1)
					while(a == b):
						b = random.randint(0, FLAGS.num_predictors-1)

					fit1  = self.agent.predictors[a].getFitness(0) #0 = Fitness type Chrisantha, 1 = Fitness type Mai 
					fit2  = self.agent.predictors[b].getFitness(0)

					winner = None
					loser = None

					if fit1 > fit2:
						winner = a
						loser = b
					else:
						winner = b
						loser = a 

					if random.uniform(0,1) < FLAGS.predictor_mutation_prob:
						self.agent.copyAndMutatePredictor(winner, loser, self.agent.problemsMutationProbabilities(self.agent.problemsDistribution()))

					# fig=subplot(5,2,1)
					# self.plot_fitness(fig)

					#Plot which outputs are being predicted by each predictor, and what the errors are. 
					# outputTypes = []
					# for i in range(FLAGS.num_predictors):
					# 	outputTypes.append(self.agent.predictors[i].outputMask)
					# fig=subplot(5,2,3)
					# fig.clear()
					# # imshow(np.array(outputTypes).T)
					# bar(np.arange(0,NUM_DIMENSIONS),self.agent.problemsAllocation(self.agent.problemsDistribution()))
					# xlabel("Problem number")
					# ylabel("Predictors")


			
			logfile.close()


if __name__ == '__main__':
	ion()

	FLAGS=parser.parse_args()


	fl=vars(FLAGS)
	for k in sorted(fl.iterkeys()):
		print k+": "+str(fl[k])

	avg=0.0
	errs=[]

	for i in range(FLAGS.runs):
		c = Cumule()
		result=c.run()
		tries=0
		while result==-1 and tries<10:
			result=c.run() # we need to get those N runs
			tries+=1
		
		if result==-1:
			print "Couldn't find a solution for one of the runs in "+str(tries)+" tries. Something is clearly wrong."
		else:
			errs.append(result)
			print result
	print "Average: "+str(np.mean(errs))
	print "Standard deviation: "+str(np.std(errs))
	print "Min: "+str(np.min(errs))
	print "Max: "+str(np.max(errs))

	if FLAGS.show_test_error:
		c.test_archive()
		raw_input("Press Enter to exit")





