import pickle,sys,pprint,random,time,math
from collections import deque
import numpy as np
from numpy.random import RandomState
import pickle
from copy import copy,deepcopy
import argparse

import sys

#FFNN supervised learning packages 
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer, LinearLayer
from pybrain.tools.validation import ModuleValidator


#5th January 2014. Cumule Algorithm (Chrisantha Fernando)

PHASE_1_LENGTH = 100000

WEIGHT_DECAY=0.1
BACKTIME=10
PREDICTOR_MUTATION_PROBABILITY=0.8

parser = argparse.ArgumentParser()
parser.add_argument("timelimit",default=50,type=int)
parser.add_argument("-n","--num_predictors",help="population size(default:50)",default=50,type=int)
parser.add_argument("--runs",help="number of runs(default:1)",default=1,type=int)
parser.add_argument("--epochs",help="number of epochs for each training(default:5)",default=5,type=int)
parser.add_argument("-ts","--test_set_length",help="test set length(default:50)",default=50,type=int)
parser.add_argument("-e","--evolution_period", help="evolution period(default:10)", type=int, default=10)
# parser.add_argument("-a","--archive_threshold", help="threshold for getting into the archive(default: 0.02)", type=float, default=0.02)
parser.add_argument("-lr","--learning_rate", help="learning rate for predictors(default: 0.01)", type=float, default=0.01)
parser.add_argument("-r","--replication", help="enable weights replication(default: no)",action="store_true", default=False)
parser.add_argument("-lg","--logfile", help="log file name(default: prediction.log)",type=str, default="prediction.log")
parser.add_argument("-s","--statsname", help="stats file name(default: )",type=str, default="")
parser.add_argument("-i","--mutate_input", help="enable input mask mutation(default: yes)",action="store_true", default=True)
parser.add_argument("--episode_length", help="number of samples per episode(default: 50)",action="store_true", default=10)
parser.add_argument("--show_test_error", help="test archive and show the plot", action="store_true",default=False)
parser.add_argument("--show_plots", help="show live plots", action="store_true",default=False)
parser.add_argument("--sliding_training", help="use sliding window of examples", action="store_true",default=False)
parser.add_argument("--input_mutation_prob", help="input mutation probability per bit(default: 0.05)", type=float, default=0.05)
parser.add_argument("--output_mutation_prob", help="output mutation probability per mask(default: 0.9)", type=float, default=0.9)
parser.add_argument("--replication_prob", help="weight copy probability per weight(default: 0.1)", type=float, default=0.1)
parser.add_argument("--predictor_mutation_prob", help="tournament loser mutation probability(default: 1)", type=float, default=1.0)
parser.add_argument("--small_networks", help="use small networks(default: False)", action="store_true",default=False)
parser.add_argument("--random_input_masks", help="initialise population with random input masks(default:False)", action="store_true",default=False)
parser.add_argument("--fixed_hidden_layer",default=None,type=int)
parser.add_argument("--world_file", type=str, default=None)
parser.add_argument("--test_name",type=str)
parser.add_argument("--input_bit",type=int, default=0)

from old_world import OldWorld
from new_world import NewWorld

FLAGS={}

WORLD_STATE_SIZE=0
WORLD_ACTION_SIZE=0



class Predictor(): 

	def __init__(self, inSize, hiddenSize, outSize, LearningRate):

		self.learning_rate = LearningRate

		if FLAGS.random_input_masks:
			self.inputMask = [random.randint(0,1) for i in range(inSize)]
		else:
			#self.inputMask = [1 for i in range(inSize)]
			self.inputMask = [0]*inSize
			self.inputMask[FLAGS.input_bit] = 1

		self.inSize=self.inputMask.count(1)

		if self.inSize==0:
			self.inputMask[random.randint(0,inSize)]=1
			self.inSize=1	

		if FLAGS.small_networks:
			self.outSize=1
		else:
			self.outSize=outSize

		if FLAGS.fixed_hidden_layer:
			hiddenSize=FLAGS.fixed_hidden_layer

		self.ds = SupervisedDataSet(self.inSize, self.outSize)
		self.net = buildNetwork(self.inSize, hiddenSize, self.outSize, hiddenclass=TanhLayer, bias=True)
		self.trainer = BackpropTrainer(self.net, self.ds, learningrate=self.learning_rate, verbose = False, weightdecay=WEIGHT_DECAY)



		self.prediction = [0] * outSize
		self.mse = 100
		self.age=0

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
	def setProblem(self,problem):
		self.outputMask[self.problem]=0
		self.outputMask[problem]=1
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

		self.error = e

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

	def getFitness(self):
		return self.fitness 

	def storeDataPoint(self, inputA, targetA):
		self.ds.addSample(self.prepareInput(inputA), self.prepareTarget(targetA))

	def prepareTarget(self,targetA):
		if FLAGS.small_networks:
			return targetA[self.problem]
		else:
			target=[0 for i in range(len(targetA))]
			target[self.problem]=1
			return target



	def prepareInput(self,raw):
		if FLAGS.small_networks:
		    inputA = []
		    for j in range(len(raw)):
		            if self.inputMask[j]==1:
		                    inputA.append(raw[j])
		else:
			inputA=[]
			for j in range(len(raw)):
		            if self.inputMask[j]==1:
		                    inputA.append(raw[j])
		            else:
		            		inputA.append(0)
		return inputA

	def predict(self,inputA):
		return self.net.activate(self.prepareInput(inputA))


class Agent(): 
		def __init__(self):

			#The agent has a population of M predictors. 
			self.predictors = []
			self.archive=[0 for i in range(WORLD_STATE_SIZE)]
			for i in range(FLAGS.num_predictors):
				p=Predictor(WORLD_STATE_SIZE + WORLD_ACTION_SIZE, WORLD_STATE_SIZE + WORLD_ACTION_SIZE,WORLD_STATE_SIZE, FLAGS.learning_rate)
				self.predictors.append(p)

		def problemsDistribution(self):
			r=[[] for i in range(WORLD_STATE_SIZE)]
			for predictor in self.predictors:
				r[predictor.problem].append(predictor)
			return r

		def averageErrors(self,distr):
			r=[]

			for problem, predictors in enumerate(distr):
				error=np.mean([p.error for p in predictors])
				r.append(error)	

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
				self.predictors[i].storeDataPoint(inp, targ)

		def trainPredictors(self):
			ep = []
			for i in range(FLAGS.num_predictors):
				e = self.predictors[i].trainPredictor()
				ep.append(e)
			return ep

		def createPredictor(self,hiddenLayerSize,problem):
			p=Predictor(WORLD_STATE_SIZE + WORLD_ACTION_SIZE, WORLD_STATE_SIZE + WORLD_ACTION_SIZE ,WORLD_STATE_SIZE, FLAGS.learning_rate)
			p.problem=problem
			p.outputMask = [0]*WORLD_STATE_SIZE
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

			if FLAGS.fixed_hidden_layer!=None:
				hiddenSize=FLAGS.fixed_hidden_layer
			else:
				if FLAGS.small_networks:
					hiddenSize=newLoser.inSize+1
				else:
					hiddenSize=WORLD_ACTION_SIZE+WORLD_STATE_SIZE

			if FLAGS.small_networks:
				newLoser.learning_rate =  FLAGS.learning_rate
				newLoser.ds = SupervisedDataSet(newLoser.inSize, 1)
				newLoser.net = buildNetwork(newLoser.inSize,hiddenSize,1, bias=True)
				newLoser.trainer = BackpropTrainer(newLoser.net, newLoser.ds, learningrate=newLoser.learning_rate, verbose = False, weightdecay=WEIGHT_DECAY)
			else:
				self.predictors[loser].learning_rate =  FLAGS.learning_rate
				self.predictors[loser].ds = SupervisedDataSet(WORLD_STATE_SIZE+WORLD_ACTION_SIZE, WORLD_STATE_SIZE)
				self.predictors[loser].net = buildNetwork(WORLD_STATE_SIZE+WORLD_ACTION_SIZE,hiddenSize,WORLD_STATE_SIZE, bias=True)
				self.predictors[loser].trainer = BackpropTrainer(self.predictors[loser].net, self.predictors[loser].ds, learningrate=self.predictors[loser].learning_rate, verbose = False, weightdecay=WEIGHT_DECAY)


			if FLAGS.replication:
			        for i in range(len(newLoser.net.params)):
			                if random.uniform(0,1)<FLAGS.replication_prob:
			                        newLoser.net.params[i] = self.predictors[winner].net.params[i]


			if random.uniform(0,1) < FLAGS.output_mutation_prob:
			        newLoser.outputMask = [0]*WORLD_STATE_SIZE
			        r=random.choice(range(WORLD_STATE_SIZE))
			        newLoser.outputMask[r] = 1
			        newLoser.problem=r

			self.predictors[loser]=newLoser



class Cumule():
		def __init__(self, world_instance=None, run_n=1):

			if not world_instance:
				self.world = World()
			else:
				self.world=deepcopy(world_instance)

			self.popFitHistory=np.ndarray((FLAGS.num_predictors,BACKTIME))*0
			self.timestep=0
			self.run_n=run_n
			self.predictor_statsfile=open(FLAGS.statsname+'_predictors.csv','a',1)
			self.archive_statsfile=open(FLAGS.statsname+'_archive.csv','a',1)

		def predictors_stats(self):
			for i,p in enumerate(self.agent.predictors):
				self.predictor_statsfile.write("{run};{timestep};{predictor_number};{problem};{error};{inputmask};{hiddenlayer}\n".format(run=self.run_n, timestep=self.timestep,
																																predictor_number=i, problem=p.problem,
																																error=-p.fitness, inputmask="".join([str(k) for k in p.inputMask]),
																																hiddenlayer=p.net['hidden0'].dim
																																	))
		def archive_stats(self):
			if self.timestep>0 and (self.timestep % 3)==0:
				self.archive_statsfile.write("{run};{timestep};{min_error}\n".format(run=self.run_n,timestep=self.timestep,min_error=self.testSolution(self.getSolution(),FLAGS.test_set_length)))

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

		def testSolution(self,solution,test_set_length,test_set=None):
			m = self.agent.getRandomMotor()
			s = self.world.updateState(m)
			err=0

			dims=[]
			for k,v in enumerate(solution):
				if v!=None:
					dims.append(k)


			for t in range(test_set_length):
				m = self.agent.getRandomMotor()
				stp1 = self.world.updateState(m)
				inp = np.concatenate((s,m), axis = 0)

				s = stp1

				predicted=np.ndarray(WORLD_STATE_SIZE)
				expected=stp1
	
				for i in dims:
					if not FLAGS.small_networks:
						predicted[i]=solution[i].predict(inp)[i]
					else:
						predicted[i]=solution[i].predict(inp)
					err+=(predicted[i]-expected[i])**2

			return 0.5*err/test_set_length

		def setFitnesses(self, test_set):
			errs=[0.0]*FLAGS.num_predictors
			for sample in test_set:
				inp=sample[0]
				outp=sample[1]

				for num,p in enumerate(self.agent.predictors):
					if not FLAGS.small_networks:
						predicted=p.predict(inp)[p.problem]
					else:
						predicted=p.predict(inp)[0]
					errs[num]+=(predicted-outp[p.problem])**2

			errs=np.multiply(errs,-0.5/len(test_set))

			for num,p in enumerate(self.agent.predictors):
				p.fitness=errs[num]



		def run(self): 
			self.timestep=0
			self.agent = Agent()

			logfile=open(FLAGS.logfile,'w',1)
			errHis = []

			m = self.agent.getRandomMotor()
			s = self.world.resetState(m)

			for i in range(PHASE_1_LENGTH):
				self.timestep+=1
				
				if self.timestep==FLAGS.timelimit+1 and FLAGS.timelimit!=-1:
					return self.testSolution(self.getSolution(),FLAGS.test_set_length)

				logfile.write("Timestep:"+str(i)+"\n")
				
				distr=self.agent.problemsDistribution()
				
				# training of predictors
				test_set=[]
				for t in range(FLAGS.episode_length):#*********************************************

					m = self.agent.getRandomMotor()
					stp1 = self.world.updateState(m)
					inp = np.concatenate((s,m), axis = 0)
					self.agent.storeDataPoint(inp, stp1) 
					test_set.append((inp,stp1))

					s = stp1

				self.setFitnesses(test_set)

				self.agent.trainPredictors()
				self.agent.clearPredictorsData()

				distr=self.agent.problemsDistribution()
				# errHis.append(self.agent.minimumErrors(distr))



				if i%FLAGS.evolution_period == 0:
					a = random.randint(0, FLAGS.num_predictors-1)
					b = random.randint(0, FLAGS.num_predictors-1)
					while(a == b):
						b = random.randint(0, FLAGS.num_predictors-1)

					fit1  = self.agent.predictors[a].fitness #0 = Fitness type Chrisantha, 1 = Fitness type Mai 
					fit2  = self.agent.predictors[b].fitness

					winner = None
					loser = None

					if fit1 > fit2:
						winner = a
						loser = b
					else:
						winner = b
						loser = a
					
					if random.uniform(0,1) < FLAGS.predictor_mutation_prob and (len(distr[self.agent.predictors[loser].problem])>1):
						self.agent.copyAndMutatePredictor(winner, loser, self.agent.problemsMutationProbabilities(self.agent.problemsDistribution()))

				self.predictors_stats()
				self.archive_stats()

			
			logfile.close()
			self.statsfile.close()


if __name__ == '__main__':
	# ion()

	FLAGS=parser.parse_args()
	
	st=open(FLAGS.statsname+'_predictors.csv',"w")
	st.write("Run; Timestep; Predictor Number; Problem; Error; Input Mask; Hidden Layer Size\n")
	st.close()

	st=open(FLAGS.statsname+'_archive.csv',"w")
	st.write("Run; Timestep; MinError\n")
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






