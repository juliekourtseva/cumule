import pylab,pickle,sys,pprint,random,time,math
from collections import deque
import numpy as np
from numpy.random import RandomState
import pickle
from copy import copy,deepcopy
from matplotlib.pyplot import *

#FFNN supervised learning packages 
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer, LinearLayer
from pybrain.tools.validation import ModuleValidator

from silent_predictor import *

#5th January 2014. Cumule Algorithm (Chrisantha Fernando)



class ArrayOfFFNNs:
	def __init__(self, networks=1):
		self.world=World()
		self.nets=[buildNetwork(NUM_DIMENSIONS+NUM_MOTORS,10,NUM_DIMENSIONS) for i in range(networks)]
		self.ds=[SupervisedDataSet(NUM_DIMENSIONS+NUM_MOTORS,NUM_DIMENSIONS) for i in range(networks)]

		self.trainers=[BackpropTrainer(self.nets[k], self.ds[k], learningrate=FLAGS.learning_rate, verbose = False, weightdecay=WEIGHT_DECAY) for k in range(networks)]
		self.networks=networks


	def run(self):
		m = self.world.getRandomMotor()
		s_t = self.world.updateState(m)

		for i in range(FLAGS.timelimit):
			for n in range(self.networks):
				self.ds[n].clear()

			for t in range(FLAGS.episode_length):#*********************************************

				#Execute random motor command 
				m = self.world.getRandomMotor()
				#Get s_t(t+1) after executing this motor command in the self.world 
				s_tp1 = self.world.updateState(m)
				#Store the data in each predictors memory 
				inp = np.concatenate((s_t,m), axis = 0)
				s_t = s_tp1
				for n in range(self.networks):
					self.ds[n].addSample(inp, s_tp1)

			for n in range(self.networks):
				for j in range(FLAGS.epochs):
					self.trainers[n].train()

		self.plots=np.ndarray((self.networks,NUM_DIMENSIONS,FLAGS.test_set_length,2))*0

		errs=[0 for i in range(self.networks)]

		for t in range(FLAGS.test_set_length):#*********************************************
			#Execute random motor command 
			m = self.world.getRandomMotor()
			#Get s(t+1) after executing this motor command in the self.world 
			stp1 = self.world.updateState(m)
			#Store the data in each predictors memory 
			inp = np.concatenate((s_t,m), axis = 0)
			s = stp1

			for n in range(self.networks):
				predicted=self.nets[n].activate(inp)
				expected=stp1

				errs[n]+=0.5*sum(np.power(predicted-expected,2))
				
				for i in range(NUM_DIMENSIONS):
					self.plots[n,i,t]=[predicted[i], expected[i]]

		return np.divide(errs,FLAGS.test_set_length)
		
	def show_test_error(self,n):
		figure()
		for i in range(NUM_DIMENSIONS):
			subplot(4,2,i)
			plot(self.plots[n,i,:,:])
		show()

if __name__=='__main__':

	parser.add_argument('--networks',type=int, default=1, help="number of networks(default:1")
	FLAGS=parser.parse_args()
	
	fl=vars(FLAGS)
	for k in sorted(fl.iterkeys()):
		print k+": "+str(fl[k])

	errs=[]

	print "Minimum; Average; Maximum"
	for i in range(FLAGS.runs):
		networks = ArrayOfFFNNs(FLAGS.networks)
		results=networks.run()
		# errs.append(result)
		print str(min(results))+";"+str(np.mean(results))+";"+str(max(results))
	# print "Average: "+str(np.mean(errs))
	# print "Standard deviation: "+str(np.std(errs))
	# print "Min: "+str(np.min(errs))
	# print "Max: "+str(np.max(errs))

	# if FLAGS.show_test_error:
	# 	c.show_test_error()
	# 	raw_input("Press Enter to exit")