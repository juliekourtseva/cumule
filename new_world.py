import sys,random,time,math
import numpy as np
from numpy.random import RandomState
from random import shuffle

import sys

#FFNN supervised learning packages 
# from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
# from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer, LinearLayer
# from pybrain.tools.validation import ModuleValidator

# Author : Nguyen Sao Mai
# nguyensmai@gmail.com
# nguyensmai.free.fr
NUMO1 = 1
NUMO2 = 3
NUMI1 = 3
NUMI2 = 5
NUM_MOTORS = 2
NUM_DIMENSIONS = 3*NUMO1 + 4*NUMO2 

class World(): 
    
    state_size=NUM_DIMENSIONS
    action_size=NUM_MOTORS
    correct_masks = [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					 [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
					 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
					 [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					 [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					 [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					 [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					 [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					 [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    def __init__(self):
        #Create world data structures 
        self.s = [0]*NUM_DIMENSIONS    #CURRENT STATE 
        self.stp1 = [0]*NUM_DIMENSIONS #TEMPORARY STATE. 
        #Create function sructures
        self.f1 = buildNetwork(NUMI1, 2, NUMO1, hiddenclass=TanhLayer, bias=True)
        self.f2 = buildNetwork(NUMI2, 50, NUMO2, hiddenclass=TanhLayer, bias=True)
        #self.indI = np.random.randint(0,NUM_DIMENSIONS+NUM_MOTORS, size=3*NUMI1+4*NUMI2)
        #self.indO = range(3*NUMO1+4*NUMO2);
        #shuffle(self.indO)



    def resetState(self, m):
        self.s = [0]*NUM_DIMENSIONS    #CURRENT STATE 
        s = self.updateState(m)
        return s


    def updateState(self, m):
        initInput = 0
        initOutput = 0
        dimI1 = self.f1['in'].dim
        dimI2 = self.f2['in'].dim
        dimO1 = self.f1['out'].dim
        dimO2 = self.f2['out'].dim
        output=[]
        self.s =self.stp1
        self.stp1=[]

        #input    = m+self.s[0:1]+m+self.s[1:2]+m+self.s[2:3]
        input    = m+self.s[0:NUMI1-2]+m+self.s[NUMI1-2:2*NUMI1-4]+m+self.s[2*NUMI1-4:3*NUMI1-6]
        #for i in self.indI:
        #    input.append(sm[i])
        for i in range(3):
            output[initOutput:initOutput+dimO1] = self.f1.activate(input[initInput:initInput+dimI1])
            initInput += dimI1;
            initOutput += dimO1;

	    #output[0] = self.f1.activate(input[0:3] = m+self.s[0])
	    #output[1] = self.f1.activate(input[3:6] = m+self.s[1])
	    #output[2] = self.f1.activate(input[6:9] = m+self.s[2])

		# NUMO1 = 1
		# NUMO2 = 3
		# NUMI1 = 3
		# NUMI2 = 5
		# NUM_MOTORS = 2
		# NUM_DIMENSIONS = 3*NUMO1 + 4*NUMO2

        # dimI1 = self.f1['in'].dim = 3
        # dimI2 = self.f2['in'].dim = 5
        # dimO1 = self.f1['out'].dim = 1
        # dimO2 = self.f2['out'].dim = 3

        initInput=0;
        #input    = m+self.s[-3:]+m+self.s[-6:-3]+m+self.s[-9:-6]+m+self.s[-12:-9]
		#input    = m+self.s[12:15]m+self.s[9:12]+m+self.s[6:9]+m+self.s[3:6]
        input    = m+self.s[-NUMI2+2:]+m+self.s[-2*NUMI2+4:-NUMI2+2]+m+self.s[-3*NUMI2+6:-2*NUMI2+4]+m+self.s[-4*NUMI2+8:-3*NUMI2+6]
        for i in range(4):
            output[initOutput:initOutput+dimO2] = self.f2.activate(input[initInput:initInput+dimI2])
            initInput += dimI2
            initOutput += dimO2;

	    #output[3:6] = self.f2.activate(input[0:5] = m+self.s[12:15])
	    #output[6:9] = self.f2.activate(input[5:10] = m+self.s[9:12])
	    #output[9:12] = self.f2.activate(input[10:15] = m+self.s[6:9])
	    #output[12:15] = self.f2.activate(input[15:20] = m+self.s[3:6])

        for i in range(NUM_DIMENSIONS):
            self.stp1.append(output[i])
        
        return self.stp1
    
    def getRandomMotor(self):
        return [random.uniform(0,1), random.uniform(0,1)]
    

    def getState(self):
        return self.s

