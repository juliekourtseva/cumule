import sys,random,time,math
import numpy as np
from numpy.random import RandomState
from random import shuffle
from copy import copy

import sys

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, LinearLayer

from world import World

# Author : Nguyen Sao Mai
# nguyensmai@gmail.com
# nguyensmai.free.fr
NUMO1 = 1
NUMO2 = 3
NUMI1 = 3
NUMI2 = 5
NUM_MOTORS = 2
NUM_DIMENSIONS = 3*NUMO1 + 4*NUMO2 

class NewWorld(World): 
    
    state_size=NUM_DIMENSIONS
    action_size=NUM_MOTORS
    
    def __init__(self):
        self.s = [random.uniform(0,1) for i in range(NUM_DIMENSIONS)]    #CURRENT STATE 
        self.f1 = buildNetwork(NUMI1, 2, NUMO1, hiddenclass=TanhLayer, bias=True)
        self.f2 = buildNetwork(NUMI2, 50, NUMO2, hiddenclass=TanhLayer, bias=True)


    def resetState(self, m):
        self.s = [random.uniform(0,1) for i in range(NUM_DIMENSIONS)]    #CURRENT STATE 
        s = self.updateState(m)
        return s


    def updateState(self, m):
        initInput = 0;
        initOutput = 0;
        dimI1 = self.f1['in'].dim;
        dimI2 = self.f2['in'].dim;
        dimO1 = self.f1['out'].dim;
        dimO2 = self.f2['out'].dim;
        output=[0]*NUM_DIMENSIONS
        old_state=copy(self.s)

        input    = m+old_state[0:NUMI1-2]+m+old_state[NUMI1-2:2*NUMI1-4]+m+old_state[2*NUMI1-4:3*NUMI1-6];

        #for i in self.indI:
        #    input.append(sm[i])
        for i in range(3):
            output[initOutput:initOutput+dimO1] = self.f1.activate(input[initInput:initInput+dimI1])
            initInput += dimI1;
            initOutput += dimO1;

        initInput=0;
        input = m+old_state[-NUMI2+2:]+m+old_state[-2*NUMI2+4:-NUMI2+2]+m+old_state[-3*NUMI2+6:-2*NUMI2+4]+m+old_state[-4*NUMI2+8:-3*NUMI2+6];
        for i in range(4):
            output[initOutput:initOutput+dimO2] = self.f2.activate(input[initInput:initInput+dimI2])
            initInput += dimI2
            initOutput += dimO2;

        for i in range(NUM_DIMENSIONS):
            self.s[i]=(output[i])
        
        return self.s

    def inputMask(self,indexes):
        r=[0]*(NUM_DIMENSIONS+NUM_MOTORS)
        for k in indexes:
            r[k]=1
        return r

    def inputMasks(self):
        self.inputMask([])
    
    def getRandomMotor(self):
        return [random.uniform(0,1), random.uniform(0,1)]

    def getState(self):
        return self.s
