import sys,random,time,math

NUM_MOTORS=2
NUM_DIMENSIONS=8

class World(): 
	state_size=NUM_DIMENSIONS
	action_size=NUM_MOTORS
	correct_masks = [[0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
					 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					 [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
					 [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
					 [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
					 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
					 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
					 [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]]

	def __init__(self):

		#Create world data structures 
		self.s = [0]*NUM_DIMENSIONS    #CURRENT STATE 
		self.stp1 = [0]*NUM_DIMENSIONS #TEMPORARY STATE.

	def resetState(self, m):
		self.s = [0]*NUM_DIMENSIONS    #CURRENT STATE 
		s = self.nextState(m)
		self.updateState()
		return self.getState()

	def nextState(self, m):

		#Update each state in this weird and impenetrable manner. 
		self.stp1[0] = (self.s[7] + m[0])/2.0
		self.stp1[1] = (self.s[0] + m[1])/2.0
		self.stp1[2] = (self.s[1] + m[0])/2.0
		self.stp1[3] = (self.s[2] + m[1])/2.0
		self.stp1[4] = (self.s[3] + m[0])/2.0
		self.stp1[5] = (self.s[4] + m[1])/2.0
		self.stp1[6] = (self.s[5] + m[0])/2.0
		self.stp1[7] = (self.s[6] + m[1])/2.0

		return self.stp1

	def updateState(self):
		self.s = self.stp1[:]

	def getState(self):
		return self.s

	def getRandomMotor(self):
		return [random.uniform(0,1), random.uniform(0,1)]
