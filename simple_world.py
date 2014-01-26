import sys,random,time,math

NUM_MOTORS=2
NUM_DIMENSIONS=3

class World(): 
	state_size=NUM_DIMENSIONS
	action_size=NUM_MOTORS
	correct_masks = [[0, 1, 1, 0, 0],
					 [0, 0, 1, 1, 1],
					 [1, 1, 0, 0, 1]]

	def __init__(self):

		#Create world data structures 
		self.s = [0]*NUM_DIMENSIONS    #CURRENT STATE 
		self.stp1 = [0]*NUM_DIMENSIONS #TEMPORARY STATE.

	def resetState(self, m):
		self.s = [0]*NUM_DIMENSIONS    #CURRENT STATE 
		s = self.updateState(m)
		return s

	def updateState(self, m):
		self.stp1[0] = (self.s[1] + self.s[2])/2.0
		self.stp1[1] = (m[0] + self.s[2] + m[1])/3.0
		self.stp1[2] = (self.s[0] + self.s[1] + m[1])/3.0

		#Set s to s(t+1)
		for i in range(NUM_DIMENSIONS):
			self.s[i] = self.stp1[i]

		return self.s

	def getState(self):
		return self.s

	def getRandomMotor(self):
		return [random.uniform(0,1), random.uniform(0,1)]
