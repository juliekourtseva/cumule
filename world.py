import sys,random,time,math

NUM_MOTORS=2
NUM_DIMENSIONS=8

class World(): 
	state_size=NUM_DIMENSIONS
	action_size=NUM_MOTORS

	def __init__(self):

		#Create world data structures 
		self.s = [0]*NUM_DIMENSIONS    #CURRENT STATE 
		self.stp1 = [0]*NUM_DIMENSIONS #TEMPORARY STATE.

	def resetState(self, m):
		self.s = [random.uniform(0,1) for x in NUM_DIMENSIONS]    #CURRENT STATE 
		stp1 = self.nextState(m)
		s = self.updateState()
		return s

	def nextState(self, m):
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

		# #Set s to s(t+1)
		# for i in range(NUM_DIMENSIONS):
		# 	self.s[i] = self.stp1[i]

		return self.stp1

	def input_masks(self):
		return [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
				[0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
				[0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
				[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
				[0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
				[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 1, 1, 0, 0, 0, 1, 0]]

	def updateState(self):
		self.s = self.stp1[:]

	def getState(self):
		return self.s

	def getRandomMotor(self):
		return [random.uniform(0,1), random.uniform(0,1)]
