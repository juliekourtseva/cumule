import sys,random,time,math
import world

NUM_MOTORS=2
NUM_DIMENSIONS=50

from world import World

class OldWorld(World): 
	state_size=NUM_DIMENSIONS
	action_size=NUM_MOTORS

	def __init__(self):

		#Create world data structures 
		self.s =  [random.uniform(0,1) for i in range(NUM_DIMENSIONS)]    #CURRENT STATE 
		self.stp1 =  [random.uniform(0,1) for i in range(NUM_DIMENSIONS)] #TEMPORARY STATE. 

	def resetState(self, m):
		self.s =  [random.uniform(0,1) for i in range(NUM_DIMENSIONS)]    #CURRENT STATE 
		return self.s

	def updateState(self, m):
		#Update each state in this weird and impenetrable manner. 
		self.s = self.stp1[:]
		self.stp1[0] = math.cos(math.pi*5* m[0])
		self.stp1[1] = (self.s[0] + self.s[3])/2
		self.stp1[2] = math.cos(math.pi*self.s[4] ) #Is there a mistake here Mai?
		self.stp1[3] = math.cos(math.pi* m[0])* math.cos(math.pi* m[1])
		self.stp1[4] = (m[0]+m[1])/2
		self.stp1[5] = pow(m[0]+m[1],2);
		self.stp1[6] = 0.5
		self.stp1[7] = pow(0.7*m[1]+0.3*self.s[0],3)
		self.stp1[8] = math.exp(- pow(m[0]-0.5,2)/(2*pow(10,-2)))
		self.stp1[9] = math.exp(- pow(m[0]-0.7,2)/(2*pow(10,-3))) - math.exp(- pow(m[1]-0.1,2)/(2*pow(10,-4)))
		self.stp1[10] = math.cos(math.pi*5* m[0])
		self.stp1[11] = (self.s[10] + self.s[13])/2
		self.stp1[12] = math.cos(math.pi*self.s[14] ) #Is there a mistake here Mai?
		self.stp1[13] = math.cos(math.pi* m[0])* math.cos(math.pi* m[1])
		self.stp1[14] = (m[0]+m[1])/3
		self.stp1[15] = pow(m[0]+m[1],3);
		self.stp1[16] = 1
		self.stp1[17] = pow(0.3*m[1]+0.7*self.s[10],3)
		self.stp1[18] = math.exp(- pow(m[0]-0.5,2)/(4*pow(10,-2)))
		self.stp1[19] = math.exp(- pow(m[0]-0.7,2)/(2*pow(10,-3))) - math.exp(- pow(m[1]-0.1,2)/(2*pow(10,-4)))
		self.stp1[20] = math.cos(math.pi*5* m[0])
		self.stp1[21] = (self.s[20] + self.s[23])/2
		self.stp1[22] = math.cos(math.pi*self.s[24] ) #Is there a mistake here Mai?
		self.stp1[23] = math.cos(math.pi* m[0])* math.cos(math.pi* m[1])
		self.stp1[24] = (m[0]+m[1])/4
		self.stp1[25] = pow(m[0]+m[1],4);
		self.stp1[26] = 0.1
		self.stp1[27] = pow(0.5*m[1]+0.5*self.s[20],4)
		self.stp1[28] = math.exp(- pow(m[0]-0.5,2)/(2*pow(10,-2)))
		self.stp1[29] = math.exp(- pow(m[0]-0.7,2)/(2*pow(10,-3))) - math.exp(- pow(m[1]-0.1,2)/(2*pow(10,-4)))
		self.stp1[30] = math.cos(math.pi*5* m[0])
		self.stp1[31] = (self.s[30] + self.s[33])/2
		self.stp1[32] = math.cos(math.pi*self.s[34] ) #Is there a mistake here Mai?
		self.stp1[33] = math.cos(math.pi* m[0])* math.cos(math.pi* m[1])
		self.stp1[34] = (m[0]+m[1])/5
		self.stp1[35] = pow(m[0]+m[1],5);
		self.stp1[36] = 0.7
		self.stp1[37] = pow(0.7*m[1]+0.3*self.s[30],5)
		self.stp1[38] = math.exp(- pow(m[0]-0.5,2)/(2*pow(10,-2)))
		self.stp1[39] = math.exp(- pow(m[0]-0.7,2)/(2*pow(10,-3))) - math.exp(- pow(m[1]-0.1,2)/(2*pow(10,-4)))
		self.stp1[40] = math.cos(math.pi*5* m[0])
		self.stp1[41] = (0.3*self.s[40] + 0.7*self.s[43])/2
		self.stp1[42] = math.cos(math.pi*self.s[44] ) #Is there a mistake here Mai?
		self.stp1[43] = math.cos(math.pi* m[0])* math.cos(math.pi* m[1])
		self.stp1[44] = (m[0]+m[1])/2
		self.stp1[45] = pow(m[0]+m[1],2);
		self.stp1[46] = 0.5
		self.stp1[47] = pow(0.7*m[1]+0.3*self.s[40],3)
		self.stp1[48] = math.exp(- pow(m[0]-0.5,2)/(2*pow(10,-2)))
		self.stp1[49] = math.exp(- pow(m[0]-0.7,2)/(2*pow(10,-3))) - math.exp(- pow(m[1]-0.1,2)/(2*pow(10,-4)))

		return self.stp1

	def getState(self):
		return self.s

	def getRandomMotor(self):
		return [random.uniform(0,1) for i in range(NUM_MOTORS)]
