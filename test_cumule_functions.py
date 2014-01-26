import unittest

from silent_predictor import Predictor, Agent, Cumule
from world import World

INSIZE = 10
OUTSIZE = 8
LEARNING_RATE = 0.01

class TestAgentFunctions(unittest.TestCase):
	def test_output_mutation_multiplier(self):
		predictors = [Predictor(INSIZE, OUTSIZE, LEARNING_RATE, 2, 10,
								"ones", World.correct_masks) for p in xrange(48)]
		agent = Agent(predictors)
		# all the predictors have the same output
		for p in agent.predictors:
			p.outputMask = [0]*OUTSIZE
			p.problem = 2

		for p in agent.predictors:
			self.assertEqual(agent.outputMutationMultiplier(
					OUTSIZE, p.problem, 48, 5), 32.0)

		# half the predictors have the same output
		for p in xrange(24):
			agent.predictors[p].outputMask = [0]*OUTSIZE
			agent.predictors[p].problem = 1

		for p in agent.predictors:
			self.assertEqual(agent.outputMutationMultiplier(
					OUTSIZE, p.problem, 48, 5), 7.59375)

		# 1/4 have 3, 1/4 have 1, 1/2 have 2
		for p in xrange(12):
			agent.predictors[p].outputMask = [0]*OUTSIZE
			agent.predictors[p].problem = 3

		for p in xrange(24):
			self.assertEqual(agent.outputMutationMultiplier(
					OUTSIZE, agent.predictors[p].problem, 48, 5), 1.25**5)

		for p in xrange(24, 48):
			self.assertEqual(agent.outputMutationMultiplier(
					OUTSIZE, agent.predictors[p].problem, 48, 5), 1.5**5)

		agent.predictors[24].problem = 4
		for p in xrange(24):
			self.assertEqual(agent.outputMutationMultiplier(
					OUTSIZE, agent.predictors[p].problem, 48, 5), 1.25**5)

		for p in xrange(25, 48):
			self.assertEqual(agent.outputMutationMultiplier(
					OUTSIZE, agent.predictors[p].problem, 48, 5), (1 + 23.0/48)**5)

		self.assertEqual(agent.outputMutationMultiplier(
					OUTSIZE, 4, 48, 5), 0)

		agent.predictors[25].problem = 4
		self.assertEqual(agent.outputMutationMultiplier(
					OUTSIZE, 4, 48, 5), (1 + (2.0/48))**5)

if __name__ == '__main__':
	unittest.main()
