import unittest
import numpy as np
from structure_probabilities import StructureProbabilities

class TestStructureProbabilities(unittest.TestCase):
	def test_init(self):
		# 5 problems, 2 hidden layers
		sp = StructureProbabilities(5, 2)
		self.assertEqual(len(sp.distributions), 5)
		for i in sp.distributions:
			self.assertEqual(i, [[5, 4], [0, 1]])

		# 5 problems, 3 hidden layers, new defaults
		sp = StructureProbabilities(5, 3, default_means=[4, 2, 1], default_sds=[3, 1, 1])
		self.assertEqual(len(sp.distributions), 5)
		for i in sp.distributions:
			self.assertEqual(i, [[4, 3], [2, 1], [1, 1]])

	def test_update_probability(self):
		# 5 problems, 2 hidden layers
		sp = StructureProbabilities(5, 2)
		# move away from this structure
		self.update_probability((5, 1), 1, False)
		for i, probs in enumerate(sp.distributions):
			if i != 1:
				self.assertEqual(probs, [[5, 4], [0, 1]])
		self.assertEqual(sp.distributions[i], [[5 + 4/3.0, 4], [0, 0.8]])
		
		

if __name__ == '__main__':
	unittest.main()
