import unittest
import numpy as np
from structure_probabilities import StructureProbabilities

def round_list_of_lists(list_of_lists, places=4):
	for index, l in enumerate(list_of_lists):
		list_of_lists[index] = [round(x, places) for x in l]
	return list_of_lists

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

	def test_get_mu_sigma(self):
		# 5 problems, 2 hidden layers
		sp = StructureProbabilities(5, 2, [5, 0], [4, 1])
		for problem, probs in enumerate(sp.distributions):
			self.assertEqual(sp.get_mu_sigma(problem, 0), [5, 4])
			self.assertEqual(sp.get_mu_sigma(problem, 1), [0, 1])

		sp.distributions[2][0] = [4, 3]
		self.assertEqual(sp.get_mu_sigma(2, 0), [4, 3])
		self.assertEqual(sp.get_mu_sigma(2, 1), [0, 1])
		self.assertEqual(sp.get_mu_sigma(3, 0), [5, 4])

	def test_set_mu_sigma(self):
		sp = StructureProbabilities(5, 2, [5, 0], [4, 1])
		self.assertEqual(sp.get_mu_sigma(2, 0), [5, 4])
		self.assertEqual(sp.get_mu_sigma(2, 1), [0, 1])

		sp.set_mu_sigma(2, 0, 4, 3)
		self.assertEqual(sp.get_mu_sigma(2, 0), [4, 3])
		self.assertEqual(sp.get_mu_sigma(2, 1), [0, 1])

		self.assertEqual(sp.get_mu_sigma(3, 0), [5, 4])
		sp.set_mu_sigma(2, 1, 1, 1)
		self.assertEqual(sp.get_mu_sigma(2, 1), [1, 1])

	def test_update_probability(self):
		# 5 problems, 2 hidden layers
		sp = StructureProbabilities(5, 2, [5, 0], [4, 1])
		# move away from this structure
		new_mu, new_sigma = sp.update_probability(6, 5, 4, False)
		assert(new_mu == 4.0)
		assert(round(new_sigma*1e4)/1.0e4 == 4.0396)

		# move towards this structure
		new_mu, new_sigma = sp.update_probability(6, 5, 4, True)
		assert(new_mu == 6.0)
		assert(round(new_sigma*1e4)/1.0e4 == 3.9223)

	def test_update_probabilities(self):
		# 5 problems, 2 hidden layers
		sp = StructureProbabilities(5, 2, [5, 0], [4, 1])
		# move away from this structure
		sp.update_probabilities((6, 2), 2, False)
		for i, probs in enumerate(sp.distributions):
			if i != 2:
				self.assertEqual(probs, [[5, 4], [0, 1]])
		self.assertEqual(round_list_of_lists(sp.distributions[2]), [[4.0, 4.0396], [0, 1.0050]])

		# move away from this structure again
		sp.update_probabilities((6, 2), 2, False)
		for i, probs in enumerate(sp.distributions):
			if i != 2:
				self.assertEqual(probs, [[5, 4], [0, 1]])
		self.assertEqual(round_list_of_lists(sp.distributions[2]), [[2.0, 4.0597], [0, 1.0100]])

		# move back towards it
		sp.update_probabilities((6, 2), 2, True)
		for i, probs in enumerate(sp.distributions):
			if i != 2:
				self.assertEqual(probs, [[5, 4], [0, 1]])
		self.assertEqual(round_list_of_lists(sp.distributions[2]), [[6.0, 4.0396], [2, 1.0]])

	def test_get_sample(self):
		# 5 problems, 2 hidden layers
		sp = StructureProbabilities(5, 2, [5, 0], [4, 1])
		sample = sp.get_sample(2, 0)
		self.assertTrue(type(sample) == int)
		self.assertTrue(sample > 0)

		sp = StructureProbabilities(5, 2, [5, 0], [4, 1])
		sample = sp.get_sample(2, 1)
		self.assertTrue(type(sample) == int)
		self.assertTrue(sample >= 0)

if __name__ == '__main__':
	unittest.main()
