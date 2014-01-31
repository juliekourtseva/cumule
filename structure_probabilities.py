import math

class StructureProbabilities(object):
	def __init__(self, num_problems, num_hidden_layers, default_means=[5, 0], default_sds=[4, 1]):
		if len(default_means) < num_hidden_layers:
			raise Exception("Not enough default values for mean and standard deviation")
		self.distributions = []
		means_sds = [list(x) for x in zip(default_means, default_sds)]
		for i in xrange(num_problems):
			hl_probs = []
			for j in xrange(num_hidden_layers):
				hl_probs.append(means_sds[j])
			self.distributions.append(hl_probs)

	def get_mu_sigma(self, problem, hidden_layer):
		return [None, None]

	def set_mu_sigma(self, problem, hidden_layer):
		return [None, None]

	def update_probabilities(self, structure, problem, change_towards):
		pass

	def update_probability(self, value, mu, sigma, change_towards):
		return [None, None]

	def get_sample(self, problem, layer):
		return None
