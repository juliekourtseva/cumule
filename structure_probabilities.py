import math
import random

class StructureProbabilities(object):
	def __init__(self, num_problems, num_hidden_layers, default_means=[5, 0], default_sds=[4, 1], diff_factor=0.3, diff_limit=1.5):
		self.diff_factor = diff_factor
		# diff limit in terms of standard deviations
		self.diff_limit = diff_limit
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
		return self.distributions[problem][hidden_layer]

	def set_mu_sigma(self, problem, hidden_layer, mu, sigma):
		self.distributions[problem][hidden_layer] = [mu, sigma]

	def update_probabilities(self, structure, problem, is_winner):
		for hl, num_units in enumerate(structure):
			mu, sigma = self.get_mu_sigma(problem, hl)
			new_mu, new_sigma = self.update_probability(num_units, mu, sigma, is_winner)
			self.set_mu_sigma(problem, hl, new_mu, new_sigma)

	def update_probability(self, num, mu, sigma, is_winner):
		diff = num-mu
		signdiff = -1 if (diff < 0) else 1
		if is_winner:
			diff_sd = min(self.diff_factor*abs(diff), sigma*self.diff_limit)
			new_mu = mu + (signdiff*diff_sd)
			new_sigma = sigma/(1 + 2.0/((100*diff_sd)+1))
		else:
			diff_sd = max(self.diff_factor*abs(diff), sigma*self.diff_limit)
			new_mu = mu - (signdiff*diff_sd)
			new_sigma = sigma*(1 + 1.0/((100*diff_sd)+1))
		if new_mu < 0:
			new_mu = 0
		return new_mu, new_sigma

	def get_sample(self, problem, layer):
		mu, sigma = self.get_mu_sigma(problem, layer)
		sample = int(math.floor(random.gauss(mu, sigma)))
		min_sample = 1 if layer == 0 else 0
		if sample < min_sample:
			sample = min_sample
		return sample
