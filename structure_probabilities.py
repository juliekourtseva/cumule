class StructureProbabilities(object):
	def __init__(self, num_problems, num_hidden_layers, default_means=[5, 0], default_sds=[4, 1]):
		if len(default_means) < len(num_hidden_layers):
			default_means = default_means + [0]*(num_hidden_layers-len(default_means))
			default_sds = default_sds + [1]*(num_hidden_layers-len(default_sds))
		self.distribution = []
		means_sds = [list(x) for x in zip(default_means, default_sds)]
		for i in num_problems:
			hl_probs = []
			for j in num_hidden_layers:
				hl_probs.append(means_sds[j])
			self.distribution.append(hl_probs)

	def update_probability(self, structure, problem, change_towards):
		# For moving away from a number of hidden units (n),
		# change mean by sd/(abs(mean-n)+1)*(sign(mean-n) depending on change_towards),
		# increase sd.
		# For moving towards it, change mean by sd/(abs(mean-n)+1)*sign(mean-n)
		# and decrease sd
		pass

