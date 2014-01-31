import sys

def get_masks(lines, num_dim, num_motors):
	mappings = {}
	for d in xrange(num_dim):
		mappings["self.s[%s]" % d] = [d]

	for d in xrange(num_motors):
		mappings["m[%s]" % d] = [d + num_dim]

	mappings["state"] = range(num_dim)
	masks = [[] for x in xrange(num_dim)]
	for l in lines:
		try:
			output = int(l.split("=")[0].replace("self.stp1", "").strip("\t []"))
		except:
			continue
		functions = l.split("=")[1]
		inputs = set()
		for m in mappings:
			if m in functions:
				for i in mappings[m]:
					inputs.add(i)
		print output, inputs
		masks[output]=[int(x in inputs) for x in range(num_dim+num_motors)]
	return masks

if __name__ == '__main__':
	num_dim = int(sys.argv[2])
	num_motors = int(sys.argv[3])
	lines = []

	with open(sys.argv[1], 'r') as worldfile:
		data = worldfile.read()
		data = data.split("def updateState")[1].split("return")[0]
		lines = filter(lambda x: "self.stp1" in x and "=" in x, data.split("\n"))	

	masks = get_masks(lines, num_dim, num_motors)
	print "["
	for m in masks:
		print ",".join([str(x) for x in m])+","
	print "]"
