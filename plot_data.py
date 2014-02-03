from matplotlib.pyplot import *
from collections import defaultdict
import os
import sys
import math

def plot_solution(sol_file):
	#"{run};{problem};{expected};{predicted};{example}
	plots = defaultdict(list)
	with open(sol_file, 'r') as f:
		data = f.readlines()
	data = [d.split(";") for d in data[1:] if ";" in d]
	data = [[float(x) for x in d] for d in data]
	for d in data:
		_, problem, expected, predicted, _ = d
		plots[problem].append([predicted, expected])

	problems = [int(k) for k in plots.keys()]
	problems.sort()
	num_figures = (len(problems)+7)/8
	for fig_num in xrange(num_figures):
		figure()
		for i in xrange(0, 8):
			try:
				prob = problems[(fig_num*8)+i]
			except:
				break
			subplot(4, 2, i)
			title("Problem #"+str(prob))
			try:
				plot(plots[prob])
			except Exception, e:
				print "Failed to plot graphs:", e.message
			savefig("%s.png" % (sol_file.replace(".csv", "_%s" % fig_num)))

def plot_structure(struct_file):
	#"{run};{timestep};{problem};{hidden_layer};{mu};{sigma}
	plots_split = defaultdict(dict)
	with open(struct_file, 'r') as f:
		data = f.readlines()
	data = [d.split(";") for d in data[1:] if ";" in d]
	data = [[float(x) for x in d] for d in data]
	timesteps = []
	max_hidden_layer = None
	for d in data:
		_, timestep, problem, hidden_layer, mu, sigma = d
		#print timestep, problem, hidden_layer, mu, sigma
		if (max_hidden_layer is None) or (max_hidden_layer < hidden_layer):
			max_hidden_layer = hidden_layer
		timesteps.append(timestep)
		try:
			plots_split[problem][timestep][int(hidden_layer)] = [mu, sigma]
		except KeyError:
			plots_split[problem][timestep] = {int(hidden_layer):[mu, sigma]}

	plots = defaultdict(list)
	for problem in plots_split:
		time_and_stats = [[], []]
		sorted_timesteps = plots_split[problem].keys()
		sorted_timesteps.sort()
		for timestep in sorted_timesteps:
			time_and_stats[0].append(timestep)
			hl_stats = [None]*(int(max_hidden_layer)*4)
			for hl in plots_split[problem][timestep]:
				hl_stats[hl*2] = plots_split[problem][timestep][hl][0]
				hl_stats[(hl*2)+1] = plots_split[problem][timestep][hl][1]
			time_and_stats[1].append(hl_stats)
		plots[problem] = (time_and_stats)

	problems = [int(k) for k in plots.keys()]
	problems.sort()
	num_figures = (len(problems)+7)/8
	for fig_num in xrange(num_figures):
		figure()
		for i in xrange(0, 8):
			try:
				prob = problems[(fig_num*8)+i]
			except:
				break
			subplot(4, 2, i)
			title("Problem #"+str(prob))
			try:
				plot(plots[prob][0], plots[prob][1])
				ylim((-1, max([max(y) for y in plots[prob][1]])+1))
			except Exception, e:
				print "Failed to plot graphs:", e.message
			savefig("%s.png" % (struct_file.replace(".csv", "_%s" % fig_num)))

def plot_archive(arch_file):
	#{run};{timestep};{problem};{error};{fitness};{structure}
	err_plots = defaultdict(dict)
	struct_plots = defaultdict(dict)
	with open(arch_file, 'r') as f:
		data = f.readlines()
	data = [d.split(";") for d in data[1:] if ";" in d]
	for index, d in enumerate(data):
		data[index] = [float(x) for x in data[index][:-1]] + [data[index][-1]]
	max_hidden_layers = None
	max_num_units = None
	min_num_units = None
	for d in data:
		_, timestep, problem, error, _, structure = d
		if error > 0:
			err_plots[problem][timestep] = math.log(error, 10)
		else:
			err_plots[problem][timestep] = -323
		exec "structure = %s" % structure
		if (max_hidden_layers is None) or (len(structure) > max_hidden_layers):
			max_hidden_layers = len(structure)
		struct_plots[problem][timestep] = structure
		for s in structure:
			if (max_num_units is None) or (s > max_num_units):
				max_num_units = s
			if (min_num_units is None) or (s < min_num_units):
				min_num_units = s

	err_plots_sorted = defaultdict(list)
	for problem in err_plots:
		time_sorted = err_plots[problem].keys()
		time_sorted.sort()
		plots = [time_sorted, []]
		for t in time_sorted:
			plots[1].append(err_plots[problem][t])
		err_plots_sorted[problem] = plots

	struct_plots_sorted = defaultdict(list)
	for problem in struct_plots:
		time_sorted = struct_plots[problem].keys()
		time_sorted.sort()
		plots = [time_sorted, []]
		for t in time_sorted:
			structure = list(struct_plots[problem][t])
			for i in xrange(max_hidden_layers - len(structure)):
				structure.append(0)
			plots[1].append(structure)
		struct_plots_sorted[problem] = plots
		#print plots

	problems = [int(k) for k in err_plots_sorted.keys()]
	problems.sort()
	num_figures = (len(problems)+7)/8
	for fig_num in xrange(num_figures):
		figure()
		for i in xrange(0, 8):
			try:
				prob = problems[(fig_num*8)+i]
			except:
				break
			subplot(4, 2, i)
			title("Problem #"+str(prob))
			try:
				plot(err_plots_sorted[prob][0], err_plots_sorted[prob][1])
			except Exception, e:
				print "Failed to plot graphs:", e.message
			savefig("%s.png" % (arch_file.replace(".csv", "_%s_errors" % fig_num)))

		figure()
		for i in xrange(0, 8):
			try:
				prob = problems[(fig_num*8)+i]
			except:
				break
			subplot(4, 2, i)
			title("Problem #"+str(prob))
			ylim((-1, max_num_units+1))
			try:
				plot(struct_plots_sorted[prob][0], struct_plots_sorted[prob][1])
				ylim((-1, max([max(y) for y in struct_plots_sorted[prob][1]])+1))
			except Exception, e:
				print "Failed to plot graphs:", e.message
			savefig("%s.png" % (arch_file.replace(".csv", "_%s_struct" % fig_num)))

outputdir = sys.argv[1]

csvs = [f for f in os.listdir(outputdir) if f.endswith(".csv")]
print csvs

solution = None
archive = None
structure = None
predictors = None

for f in csvs:
	if f.endswith("_solution.csv"):
		solution = outputdir + "/" + f
	elif f.endswith("_archive.csv"):
		archive = outputdir + "/" + f
	elif f.endswith("_predictors.csv"):
		predictors = outputdir + "/" + f
	elif f.endswith("_structure.csv"):
		structure = outputdir + "/" + f

if solution is not None:
	plot_solution(solution)

if structure is not None:
	plot_structure(structure)

if archive is not None:
	plot_archive(archive)

