# Testing out my idea about changing the probability distribution
# based on new values and their correctness

from matplotlib.pyplot import *
import random
import math

optimal = 5
threshold = 0.01
mu = 5
sd = 3

def error(num):
	return abs(optimal-num)*1.0/(optimal+threshold)

def update_probs(num, fitness, towards):
	global mu
	global sd
	# if the current mean is bad, move away from it
	# and increase sd in a way that's inversely proportional to distance of num from new mean
	# if the current mean is good, stay at it
	# and decrease sd in a way that's inversely proportional to distance of num from new mean
	diff = num-mu
	sign = -1 if (diff < 0) else 1
	abs_diff = abs(diff)
	print "mu", mu, "value", num
	if towards:
		diff_sd = max(abs_diff, sd*0.1)
		print "towards", abs_diff, sd, diff_sd, sign
		mu += sign*diff_sd
		sd /= (1 + 2.0/(100*(diff_sd)+1))
	else:
		diff_sd = max(abs_diff, sd*0.1)
		#diff_sd = sd*1.0/(abs_diff+0.001)
		print "away", abs_diff, sd, diff_sd, sign
		mu -= sign*diff_sd
		sd *= (1 + 1.0/(100*(diff_sd)+1))
	if mu < 0:
		mu = 0

def gen_random_values():
	value1 = math.floor(random.gauss(mu, sd))
	value1 = 0 if value1 < 0 else value1
	value2 = math.floor(random.gauss(mu, sd))
	value2 = 0 if value2 < 0 else value2
	return value1, value2

plots = []

for i in range(1000):
	value1, value2 = gen_random_values()
	err1 = error(value1)
	err2 = error(value2)
	if err1 < err2:
		winner = value1
		loser = value2
		win_error = err1
		loser_error = err2
	else:
		winner = value2
		loser = value1
		win_error = err2
		loser_error = err1

	update_probs(winner, win_error, True)
	if (loser_error != win_error):
		update_probs(loser, loser_error, False)

	print "win_error", win_error
	plots.append([winner, mu, sd])

figure()
plot(plots)
savefig('distrib.png')

