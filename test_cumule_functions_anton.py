import unittest
import numpy as np
from pybrain.tools.shortcuts import buildNetwork

from anton_silent_predictor import (getInputMaskParamRange, truncateParams,
									replaceNetParams, extractCommonInputWeights,
									copyInputWeights)

class TestPredictorFunctions(unittest.TestCase):
	def test_extractCommonInputWeights(self):
		net = buildNetwork(3, 5, 1, bias=True)
		selfInputMask = [1, 0, 0, 1, 1]
		otherInputMask = [1, 0, 0, 0, 1]
		net._setParameters(np.array(range(26)))
		all_weights = net.connections[net['in']][0].params.reshape(3, 5)
		common_weights = extractCommonInputWeights(all_weights, selfInputMask, otherInputMask)
		common_weights_exp = np.array([[6, 7, 8, 9, 10], None, None, None, [16, 17, 18, 19, 20]])
		self.assertEqual(len(common_weights), len(common_weights_exp))
		for i in xrange(len(common_weights)):
			if common_weights_exp[i] is None:
				self.assertTrue(common_weights[i] is None)
			else:
				self.assertEqual(list(common_weights[i]), list(common_weights_exp[i]))

		otherInputMask = [0, 1, 1, 0, 0]
		common_weights = extractCommonInputWeights(all_weights, selfInputMask, otherInputMask)
		common_weights_exp = np.zeros(0)
		self.assertEqual(list(common_weights), list(common_weights_exp))

	def test_getInputMaskParamRange(self):
		net = buildNetwork(2, 5, 6, 1, bias=False)
		param_range = getInputMaskParamRange(net)
		self.assertEqual(param_range, (0, 10))

		net = buildNetwork(2, 5, 6, 1, bias=True)
		param_range = getInputMaskParamRange(net)
		self.assertEqual(param_range, (12, 22))

	def test_truncateParams(self):
		array_4by5 = np.array(range(20)).reshape(4, 5)
		array_trunc = truncateParams(array_4by5, (4, 3))
		array_exp = np.array([0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]).reshape(4, 3)
		self.assertEqual(len(array_trunc), len(array_exp))
		for i in xrange(len(array_trunc)):
			self.assertEqual(list(array_trunc[i]), list(array_exp[i]))

		array_trunc = truncateParams(array_4by5, (2, 3))
		array_exp = np.array([0, 1, 2, 5, 6, 7]).reshape(2, 3)
		self.assertEqual(len(array_trunc), len(array_exp))
		for i in xrange(len(array_trunc)):
			self.assertEqual(list(array_trunc[i]), list(array_exp[i]))

	def test_replaceNetParams(self):
		net = buildNetwork(2, 5, 6, 1, bias=True)
		params_saved = list(np.copy(net.params))
		param_range = getInputMaskParamRange(net)
		# params_new are the same size as the existing ones
		params_new = np.arange(10).reshape(2, 5)
		replaceNetParams(net, param_range, (2, 5), params_new)
		self.assertEqual(list(net.params[:11]), params_saved[:11])
		self.assertEqual(list(net.params[12:22]), range(10))
		self.assertEqual(list(net.params[22:]), list(params_saved[22:]))
		self.assertEqual(list(net.connections[net['in']][0].params), range(10))

		net = buildNetwork(3, 5, 6, 1, bias=True)
		params_saved = list(np.copy(net.params))
		param_range = getInputMaskParamRange(net)
		# param_range = [11, 26]
		# params_new are smaller than the existing ones
		params_new = np.arange(8).reshape(2, 4)
		replaceNetParams(net, param_range, (3, 5), params_new)
		self.assertEqual(list(net.params[:12]), params_saved[:12])
		self.assertEqual(list(net.params[12:]), range(4) + [params_saved[16]] + range(4, 8) + params_saved[21:])
		self.assertEqual(list(net.connections[net['in']][0].params), (range(4) + [params_saved[16]] +
																	  range(4, 8) + params_saved[21:27]))

	def test_copyInputWeightsAll(self):
		net1 = buildNetwork(3, 5, 5, 1, bias=True)
		net1._setParameters(range(56))
		net2 = buildNetwork(2, 4, 5, 1, bias=True)
		net2._setParameters(range(100, 143))
		inputMask1 = [0, 1, 1, 0, 1]
		inputMask2 = [1, 0, 1, 0, 0]
		# copy weights of net1 to net2
		# net1 is bigger than net2
		copyInputWeights(net2, net1, inputMask2, inputMask1, use_common_weights=False)
		self.assertEqual(list(net1.params), range(56))
		net2_params_exp = range(100, 110) + range(11, 15) + range(16, 20) + range(118, 143)
		self.assertEqual(list(net2.params), net2_params_exp)

		net2 = buildNetwork(4, 6, 5, 1, bias=True)
		net2._setParameters(range(100, 171))
		inputMask1 = [0, 1, 1, 0, 1]
		inputMask2 = [1, 0, 1, 1, 1]
		# copy weights of net1 to net2
		# net2 is bigger than net1
		copyInputWeights(net2, net1, inputMask2, inputMask1, use_common_weights=False)
		self.assertEqual(list(net1.params), range(56))
		net2_params_exp = (range(100, 112) + range(11, 16) + [117] + range(16, 21) + [123] + 
						   range(21, 26) + range(129, 171))
		self.assertEqual(list(net2.params), net2_params_exp)

	def test_copyInputWeightsCommon(self):
		net1 = buildNetwork(3, 5, 5, 1, bias=True)
		net1._setParameters(range(56))
		net2 = buildNetwork(2, 4, 5, 1, bias=True)
		net2._setParameters(range(100, 143))
		inputMask1 = [0, 1, 1, 0, 1]
		inputMask2 = [1, 0, 0, 1, 0]
		# copy weights of net1 to net2
		# no common weights
		copyInputWeights(net2, net1, inputMask2, inputMask1, use_common_weights=True)
		self.assertEqual(list(net1.params), range(56))
		self.assertEqual(list(net2.params), range(100, 143))

		net1 = buildNetwork(3, 5, 5, 1, bias=True)
		net1._setParameters(range(56))
		net2 = buildNetwork(3, 4, 5, 1, bias=True)
		net2._setParameters(range(100, 147))
		inputMask1 = [0, 1, 1, 0, 1]
		inputMask2 = [1, 0, 1, 0, 1]
		# copy weights of net1 to net2
		# 2 common weight
		copyInputWeights(net2, net1, inputMask2, inputMask1, use_common_weights=True)
		self.assertEqual(list(net1.params), range(56))
		net2_params_exp = range(100, 114) + range(16, 20) + range(21, 25) + range(122, 147)
		self.assertEqual(list(net2.params), net2_params_exp)

if __name__ == '__main__':
	unittest.main()
