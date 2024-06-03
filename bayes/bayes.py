
import numpy
from bayes_opt import BayesianOptimization
from matplotlib import pyplot

class Bubbles:
	def __init__(self, fname):
		with open(fname, 'rb') as f:
			self.frame10s = numpy.load(f)
			self.xyr = numpy.load(f)
		self.closest = None

	def xyr_to_ndx(self, x, y, r) -> int:
		"""Find the example with x,y,r closest to the given. I privilege x,y first, then r"""
		# The xyr array isn't actually ordered as intelligently as I'd want, and I don't feel
		# like fixing it, so just brute search.
		xyr = numpy.array([x,y,r])
		euclidean_dist = [numpy.linalg.norm(xyr - self.xyr[i]) for i in range(len(self.xyr))]
		return numpy.argmin(euclidean_dist)
	
	def reward(self, o, g):
		"""o is index of observed, and g is index of example we want to observe"""
		O = self.frame10s[o]
		G = self.frame10s[g]
		fro_dist = numpy.linalg.norm(O - G)
		if numpy.isclose(fro_dist, 0):
			if self.closest is None:
				self.closest = o
			elif self.closest != o:
				raise ValueError("doesn't agree!")

		return -fro_dist

bubs = Bubbles('../inverse/data.npy')
g = numpy.random.randint(len(bubs.xyr)) # choose a random target example

def black_box_function(x, y, r):
	"""Function with unknown internals we wish to maximize.

	For all intents and purposes think of the internals of this function, i.e.: the process
	which generates its output values, as unknown.
	"""
	o = bubs.xyr_to_ndx(x, y, r)
	return bubs.reward(o, g)

# Bounded region of parameter space
pbounds = {'x': (0.5, 2.5), 'y': (0, 1), 'r': (0.05, 0.25)}

optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds)
optimizer.maximize(init_points=50, n_iter=200)

print("true xyr:", bubs.xyr[g])
print("xyr found by Bayesian Optimizatin:", optimizer.max)
print("Bayes-found xyr rounded to agree with discretization:", bubs.xyr[bubs.closest])

target_values = [x['target'] for x in optimizer.res]
pyplot.plot(range(len(target_values)), target_values)
pyplot.title("Bayesian Optimization reward curve")
pyplot.xlabel("query")
pyplot.ylabel("reward")
pyplot.savefig("bayes_reward")


