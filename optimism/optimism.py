
import numpy
from matplotlib import pyplot
#numpy.set_printoptions(threshold=numpy.inf)
#from scipy.ndimage import center_of_mass

n_classes = 40

class BubblesBandit:
	def __init__(self, fname):
		with open(fname, 'rb') as f:
			self.frame10s = numpy.load(f)
			self.xyr = numpy.load(f)

		self.c = numpy.array([self.xyr_to_class(xyr) for xyr in self.xyr])
		
	def xyr_to_class(self, xyr) -> int:
		# We're gridding like this:
		# y=1
		# | 4 | 5 | 6 | 7 |
		# | 0 | 1 | 2 | 3 |
		# x=0.5, y=0		x=2.5
		# So the box we end up in is given by (x - 0.5)//0.5
		# and by y//0.5
		# Add in min()s to handle case where y=1.0 and x=2.5 exactly, which can push us in
		# to other, higher-numbered boxes off the border that we don't want to exist.
		if n_classes == 40:
			return 8 * int(numpy.round((xyr[2] - 0.05)/0.05)) + \
				4 * min(int(xyr[1]//0.5), 1) + min(int((xyr[0] - 0.5)//0.5), 3)
		elif n_classes == 8:
			return 4 * min(int(xyr[1]//0.5), 1) + min(int((xyr[0] - 0.5)//0.5), 3)
	
	def pull(self, c):
		"""pull the cth class' lever. Something happens! We "run the simulation" by just getting the
		corresponding stored result."""
		if not isinstance(c, (int, numpy.int64)) or not 0 <= c < n_classes:
			raise ValueError("c needs to be an int in [0," + str(n_classes) + ")")
		
		# Select a random example from the corresponding class
		return numpy.random.choice(numpy.where(self.c == c)[0])

	def reward(self, o, g):
		"""o is index of observed, and g is index of example we want to observe"""
		# Calculating the diff of a Frobenius norm. I'm not sure this will be smooth enough!
		#return -numpy.linalg.norm(G - O) # This doesn't pass the test_reward fn test!
		#O_com = numpy.array(center_of_mass(O))
		#G_com = numpy.array(center_of_mass(G))
		#return -numpy.linalg.norm(O_com - G_com)
		#return -numpy.linalg.norm(O_com - G_com) - numpy.linalg.norm(G - O)
		# g = numpy.concatenate((numpy.mean(-G+1, axis=0), numpy.mean(-G+1, axis=1),
		# 					numpy.std(-G+1, axis=0), numpy.std(-G+1, axis=1)))
		# o = numpy.concatenate((numpy.mean(-O+1, axis=0), numpy.mean(-O+1, axis=1),
		# 				 numpy.std(-O+1, axis=0), numpy.std(-O+1, axis=1)))
		# return -numpy.linalg.norm(g - o)
		#print(numpy.linalg.norm(O - G))
		#g = numpy.concatenate((numpy.sum(-G+1, axis=0), numpy.sum(-G+1, axis=1)))
		#o = numpy.concatenate((numpy.sum(-O+1, axis=0), numpy.sum(-O+1, axis=1)))
		#d = numpy.linalg.norm(g - o)
		O = self.frame10s[o]
		G = self.frame10s[g]
		bonus = self.c[o] == self.c[g]

		return bonus - 0.2*numpy.linalg.norm(O - G)

# Sutton and Barto p41 use A_t = argmax_a{ Q_t(a) + c sqrt(ln(t)/N_t(a)) }, where t is the number
# of pulls so far, and c controls amount of exploration.

bubs = BubblesBandit('../inverse/data.npy')

# I want to see what the average reward is across each class when I choose a random
# example and compare to all classes.
def test_reward_fn():
	n_agree = 0
	for c in range(n_classes):
		print("Random G chosen from class", c)
		g = numpy.random.choice(numpy.where(bubs.c == c)[0])

		cum_rewards = [0 for i in range(n_classes)]

		for o in range(len(bubs.frame10s)):
			k = bubs.c[o]
			r = bubs.reward(o, g)
			cum_rewards[k] += r
		
		#for k in range(len(cum_rewards)):
		#	print("avg reward comparing to class", k, ":", cum_rewards[k]/len(bubs.frame10s))
		arg_max = numpy.argmax(cum_rewards)
		print("maximal reward class:", arg_max)
		print("value:", cum_rewards[arg_max]/len(bubs.frame10s))
		second_largest = numpy.partition(cum_rewards, -2)[-2]
		print("diff with second-largest class", (cum_rewards[arg_max] - second_largest)/len(bubs.frame10s))
		n_agree += arg_max == c

	print("n_agree", n_agree)

#test_reward_fn()

# Now for the main event, the core exploration algorithm.
# We start off by trying every action once to establish a mu, and then in further time steps
# we select according to Sutton and Bartow's formula.

target_class = numpy.random.randint(n_classes)
print("randomly choosing target example from class", target_class)
g = bubs.pull(target_class) # goal is indexed by g

mus = numpy.zeros(n_classes)
for t in range(n_classes):
	o = bubs.pull(t) # I pull the tth lever and get back observation (indexed by) o
	mus[t] = bubs.reward(o, g)
Ns = numpy.array([1 for t in range(n_classes)])

c = 10
T = 5000
all_mus = numpy.zeros((len(mus), T-n_classes))
all_upper_bounds = numpy.zeros((len(mus), T-n_classes))
all_lower_bounds = numpy.zeros((len(mus), T-n_classes))

exp_mov_avg_rewards = [None]
for t in range(n_classes, T):
	bounds = c*numpy.sqrt(numpy.log(t)/Ns)

	j = t-n_classes
	if j % 1000 == 0:
		print(j)
	all_mus[:,j] = mus
	all_upper_bounds[:,j] = mus + bounds
	all_lower_bounds[:,j] = mus - bounds

	a = numpy.argmax(mus + bounds)
	o = bubs.pull(a)
	r = bubs.reward(o, g)
	if exp_mov_avg_rewards[-1] is None:
		exp_mov_avg_rewards[0] = r
	else:
		exp_mov_avg_rewards.append(0.99*exp_mov_avg_rewards[-1] + 0.01*r)

	mus[a] =(Ns[a] * mus[a] + r)/(Ns[a] + 1) # calculate running mean in the ordinary way
	Ns[a] += 1 # we pulled the ath lever, so increment counter

all_mus[:,-1] = mus
bounds = c*numpy.sqrt(numpy.log(T)/Ns)
all_upper_bounds[:,-1] = mus + bounds
all_lower_bounds[:,-1] = mus - bounds

pyplot.figure(figsize=(10,5))
for k in range(n_classes):
	pyplot.plot(range(all_mus.shape[1]), all_mus[k], label='target class' if k==target_class else None)
	pyplot.fill_between(range(all_mus.shape[1]), all_upper_bounds[k], all_lower_bounds[k], alpha=0.3)
pyplot.legend()
pyplot.title("Multiarm Bandit with Optimism (Upper Confidence Bound)")
pyplot.xlabel('time')
pyplot.ylabel('value')
pyplot.savefig('confidence_bounds.png')

pyplot.figure()
pyplot.plot(range(len(exp_mov_avg_rewards)), exp_mov_avg_rewards)
pyplot.title("Optimistic Agent Reward (exponential moving avg)")
pyplot.xlabel('time')
pyplot.ylabel('reward')
pyplot.savefig('reward_optimism.png')


#mus = [None for i in range(n_classes)]
#Ns = [0 for i in range(n_classes)]



