# I want to get a sense of how solvable this problem is. If I give a conv net the image and train
# it to return me the input xyr numbers, can it succeed?

import numpy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from matplotlib import pyplot


class ConvNet(nn.Module):

	def __init__(self, input_shape):
		super(ConvNet, self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution kernel
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.conv3 = nn.Conv2d(16, 32, 5)

		# an affine operation: y = Wx + b
		x = (((input_shape[0] - 4)//2 - 4)//2 - 4)//2
		y = (((input_shape[1] - 4)//2 - 4)//2 - 4)//2
		self.fc1 = nn.Linear(32 * x * y, 2000)
		self.fc2 = nn.Linear(2000, 320)
		self.fc3 = nn.Linear(320, 3)

	def forward(self, x):
		# Convolution layer C1: 1 input image channel, input_shape, 6 output channels,
		# 5x5 square convolution, it uses RELU activation function, and
		# outputs a Tensor with size (N, 6, input_shape[0] - 5 + 1, input_shape[1] - 5 + 1),
		# where N is the size of the batch
		c1 = F.relu(self.conv1(x))
		# Subsampling layer S2: 2x2 grid, purely functional,
		# this layer does not have any parameter, and outputs a
		# (N, 6, (input_shape[0] - 5 + 1) // 2, (input_shape[1] - 5 + 1) // 2) Tensor
		s2 = F.max_pool2d(c1, 2)
		# Convolution layer C3: 6 input channels, 16 output channels,
		# 5x5 square convolution, it uses RELU activation function, and
		# outputs a (N, 16, (input_shape[0] - 5 + 1) // 2 - 5 + 1, (input_shape[1] - 5 + 1) // 2) - 5 + 1) Tensor
		c3 = F.relu(self.conv2(s2))
		# Subsampling layer S4: 2x2 grid, purely functional,
		# this layer does not have any parameter, and outputs a
		# (N, 16, ((input_shape[0] - 5 + 1) // 2 - 5 + 1) // 2, ((input_shape[1] - 5 + 1) // 2) - 5 + 1) // 2) Tensor
		s4 = F.max_pool2d(c3, 2)
		# convolve one last time. Yields shape ((input_shape[0] - 4) // 2 - 4) // 2 - 4, ((input_shape[1] - 4) // 2) - 4) - 4
		c5 = F.relu(self.conv3(s4))
		# Then pool one last time. Yields shape (((input_shape[0] - 4) // 2 - 4) // 2 - 4) // 2, (((input_shape[1] - 4) // 2) - 4) - 4) // 2
		s6 = F.max_pool2d(c5, 2)
		# Flatten operation: purely functional, outputs a (N, 32 * shape) Tensor
		s6 = torch.flatten(s6, 1)
		# Fully connected layer F5: (N, 400) Tensor input,
		# and outputs a (N, 120) Tensor, it uses RELU activation function
		f7 = F.relu(self.fc1(s6))
		# Fully connected layer F6: (N, 120) Tensor input,
		# and outputs a (N, 80) Tensor, it uses RELU activation function
		f8 = F.relu(self.fc2(f7))
		# Gaussian layer OUTPUT: (N, 80) Tensor input, and
		# outputs a (N, 3) Tensor
		yhat = self.fc3(f8)

		return yhat

class Bubbles(Dataset):
	"""It's easiest to use torch's builting dataloader to manage randomization and stuff, so
	give the people what they want! (It expects a DataSet object.)"""
	def __init__(self, fname):
		with open(fname, 'rb') as f:
			self.frame10s = numpy.load(f)
			self.xyr = numpy.load(f)

	def __getitem__(self, key):
		"""this returns data and label for a particular example"""
		return self.frame10s[key][None, :, :].astype(numpy.float32), self.xyr[key].astype(numpy.float32) #self.xyr_to_class(self.xyr[key])

	def __len__(self):
		return self.xyr.shape[0]
	
	# def xyr_to_class(self, xyr) -> int:
	# 	# We're gridding like this:
	# 	# y=1
	# 	# | 4 | 5 | 6 | 7 |
	# 	# | 0 | 1 | 2 | 3 |
	# 	# x=0.5, y=0		x=2.5
	# 	# So the box we end up in is given by (x - 0.5)//0.5
	# 	# and by y//0.5
	# 	# Add in min()s to handle case where y=1.0 and x=2.5 exactly, which can push us in
	# 	# to other, higher-numbered boxes off the border that we don't want to exist.
	# 	return 8 * int(np.round((xyr[2] - 0.05)/0.05)) + \
	# 		4 * min(int(xyr[1]//0.5), 1) + min(int((xyr[0] - 0.5)//0.5), 3)


batch_size = 128
n_epoch = 12
device = "cuda:1"
lrate = 1e-4

dataset = Bubbles("../inverse/data.npy")
train, test = random_split(dataset, (int(len(dataset)*0.7), int(len(dataset)*0.3)))

train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=5)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=5)

net = ConvNet((167, 497))
net.to(device)

optim = torch.optim.Adam(net.parameters(), lr=lrate)
criterion = nn.MSELoss()

train_losses = []
test_losses = []
for ep in range(n_epoch):
	print(f'epoch {ep}')
	net.train() # put it in training mode

	optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch) # linear learning rate decay

	pbar = tqdm(train_dataloader)
	loss_ema = None
	for x, y in pbar:
		optim.zero_grad()
		x = x.to(device)
		y = y.to(device)

		yhat = net(x)
		loss = criterion(y, yhat)
		loss.backward()

		if loss_ema is None:
			loss_ema = loss.item()
		else:
			loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
		pbar.set_description(f"train loss: {loss_ema:.4f}")
		optim.step()

	net.eval()
	with torch.no_grad():
		pbar = tqdm(train_dataloader)
		cum_loss = 0
		n = 0
		for x, y in pbar:
			x = x.to(device)
			y = y.to(device)

			yhat = net(x)

			loss = criterion(y, yhat)
			cum_loss += float(loss)*y.shape[0]
			n += y.shape[0]

			pbar.set_description(f"evaluating train loss: {loss:.4f}")

		train_losses.append(cum_loss / n)

		pbar = tqdm(test_dataloader)
		cum_loss = 0
		n = 0
		for x, y in pbar:
			x = x.to(device)
			y = y.to(device)

			yhat = net(x)

			loss = criterion(y, yhat)
			cum_loss += float(loss)*y.shape[0]
			n += y.shape[0]

			pbar.set_description(f"evaluating test loss: {loss:.4f}")

		test_losses.append(cum_loss / n)


pyplot.plot(range(len(train_losses)), train_losses, label='train')
pyplot.plot(range(len(test_losses)), test_losses, label='test')
pyplot.legend()
pyplot.title('Conv Net trying to learn Image -> xyr')
pyplot.xlabel('epoch')
pyplot.ylabel('loss')
pyplot.savefig('conv_loss.png')