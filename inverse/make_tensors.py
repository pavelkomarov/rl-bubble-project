import os, numpy
from matplotlib import pyplot

frame10s = []
frame0s = []
xyr = []
i = 0
for f in sorted(os.listdir('.')):
	if f.startswith("plots"):
		if i % 1000 == 0:
			print(i)
		i += 1

		parts = f.split("_")
		xyr.append([[float(n) for n in parts[1:]]]) # x0, y0, r0 in rows

		h = pyplot.imread(f + '/frame0010fig0.png')
		g = h[31:198,61:558] # The 4th channel is all 1s. The first 3 are all identical (greyscale)
		
		#assert numpy.allclose(g[:,:,0], g[:,:,1])
		#assert numpy.allclose(g[:,:,0], g[:,:,2])

		frame10s.append(g[None, :,:,0]) # greyscale image in each layer

		h = pyplot.imread(f + '/frame0000fig0.png')
		g = h[31:198,61:558] # The 4th channel is all 1s. The first 3 are all identical (greyscale)
		frame0s.append(g[None, :,:,0])

frame10s = numpy.concatenate(frame10s)
frame0s = numpy.concatenate(frame0s)
xyr = numpy.concatenate(xyr)

with open('data.npy', 'wb') as f:
	numpy.save(f, frame10s)
	numpy.save(f, xyr)
	numpy.save(f, frame0s)


