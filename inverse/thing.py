import numpy
from matplotlib import pyplot

N = [numpy.random.randint(25000) for i in range(10)]
with open('data.npy', 'rb') as f:
    frame10s = numpy.load(f)
    xyr = numpy.load(f)
    frame0s = numpy.load(f)

for n in N:
    pyplot.imshow(numpy.concatenate((frame0s[n], frame10s[n])), cmap='gray')
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.title("x, y, r =" + str(xyr[n,0]) + ", " + str(xyr[n,1]) + ", " + str(xyr[n,2]))
    pyplot.savefig("test" + str(n) + ".png")

print(frame10s.shape)
print(xyr.shape)
