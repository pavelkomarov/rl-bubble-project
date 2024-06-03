from scipy import integrate
import numpy as np

y0 = 3
r0 = 2
x0 = 3

def ycirc(x, ymin, ymax):
	"""This function describes the bubble itself. It's passed to an integrator.
	x0 is where the bubble is centered, left-right. y0 is its vertical center, and r0 is its radius.
	The bubble will be integrated over a little box: between a and b in x
	(https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
	and between ymin and ymax in y dimension. It's useful to plot
	max(0, min(y0 + sqrt{r^2 - (x - x0)^2}, ymax) - ymin) in desmos to get a feel for what's happening.

	The hardest thing to visualize about this is the vertical: Essentially a vertical slice of (the upper
	half of) bubble looks like __,-----.__ If the bubble were drawn out in full, centered at 0, with a
	radius of 2, and ymin=1 and ymax=2, then you get the upper quarter of the bubble, shifted down to
	rest against the x axis so that when we integrate we get its volume.

	In the original example, only the upper half of the bubble is considered, because this function
	breaks/gets more complicated for the lower half. Here I've attempted to add in that lower half
	"""
	if (x-x0)**2 < r0**2:
		if ymin >= y0: # the standard case, where ymin and ymax are both above the circle's centerline
			return max(min(y0 + np.sqrt(r0**2 - (x - x0)**2), ymax) - ymin, 0)
		elif ymax <= y0: # the mirror case, where ymin and ymax are both below the circle's centerline
			# Draw a picture to see that now ymax = 2y0 - ymin and ymin = 2y0 - ymax
			return max(min(y0 + np.sqrt(r0**2 - (x - x0)**2), 2*y0 - ymin) - (2*y0 - ymax), 0)
		elif ymin < y0 < ymax: # ymin and ymax straddle the centerline 
			# split up into two! The upper slice has height between y0 and ymax, and the lower slice
			# can be thought of in mirror as a slice with height between y0 and y0 + y0 - ymin
			return max(min(y0 + np.sqrt(r0**2 - (x - x0)**2), ymax) - y0, 0) + \
				max(min(y0 + np.sqrt(r0**2 - (x - x0)**2), 2*y0 - ymin) - y0, 0)
	else:
		return 0

print(integrate.quad(ycirc, 0, 10, args=(4,5))) # integral from 0 to 10 max(0, min(3 + sqrt(2^2 - (x - 3)^2), 5) - 4) = 2.45 ish
print(integrate.quad(ycirc, 0, 10, args=(3,4))) # 3.82 ish
print(integrate.quad(ycirc, 0, 10, args=(2,3))) # 3.82 again
print(integrate.quad(ycirc, 0, 10, args=(1,2))) # 2.45 again
print(integrate.quad(ycirc, 0, 10, args=(2,4))) # 7.65 ish
print(integrate.quad(ycirc, 0, 10, args=(0,1))) # 0
print(integrate.quad(ycirc, 0, 10, args=(0,5))) # 12.57 ish





