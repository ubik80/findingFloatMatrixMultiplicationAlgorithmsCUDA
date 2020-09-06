
import backpropCUDA
import numpy

vec = numpy.linspace(0,1,10)

print("before: ", vec)
backpropCUDA.multiply_with_scalar(vec, 10)
print("after: ", vec)
