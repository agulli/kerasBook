import theano
from theano import tensor

a = tensor.dscalar()
b = tensor.dscalar()

c = a + b

f = theano.function([a,b], c)

result = f(2.5, 2.5)
print(result)