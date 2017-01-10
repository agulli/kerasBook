from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 9
numpy.random.seed(seed)

def _col_filter (s):
  if (s == 'b'):
  	return float(1)
  else:
  	return float(0)

# Dataset load
dataset = numpy.loadtxt('/Users/gulli/Keras/codeBook/code/data/ionosphere.data.txt', 
	converters = {34: _col_filter},
	delimiter=",")
# the initial 34 colums are attributes
# column 34 is the truth value to be predicted
X = dataset[:,0:34]
Y = dataset[:,34]

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=34, init='uniform', activation='relu'))
model.add(Dense(34, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#Compile the model and define loss function, optimizer and metric for evaluation
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=200, batch_size=10)

# Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))