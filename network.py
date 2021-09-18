import numpy as np
import random

"""
Assuming one input layer(32), one hidden layer(16) and one output layer(10)
  - Biases is an array of two arrays of sizes 16, 10 containing random values
  - Weights is an array of two arrays;
    - The first containing array of 10 arrays, each containing 16 elements (links/synpase)
    - The second containing array of 16 arrays, each containing 32 elements (links/synapse)
"""
class Network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
	self.weights = [np.random.randn(x,y) for x, y in zip(sizes[1:], sizes[:-1])]

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

	if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

  
    def update_mini_batch(self, mini_batch, eta):
        """
        Calls backpropagation which returns delta matrices to be used to change
        the weights and the biases of the network.
	
	This happens in batches though; we keep accumulating the delta weight for
	all elements in this batch (ex. -0.02 + 0.001 + 0.012). And ONLY after the
	batch is done, we update the network with the accumulated delta.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_w, delta_b = self.backprop(x,y)
            nabla_w = [w+dw for w, dw in zip(nabla_w, delta_w)]
            nable_b = [b+db for b, db in zip(nable_b, delta_b)]
        
	self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] 
	self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)] 








