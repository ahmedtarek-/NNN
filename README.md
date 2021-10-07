# NNN
A basic neural network written in Python with the help of numpy


To use

```python

import network
import data_loader

training_data, validation_data, test_data = data_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 0.1, test_data=test_data)


```