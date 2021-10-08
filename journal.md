## Journal
This file tracks concepts learned in the Deep Learning course https://www.youtube.com/watch?v=c36lUUr864M


### Autograd

```python
import torch

# 1. Basic gradient using backward
weights = torch.ones(3, requires_grad=True)
x = weights.mean()
x.backward()

print(weights.grad)

# 2. To get rid of grad
weights.requires_grad_(False) # OR
weights.detach() # OR
with torch.no_grad():
	y = weights + 2
	print(y)


# 3. Basic feed based on training sets
weights = torch.ones(3, requires_grad=True)

for epoch in range(1):
	r = (weights+3).sum()
	r.backward()

	print(weights.grad)

	weights.grad.zero_()
```

### Back Propagation & SGD

- Basic linear regression example:
	1. Forward pass: to calculate loss
	2. Calculate intermediate gradient as you go
	3. Backward pass to get gradient of loss with respect to weight

```python
import numpy as np

# Our desired function is y = 2*x
# What we want to do is optimise weight (w)
# so that it learns to give the correct value
# that would satisfy the function
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

learning_rate = 0.01
iterations = 10

w = 0.0

def forward(x):
	return w * x;

def loss(y,y_predicted):
	return ((y_predicted-y)**2).mean()

# dJ/dw = 1/N 2x (w*x - y)
def gradient(y,y_predicted,x):
	return np.dot(2*x, (y_predicted - y)).mean()


print(f'Prediction before training the weight: f(7): {forward(7)}')

for epoch in range(iterations):
	# 1. forward
	y_predicted = forward(X)
	l = loss(Y,y_predicted)
	# 2. gradient
	dw = gradient(Y,y_predicted,X)
	# 3. update weight
	w -= (dw * learning_rate)
	if epoch % 1 == 0:
		print(f'Epoch {epoch} => loss: {l}, w: {w}')

print(f'Prediction after training the weight: f(7): {forward(7)}')

```

- Linear regression example (using Pytorch for gradient):
	1. Forward pass: to calculate loss
	2. Calculate intermediate gradient as you go
	3. Backward pass to get gradient of loss with respect to weight

```python
import numpy as np
import torch

# Our desired function is y = 2*x
# What we want to do is optimise weight (w)
# so that it learns to give the correct value
# that would satisfy the function
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

learning_rate = 0.01

w = torch.tensor(0.0, requires_grad=True)

def forward(x):
	return w * x;

def loss(y,y_predicted):
	return ((y_predicted-y)**2).mean()

# dJ/dw = 1/N 2x (w*x - y)
# def gradient(y,y_predicted,x):
# 	return np.dot(2*x, (y_predicted - y)).mean()


print(f'Prediction before training the weight: f(7): {forward(7)}')

iterations = 40
for epoch in range(iterations):
	# 1. forward
	y_predicted = forward(X)
	l = loss(Y,y_predicted)
	# 2. gradient
	l.backward()
	# 3. update weight
	with torch.no_grad():
		w -= (w.grad * learning_rate)
	# 4. zero the weight
	w.grad.zero_()
	if epoch % 4 == 0:
		print(f'Epoch {epoch} => loss: {l}, w: {w}')

print(f'Prediction after training the weight: f(7): {forward(7)}')

```



### Linear Regression using pytorch and plotting

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)
print(f'X is {X}')
print(f'Y is {Y}')

n_samples, n_features = X.shape

# 1) Model
model = nn.Linear(n_features, 1)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) learning iteration
n_epochs = 1000

print(f'Prediction before running for f(5): {model(torch.tensor([5], dtype=torch.float32))[0]}')
for epoch in range(n_epochs):
	# forward pass and loss
	y_predicted = model(X)
	loss = criterion(y_predicted, Y)
	# back propagation (to get gradient)
	loss.backward()
	# update weight/feature
	optimizer.step()
	# zero the gradient
	optimizer.zero_grad()
	if (epoch+1) % 10 == 0:
		print(f'Epoch {epoch+1}; loss: {loss.item():.4f}')

print(f'Prediction before running for f(5): {model(torch.tensor([5], dtype=torch.float32))[0]}')

predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show
```




_Need to move this folder inside NNN and push to the repo after I finish_
