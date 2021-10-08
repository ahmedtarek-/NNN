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

