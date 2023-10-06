import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.init as init

# Define a two-dimensional quadratic function to be minimized
p_f = 0.5  # some parameter changing the shape of the function

def f(x, y):
    return (p_f * x) ** 2 + y ** 2 + p_f * x * y

# Define starting point in the upper right corner of the plot
r = 1.0
xi = torch.tensor([0.9 * r], requires_grad=True)
yi = torch.tensor([0.8 * r], requires_grad=True)
p_x = [xi.item()]  # List of x-values
p_y = [yi.item()]  # List of y-values

# Learning rate
eta = 0.5
# Number of steps
n_iter = 4

# Define the optimizer
optimizer = torch.optim.SGD([xi, yi], lr=eta)

# Perform gradient descent
for i in range(n_iter):
    optimizer.zero_grad()  # Reset gradients
    z = f(xi, yi)  # Compute the function value
    z.backward()    # Compute gradients using autograd
    optimizer.step()  # Update parameters
    p_x.append(xi.item())  # Store x-coordinate
    p_y.append(yi.item())  # Store y-coordinate

# Make contour plot
x_values = np.linspace(-r, r, 400)
y_values = np.linspace(-r, r, 400)
X, Y = np.meshgrid(x_values, y_values)
Z = f(torch.Tensor(X), torch.Tensor(Y))
contours = plt.contour(X, Y, Z, levels=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0], colors='grey')
plt.clabel(contours, inline=True, fontsize=6)
plt.imshow(Z, extent=[-r, r, -r, r], origin='lower', cmap='RdGy', alpha=0.5)

# Add optimum
plt.plot(0, 0, 'x', c='k')

# Plot gradient steps
plt.plot(p_x, p_y, marker='o', markersize=4, label='Gradient Descent Path', color='b')
plt.legend()

plt.show()