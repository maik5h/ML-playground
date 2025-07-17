import numpy as np
from gaussians import Gaussian, get_gaussian
import matplotlib.pyplot as plt

number_samples = 100
x_lim = (-5, 5)
y_lim = (-15, 15)

# Create a linear function with random offset and slope.
offset = np.random.rand() * 5 - 2.5
slope = np.random.rand() * 6 - 3

# Create noise vector to be added to samples.
noise_amount = 0.5
sample_noise = np.random.normal(loc=0, scale=noise_amount, size=number_samples)

x = np.linspace(x_lim[0], x_lim[1], number_samples)
y = (offset + x * slope) + sample_noise

# Initial model.
model = Gaussian(np.array((0, 0)), np.eye(2))

phi = np.stack((np.ones(number_samples), x)).T
function_dist = model.project(phi)
sigma = function_dist.sigma.diagonal()

densities = []
for i in range(number_samples):
    y_atx = np.linspace(y_lim[0], y_lim[1], 600)
    column = get_gaussian(y_atx, function_dist.mu[i], sigma[i])
    densities.append(column)

densities = np.stack(densities).T

fig, ax = plt.subplots()

ax.imshow(densities, cmap='Blues', aspect='auto', extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]))

ax.plot(x, y, '.', color='black')
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
