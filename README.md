# A Playground for Parametric Gaussian Regression

## Overview
This code provides an interactive visualization of parametric Gaussian models and regression. It lets you build and edit a simple parametric function interactively at runtime. You can adjust the weights of your model to define the prior distribution and let the model learn using Bayesian inference.\
\
Required Python packages are:
- Matplotlib
- NumPy
- SciPy

Run main.py to open the plot.

## What you can do

### Design parametric functions with random weights

Using the buttons on the left you can define a simple parametric function
```math
\begin{align}
f(x) &= w_1 \cdot \phi_1(x) + w_2 \cdot \phi_2(x) + ...\\
&= \phi w
\end{align}
```
from predefined $\phi(x; n)$:

- Polynomial: $\phi(x; n) = x^n $
- Sine: $\phi(x; n) = \sin(n\pi \cdot x)$
- Cosine: $\phi(x; n) = \cos(n\pi \cdot x)$

The parameters $n$ are set using the sliders next to the feature label. The weights $w$ are Gaussian random variables (RVs). Their probability density is shown in the left plot of the figure. By choosing RVs as the coefficients in a parametric function, the function output itself becomes an RV! The distribution over function values $f(x)$ is shown on the right plot.

### Edit the weight distribution
As it is difficult to visualize high dimensional distributions, only a two dimensional subset of the actual weight space is displayed in the left plot. If the model consists of more than two weights, the displayed weights can be changed by clicking on the x or y label of the plot.\
Both displayed weights can be edited together by dragging, the function distribution is updated in real time. By using the mouse wheel, the variance of the distribution can be scaled. Pressing the following keys will change the scrolling behaviour:

- x: scale only along x-direction.
- y: scale only along y-direction.
- r: rotate the density around the mean.

Doubleclicking anywhere on the figure resets the two currently displayed weights.


### Let the model learn

To visualize learning, a set of random samples is generated and added to the right plot. You can generate new samples by clicking the 'Generate samples' button at the top.\
Clicking on the 'Start/ stop training' button starts the training. This will condition the weight distribution on individual data points and update the weights as the training goes on. The way the data is selected can be changed by pressing the 'Sampling order' button. Possible orders are:
- Sequential: Select samples from left to right.
- random: Select samples randomly.
- Least likely: Select the sample that is the least likely based on the current model. Not implemented yet.

## What is happening

### How to obtain the function space distribution

Linear transformations of multivariate Gaussian random variables are Gaussian random variables. Their mean and variance are linear transformations of the original mean vector and covariance matrix:

```math
p(W) \sim \mathcal{N}(W; \mu, \Sigma)
\\
\Rightarrow\\
p(\phi W) \sim \mathcal{N}(\phi W; \phi \mu, \phi \Sigma \phi^T)
```

The projection from weight space into function space is linear, so the properties of the function space distribution for a given x-value are determined by linear operations on $\mu$ and $\Sigma$. Using broadcasting, they can be calculated efficiently for several x-values at once enabling real time updates of the function space distribution with decent resolution.

### How to learn from data

Bayes theorem describes how a model changes after observing data.
The posterior conditional distribution of weights $W$ after observing the data $y$ is given as
```math
p(W|y) = \frac{p(y|W)p(W)}{\int p(y|W)p(W)dW}
```
where $p(y|W)$ is the probability to observe $y$ assuming that the model $W$ is correct and $p(W)$ is the prior distribution over weights. For observations of linear projections of $W$ the posterior can be expressed as

```math
p(W|\phi W = y) = \mathcal{N}(W;\Sigma \phi (\phi^T \Sigma \phi + \sigma^2)^{-1}(y - \phi^T \mu), \Sigma - \Sigma \phi(\phi^T \Sigma \phi + \sigma^2)^{-1}\phi^T \Sigma)
```
where $\sigma$ is the expected amount of noise in the target data.
