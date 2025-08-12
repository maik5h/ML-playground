# An Interactive Playground for Basic Machine Learning.

## Overview
This project aims to give insights into basic concepts of machine learning using interactive plots. As for now, there is only one plot available, which covers parametric regression using Gaussian models. More plots are about to come.\
\
This code uses Python only (>=3.9.13 recommended) and can be run without being installed, as long as the required packages are available:
- Matplotlib
- NumPy
- SciPy

To setup the project, navigate to the desired location, activate a python environment and run
```bash
git clone git@github.com:maik5h/ML-playground.git
cd ML-playground
pip install -r requirements.txt
```

Run `python run_parametric_regression.py` to open the plot.

# Parametric Gaussian Regression

The idea of parametric Gaussian regression is to build a parametric function 

```math
\begin{align}
f(x) &= w_1 \cdot \phi_1(x) + w_2 \cdot \phi_2(x) + ...\\
&= \phi w
\end{align}
```
from predefined features $\phi(x)$. These features are fixed and not part of the trainable parameters. The weights $w$ however are modeled as Gaussian random variables (RVs) and can be conditioned on training data using  Bayesian inference (no backpropagation or SGD required!). Choosing the weights as RVs results in an $f(x)$ that is also an RV over function values at every given $x$. This way, quite complex probability densities can be constructed.

## What you can do

### Design parametric functions with random weights

While parametric regression can handle multidimensional data, it was decided to stick with 1D data for the sake of simplicity.\
\
On the left hand side of the window is an interface used to set up the parametric function. The following features are available:

- Polynomial: $\phi(x; p) = x^{p} $
- Harmonic: $\phi(x; p_1, p_2) = \sin(p_1\pi \cdot x + p_2\dfrac{\pi}{2})$
- Gaussian: $\phi(x; \sigma, \mu) = \frac{1}{\sqrt{2\pi}\sigma} \cdot \exp(\frac{(x - \mu)^2}{2\sigma^2})$

Features can be added using the three buttons on top and removed by the red buttons next to the feature label. The parameters of features are set using the corresponding sliders.

### Edit the weight distribution
The weights are initialized independently with $\mu=0$ and $\sigma=1$. Their probability density is displayed in the left plot. As it is difficult to visualize high dimensional distributions, only a two dimensional subset of the actual weight space is plotted. If the model consists of more than two weights, the displayed weights can be changed by clicking on the x or y label of the plot. The distribution over $f(x)$ is shown in the right plot.\
Both displayed weights can be edited by dragging, the function distribution is updated in real time. By using the mouse wheel, the variance of the distribution can be scaled. Pressing the following keys will change the scrolling behaviour:

- x: scale only along x-direction.
- y: scale only along y-direction.
- r: rotate the density around the mean.

Doubleclicking anywhere on the figure resets the two currently displayed weights.

### Let the model learn

To visualize learning, a set of random samples is generated and added to the right plot. You can generate new samples by clicking the 'Generate samples' button at the top.\
Clicking on the 'Start/ stop training' button starts the training. This will condition the weight distribution on individual data points and update the weights as the training goes on. The way the data is selected can be changed by pressing the 'Sampling order' button. Possible orders are:
- Sequential: Select samples from left to right.
- Random: Select samples randomly.
- Least likely: Select the sample that is the least likely based on the current model.

You can try to find a parametric function that fits the data and observe the distribution approach the distribution of target samples.

## What is happening

### How to obtain the function space distribution

Linear transformations of multivariate Gaussian random variables are Gaussian random variables. Their mean and variance are linear transformations of the original mean vector and covariance matrix:

```math
p(W) \sim \mathcal{N}(W; \mu, \Sigma)
\\
\Rightarrow\\
p(\phi W) \sim \mathcal{N}(\phi W; \phi \mu, \phi \Sigma \phi^T)
```

The projection from weight space into function space is linear by definition, so the function space distribution at every x-value is Gaussian and well defined. It can be calculated efficiently for several x-values at once enabling real time updates of the function space distribution with decent resolution.

### How to learn from data

Bayes theorem describes how a model changes after observing data.
The posterior conditional distribution of weights $W$ after observing the data $y$ is given as
```math
p(W|y) = \frac{p(y|W)p(W)}{\int p(y|W)p(W)dW}
```
where $p(y|W)$ is the probability to observe $y$ under the current model $W$ and $p(W)$ is the prior distribution over weights. For observations of linear projections of $W$ the posterior can be expressed as

```math
p(W|\phi W = y) = \mathcal{N}(W;\Sigma \phi (\phi^T \Sigma \phi + \sigma^2)^{-1}(y - \phi^T \mu), \Sigma - \Sigma \phi(\phi^T \Sigma \phi + \sigma^2)^{-1}\phi^T \Sigma)
```
where $\sigma$ is the expected amount of noise in the target data. The posterior is still a Gaussian RV with $\mu$ and $\sigma$ that can be obtained through linear operations on the previous $\mu$ and $\sigma$.\
\
Most of the concepts and some inspiration for the code are from [this lecture](https://www.youtube.com/playlist?list=PL05umP7R6ij0hPfU7Yuz8J9WXjlb3MFjm) ([CC BY-SA 4.0](https://creativecommons.org/licenses/by/4.0/)) by Prof. Philipp Hennig which i strongly recommend to anyone interested in probabilistic machine learning.