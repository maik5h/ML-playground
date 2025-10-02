# An Interactive Playground for Basic Machine Learning.

## Overview
I often find myself struggling to imagine certain concepts and the best way to get an intuitive understanding for me is to have an interactive visualization. Whenever I find something machine learning related, which is difficult to visualize in my head but doable in code, I try to implement it here. Hopefully you will find this helpful as well or at least fun to play around with!\
Currently there are two interactive plots available, which cover regression using linear Gaussian models and Gaussian processes. More plots are about to come.\
\
This project is exclusively written in Python (>=3.9.13 recommended) and can be run without being installed, as long as the required packages are available:
- Matplotlib
- NumPy
- SciPy
- pyYAML

To setup the project, navigate to the desired location, activate a python environment and run
```bash
git clone git@github.com:maik5h/ML-playground.git
cd ML-playground
pip install -r requirements.txt
```

In `ML-playground/src/config` you can find .yaml files with options to configure parameters that are not available at runtime. You may for example decrease the resolution of density plots to increase performance if necessary.

## Contents
1. [Parametric Gaussian Regression](#parametric-gaussian-regression)
2. [Gaussian Process Regression](#gaussian-process-regression)

# Parametric Gaussian Regression

Run `python run_parametric_regression.py` from the ML-playground directory to open this interactive plot.\
\
The idea of parametric Gaussian regression is to build a parametric function 

```math
\begin{align*}
f(x) &= w_1 \cdot \phi_1(x) + w_2 \cdot \phi_2(x) + ...\\
&= \phi w
\end{align*}
```
from predefined features $\phi(x)$. These features are fixed and not part of the trainable parameters. The weights $w$ however are modeled as Gaussian random variables (RVs) and can be conditioned on data using  Bayesian inference. At every given x, $f(x)$ is also a Gaussian RV. This results in a 'probability distribution over functions'.

## What you can do

### Design parametric functions with random weights

While parametric regression can handle multidimensional data, it was decided to stick with 1D data for the sake of simplicity.\
\
On the left hand side of the plot you will find a panel used to set up the parametric function. The following features are available:

- Polynomial: $\phi(x; p) = x^{p} $
- Harmonic: $\phi(x; p_1, p_2) = \sin(p_1\pi \cdot x + p_2\dfrac{\pi}{2})$
- Gaussian: $\phi(x; \sigma, \mu) = \frac{1}{\sqrt{2\pi}\sigma} \cdot \exp(\frac{(x - \mu)^2}{2\sigma^2})$

Features can be added using the three buttons on top and removed by the red buttons next to the feature label. The parameters of features are set using the corresponding sliders.

### Edit the weight distribution
The weights are initialized independently with $\mu=0$ and $\sigma=1$. Their probability density is displayed in the left plot. As it is difficult to visualize high dimensional distributions, only a two dimensional subset of the actual weight space is plotted. If the model consists of more than two weights, the displayed weights can be changed by clicking on the x or y label of the plot. The distribution over $f(x)$ is shown in the right plot.\
The distribution on the left can be edited by dragging on the graph and the function distribution is updated in real time. Scrolling the mouse wheel scales the variance of the distribution. Pressing the following keys will change the scrolling behaviour:

- x: scale only along x-direction.
- y: scale only along y-direction.
- r: rotate the density around the mean (change covariance between the weights).

Doubleclicking anywhere on the figure resets the two currently displayed weights.

### Let the model learn

The panel at the top of the window is used to control the training of your model. A set of random samples is generated and added to the right plot automatically. You can generate new samples by clicking the 'Generate samples' button.\
Clicking on the 'Start/ stop training' button starts the training. This will condition the weight distribution on individual data points and update the weights as the training goes on. The way the data is selected can be changed by pressing the 'Sampling order' button. Possible orders are:
- Sequential: Select samples from lowest to highest $x$.
- Random: Select samples randomly.
- Least likely: Select the sample that is the least likely based on the current model.

You can try to find a parametric function that fits the data and observe the distribution approach the distribution of target samples. You may disturb the model during training by dragging the weights around and see how it reacts.

## What is happening

### How to obtain the function space distribution

Linear transformations of multivariate Gaussian random variables are Gaussian random variables. Their mean and variance are linear transformations of the original mean vector and covariance matrix:

```math
p(W) \sim \mathcal{N}(W; \mu, \Sigma)
\\
\Rightarrow\\
p(\phi W) \sim \mathcal{N}(\phi W; \phi \mu, \phi \Sigma \phi^T)
```

The projection from weight space into function space is linear by definition, so the function space distribution at every x-value is Gaussian and well defined.

### How to learn from data

Bayes theorem describes how a model changes after observing data.
The posterior conditional distribution of weights $W$ after observing the data $y$ is given as
```math
p(W|y) = \frac{p(y|W)p(W)}{\int p(y|W)p(W)dW}
```
where $p(y|W)$ is the probability to observe $y$ given the current model weights $W$ and $p(W)$ is the prior distribution over weights. The denominator is a normalization factor.\
Gaussian priors always produce Gaussian posteriors. For observations of linear projections of $W$ the posterior can be expressed as

```math
p(W|\phi W = y + \sigma) = \mathcal{N}(W;\mu', \Sigma')\\
```
with
```math
\mu' = \mu + \Sigma \phi (\phi^T \Sigma \phi + \epsilon)^{-1}(y - \phi^T \mu)\\
\Sigma' = \Sigma - \Sigma \phi(\phi^T \Sigma \phi + \epsilon)^{-1}\phi^T \Sigma
```
It is assumed that the data contains some Gaussian noise $\sigma$ with variance $\epsilon$: $\sigma = \mathcal{N}(0, \epsilon)$. It is evident from this equation that:\
\
Firstly, the data $y$ does only contribute to the posterior mean, in the form $y - \phi^T \mu$, which is the difference between the expected value of the model and the observed value. The covariance of the posterior does *not* depend on the data. The model will gain the same amount of confidence each step regardless of the actual data $y$. This can result in models that make very confident predictions that match the data very poorly.\
\
Secondly, $\epsilon$ contributes inversely to both posterior mean and covariance. Assuming large amounts of noise does therefore attenuate the effect the data has on the model and slows down the training. Assuming $\epsilon = 0$ on the other hand leads to faster convergence, but it carries the risk of breaking the model, as it tries to fit every datapoint exactly. This is only possible if the data has a low number of points or the datapoints are perfectly aligned, which is unlikely to happen in practice due to measurement noise and machine precision.\
 Here, $\epsilon$ is equal to the variance of the noise in the training data by default.


# Gaussian Process Regression

Run `python run_gp_regression.py` from the ML-playground directory to open this interactive plot.\
\
Similar to the linear Gaussian model, Gaussian processes (GPs) do also define a Gaussian distribution over values `y` for every `x`. However they do not rely on a set of features, instead they are defined by a mean function $m(x)$ and a covariance or kernel function $k(x, x')$. For every finite set $\{x_i\}, x_i \in \mathbb{R}$, a GP defines a multivariate Gaussian distribution with mean vector $m_i = m(x_i)$ and covariance matrix $k_{ij}=k(x_i, x_j)$. Consequently, the kernel function must be a positive semidefinite function.\
\
The distribution over a function value `y` is Gaussian with $p(f(x)) = \mathcal{N}(m(x), k(x, x))$. The 'diagonal' of the kernel at $x = x'$ therefore defines the variance of function values when taking multiple samples from the GP at a specific `x`. The 'off-diagonal' elements $x\neq x'$ define how similar samples at different `x` from *the same* GP are. If the covariant terms are decaying fast, it means that `y` at different `x` values have distributions with low covariance and the GP creates very noisy functions. With slower decaying kernel values, the GP functions get smoother.

## What you can do

### Build a simple Gaussian process

On the left side of the window you find a panel which lets you select one or more kernel functions and tweak their parameters. If more than one kernel is selected, the kernel functions are added or multiplied elementwise. As the sum and element-wise (Hadamard) product of two positive semidefinite matrices is again positive semidefinite, the results of this are valid kernels as well.

<u>Radial basis function (RBF) kernel:</u>\
$f_{RBF}(x, x') = exp(\frac{(x - x')^2}{2\epsilon^2})$\
The parameter $\epsilon$ defines the smoothness of the functions produced by the GP. Too small $\epsilon$ lead to GPs that do not generalize well on data, as they allow functions to quickly fluctuate and assume correlation with data points only in very close proximity to the points. Too large $\epsilon$ produce very smooth curves, which may be too smooth to follow the data in a meaningful fashion.\
There are two RBF kernels available, so you comapre how adding and multiplying them with different parameters changes the distribution.\
\
<u>Polynomial kernel:</u>\
$f_{poly}(x, x') = (x \cdot x' + c)^n$\
This kernel has an constant offset $c$ and a power $n$ as parameters. The functions returned by this process are polynomials of degree $n$. It can for example be used to model long term trends in the data and be added to an RBF kernel which models short term fluctuations.

### Observe kernel function, mean function and GP samples

On the right-hand side of the panel are two plots: the kernel function on the left, the distribution over function values on the right. The right plot also contains multiple lines: a solid blue line shows the mean function of the GP distribution, two dashed blue lines indicate the mean $\pm$ two standard deviations. The purple line shows a random sample from the GP. It can be resampled by left-clicking on the plot. See how changing the input scale of an RBF kernel does not change the overall distribution of samples, but heavily affects how individual samples look.

## What is happening?

### How to draw samples from a GP
As discussed previously, the distribution over function values $p(f(x))$ is a Gaussian distribution with mean defined by the mean function $m$ and variance defined by the kernel function $k$:
```math
p(f(x)) = \mathcal{N}(f(x); m(x), k(x, x))
```

### How to learn from data
When observing data $y$ at $x$, the mean and kernel functions can be updated in a similar fashion to linear Gaussian models. 
```math
\begin{align*}
m'(a) &= m(a) + k(a, x) (k(x, x) + \epsilon)^{-1} (y - m(x))\\
k'(a, b) &= k(a, b) - k(a, x) (k(x, x) + \epsilon)^{-1} k(x, b)
\end{align*}
```
where $x$ and $y$ are the fixed observed datapoint and $a$ and $b$ are the function arguments of the posterior mean and kernel function. I encourage you to go to `ML-playground/src/config/gp_regression_config.yaml` and change the `model_noise_amount` parameter and watch how it affects the model predictions.\
\
\
Most of the concepts and some inspiration for the code are from [this lecture](https://www.youtube.com/playlist?list=PL05umP7R6ij0hPfU7Yuz8J9WXjlb3MFjm) ([CC BY-SA 4.0](https://creativecommons.org/licenses/by/4.0/)) by Prof. Philipp Hennig which i strongly recommend to anyone interested in probabilistic machine learning.