from gaussians import InteractiveGaussian
from features import *
import numpy as np
import matplotlib.pyplot as plt


def start_regression():
    """
    Starts the regression session.
    Creates noisy samples from the default function, creates the plot axes and connects events to
    InteractiveGaussian methods.
    """

    number_samples = 100
    xlim_weight = (-4, 4)
    ylim_weight = (-4, 4)
    xlim_func = (-5, 5)
    ylim_func = (-15, 15)

    x = np.linspace(xlim_func[0], xlim_func[1], number_samples)

    # Create a linear function with random offset and slope.
    offset = np.random.rand() * 5 - 2.5
    slope = np.random.rand() * 6 - 3

    # Create noise vector to be added to samples.
    noise_amount = 0.5
    sample_noise = np.random.normal(loc=0, scale=noise_amount, size=number_samples)
    samples = (offset + x * slope) + sample_noise

    # Set up plots for weight and function space.
    fig, (ax_weight, ax_func) = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(6, 3))
    ax_weight.set_xlim(xlim_weight)
    ax_weight.set_ylim(ylim_weight)
    ax_func.set_xlim(xlim_func)
    ax_func.set_ylim(ylim_func)
    ax_weight.set_xlabel('$w_1$')
    ax_weight.set_ylabel('$w_2$')
    ax_func.set_xlabel('$x$')
    ax_func.set_ylabel('$f_w(x)$')

    # Forward default features and axes to the model.
    # features = FeatureVector([SineFeature(n_pi=2), PolynomialFeature(power=1)])
    features = FeatureVector([PolynomialFeature(power=2), PolynomialFeature(power=0), PolynomialFeature(1)])
    model = InteractiveGaussian(features, ax_weight, ax_func)

    # Connect matplotlib callbacks.
    fig.canvas.mpl_connect('button_press_event', model.on_mouse_button_down)
    fig.canvas.mpl_connect('motion_notify_event', model.on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', model.on_mouse_button_up)

    # Let the model plots its distributions
    model.plot()

    # Plot sampled datapoints on top of function space distribution.
    ax_func.plot(x, samples, 'k.', markersize=2, label='samples')

    ax_weight.legend()
    ax_func.legend()

    plt.tight_layout()
    plt.show()