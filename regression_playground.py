from gaussians import InteractiveGaussian
from features import *
import numpy as np
import matplotlib.pyplot as plt
from feature_controls import FeatureVectorController
from config import load_config, Config


def start_regression():
    """
    Starts the regression session.
    Creates noisy samples from the default function, creates the plot axes and connects events to
    InteractiveGaussian methods.
    """

    load_config()

    plt.rcParams['font.size'] = 5

    x = np.linspace(Config.function_space_xlim[0], Config.function_space_xlim[1], Config.number_target_samples)

    # Create a linear function with random offset and slope.
    offset = np.random.rand() * 5 - 2.5
    slope = np.random.rand() * 6 - 3

    # Create noise vector to be added to samples.
    noise_amount = 0.5
    sample_noise = np.random.normal(loc=0, scale=noise_amount, size=Config.number_target_samples)
    samples = (offset + x * slope) + sample_noise

    # Set up plots for weight and function space.
    # Set ip three axes:
    # - axis for control panel, only reserving space, no plots happening here.
    # - axis for weight space plot.
    # - axis for function space plot.
    fig, (ax_controls, ax_weight, ax_func) = plt.subplots(nrows=1, ncols=3, dpi=200, figsize=(6, 3), gridspec_kw={'width_ratios': [1, 2, 2]})
    ax_weight.set_xlim(Config.weight_space_xlim)
    ax_weight.set_ylim(Config.weight_space_ylim)
    ax_func.set_xlim(Config.function_space_xlim)
    ax_func.set_ylim(Config.function_space_ylim)
    ax_weight.set_xlabel('$w_1$')
    ax_weight.set_ylabel('$w_2$')
    ax_func.set_xlabel('$x$')
    ax_func.set_ylabel('$f_w(x)$')
    ax_controls.axis('off')

    # Forward default features and axes to the model.
    init_features = FeatureVector([PolynomialFeature(power=2), SineFeature(n_pi=1), PolynomialFeature(1)])
    model = InteractiveGaussian(init_features, ax_weight, ax_func)

    feature_controller = FeatureVectorController(fig, model)

    # Connect matplotlib callbacks.
    fig.canvas.mpl_connect('button_press_event', model.on_mouse_button_down)
    fig.canvas.mpl_connect('motion_notify_event', model.on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', model.on_mouse_button_up)
    fig.canvas.mpl_connect('scroll_event', model.on_scroll_event)

    # Let the model plots its distributions
    model.plot()

    # Plot sampled datapoints on top of function space distribution.
    ax_func.plot(x, samples, 'k.', markersize=2, label='samples')
    ax_func.legend()

    plt.tight_layout()
    plt.show()