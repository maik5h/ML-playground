from gaussians import InteractiveGaussian
from features import *
import matplotlib.pyplot as plt
from feature_controls import FeatureVectorController
from config import load_config, Config
from learning import InteractiveTrainer


def start_regression() -> None:
    """
    Starts the regression session.
    Creates noisy samples from the default function, creates the plot axes and connects events to
    InteractiveGaussian methods.
    """

    load_config()

    plt.rcParams['font.size'] = 5

    # Set up plots for weight and function space.
    # Set ip three axes:
    # - axis for control panel, only reserving space, no plots happening here.
    # - axis for weight space plot.
    # - axis for function space plot.
    fig, (ax_controls, ax_weight, ax_func) = plt.subplots(nrows=1, ncols=3, dpi=200, figsize=(6, 3), gridspec_kw={'width_ratios': [1, 2, 2]})
    plt.subplots_adjust(wspace=0.3, left=0.06, right=0.98, top=0.75)
    ax_weight.set_xlim(Config.weight_space_xlim)
    ax_weight.set_ylim(Config.weight_space_ylim)
    ax_func.set_xlim(Config.function_space_xlim)
    ax_func.set_ylim(Config.function_space_ylim)
    ax_controls.axis('off')

    # Forward default features and axes to the model.
    init_features = FeatureVector([PolynomialFeature(power=0, offset=0), PolynomialFeature(power=1, offset=0)])
    model = InteractiveGaussian(init_features, ax_weight, ax_func)

    feature_controller = FeatureVectorController(fig, model)

    trainer_x = ax_weight.get_position().x0
    trainer = InteractiveTrainer((trainer_x, 0.85, 0.98, 0.95), model, fig, ax_func)

    # Connect matplotlib callbacks.
    fig.canvas.mpl_connect('button_press_event', model.on_mouse_button_down)
    fig.canvas.mpl_connect('motion_notify_event', model.on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', model.on_mouse_button_up)
    fig.canvas.mpl_connect('key_press_event', model.on_key_pressed)
    fig.canvas.mpl_connect('key_release_event', model.on_key_released)
    fig.canvas.mpl_connect('scroll_event', model.on_scroll_event)
    fig.canvas.mpl_connect('key_press_event', trainer.on_key_press)

    # Let the model plot its distributions.
    model.plot()

    plt.show()
