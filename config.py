import json
from logging import warning

class Config:
    """
    Class to store and access the configuration of the regression playground.
    """
    # ----- Visuals -----
    # The number of samples per axis used to render the weight space distribution.
    weight_space_samples: int = 100

    # The number of samples per axis used to render the function space distribution.
    function_space_samples: int = 300

    # Limits of weight and function space plots.
    weight_space_xlim: list[float] = [-4, 4]
    weight_space_ylim: list[float] = [-4, 4]
    function_space_xlim: list[float] = [-5, 5]
    function_space_ylim: list[float] = [-15, 15]
    
    # The vmax value of both plots.
    colormap_vmax: float = 0.3


    # ----- Controls -----
    # The sensitivity of the mouse wheel when editing the Gaussians covariance.
    mouse_wheel_sensitivity: float = 1


    # ----- learning -----
    # The number of samples from the function to be approximated.
    number_target_samples: int = 100

    # Variance of the Gaussian noise distribution added to the target samples.
    target_noise_amount: float = 1

    # The time in milliseconds between weight updates. This is the minimum time a step will take, the
    # actual time is limited by the time it takes to update the weights and to plot the distributions.
    time_per_learning_step: float = 300

def load_config(path='./config.json'):
    """
    Copies the configuration from the json file at path into the static attributes of the Config class.
    """
    try:
        with open(path, 'r') as f:
            cfg = json.load(f)

    except FileNotFoundError:
        warning('Config file not found, using default configuration.')
        return

    # The actual config values are separated into several sub-dictionaries for clearer structure.
    for sub_cfg in cfg.values():
        for key in sub_cfg:
            setattr(Config, key, sub_cfg[key])

    # Check if loaded values are valid:

    # Mouse wheel sensitivities larger than three might couse division by zero or negative covariance
    # values by scaling too fast.
    if Config.mouse_wheel_sensitivity > 3:
        Config.mouse_wheel_sensitivity = 3
        warning('Mouse wheel sensitivity larger than 3 is not supported. Sensitivity of 3 is used.')

