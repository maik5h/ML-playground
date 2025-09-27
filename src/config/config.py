import json
from logging import warning
from pathlib import Path
from typing import Optional


class Config:
    """
    Class to store and access the configuration of the regression playground.
    """
    # ----- Visuals -----
    # The number of samples per axis used to render the weight space
    # distribution.
    weight_space_samples: int = 100

    # The number of samples per axis used to render the function space
    # distribution.
    function_space_samples: int = 300

    # Limits of weight and function space plots.
    weight_space_xlim: list[float] = [-4, 4]
    weight_space_ylim: list[float] = [-4, 4]
    function_space_xlim: list[float] = [-5, 5]
    function_space_ylim: list[float] = [-15, 15]
    
    # The vmax value of both plots.
    colormap_vmax: float = 0.3


    # ----- Controls -----
    # The sensitivity of the mouse wheel when editing the Gaussians
    # covariance.
    mouse_wheel_sensitivity: float = 1


    # ----- learning -----
    # The number of samples from the function to be approximated.
    number_target_samples: int = 100

    # Variance of the Gaussian noise distribution added to the target
    # samples.
    target_noise_amount: float = 1

    # Amount of noise assumed by the model.
    model_noise_amount: float = 1

    # The time in milliseconds between weight updates. This is the
    # minimum time a step will take, the actual time is limited by the
    # time it takes to update the weights and to plot the
    # distributions.
    time_per_learning_step: float = 300

    # Batch size of the dataloader.
    samples_per_learning_step: int = 1


def load_config(path: Optional[str] = None) -> None:
    """
    Copies the configuration from the json file at `path` into the
    static attributes of the Config class.
    """
    # The default location of the config file is the src\config\
    # directory which also contains this file.
    path = path or Path(__file__).parent.resolve() / 'config.json'

    try:
        with open(path, 'r') as f:
            cfg: dict = json.load(f)

    except FileNotFoundError:
        warning('Config file not found, using default configuration.')
        return

    # The actual config values are separated into several
    # sub-dictionaries for clearer structure.
    for sub_cfg in cfg.values():
        for key in sub_cfg:
            setattr(Config, key, sub_cfg[key])

    # Check if loaded values are valid:

    # Mouse wheel sensitivities larger than three might couse division
    # by zero or negative covariance values by scaling too fast.
    if Config.mouse_wheel_sensitivity > 3:
        Config.mouse_wheel_sensitivity = 3
        warning(
            'Mouse wheel sensitivity larger than 3 is not supported. '
            + 'Sensitivity of 3 is used.'
        )

    # It is good practice to let the model assume at least some
    # uncertainty in the data, else the model might be looking for an
    # impossible solution (e.g. fitting a polynomial of degree N<M to
    # exactly satisfy M data points).
    if Config.model_noise_amount < 1e-6:
        Config.model_noise_amount = 1e-6
        warning(
            'Small model noise found in config file. Small noise may '
            + 'lead to invalid covariance matrices. Model noise '
            + 'amount was set to 10^-6.'
        )
