import matplotlib.pyplot as plt

from .model import ParametricGaussian
from .model_gui import WeightSpaceGUI, FunctionSpacePlot
from ..math_utils import FeatureVector
from ..math_utils import PolynomialFeature
from ..config import Config
from .feature_controls import FeatureVectorController
from ..math_utils import InteractiveTrainer
from ..math_utils import FeatureSampleGenerator


def run_parametric_regression() -> None:
    """
    Starts the regression session. Opens a window with an interactive
    plot allowing to define a Gaussian model from given feature
    functions, edit the model weights through mouse inputs and start
    the learning process.
    """
    plt.rcParams['font.size'] = 5

    # Set up plots for weight and function space.
    # Set ip three axes:
    # - axis for control panel, only reserving space, no plots there.
    # - axis for weight space plot.
    # - axis for function space plot.
    fig, (ax_controls, ax_weight, ax_func) = plt.subplots(
        nrows=1,
        ncols=3,
        dpi=200,
        figsize=(6, 3),
        gridspec_kw={'width_ratios': [1, 2, 2]}
    )
    plt.subplots_adjust(wspace=0.3, left=0.06, right=0.98, top=0.75)
    ax_weight.set_xlim(Config.weight_space_xlim)
    ax_weight.set_ylim(Config.weight_space_ylim)
    ax_func.set_xlim(Config.function_space_xlim)
    ax_func.set_ylim(Config.function_space_ylim)
    ax_controls.axis('off')

    # Create initial features and model.
    init_features = FeatureVector(
        [PolynomialFeature(power=0), PolynomialFeature(power=1)]
    )
    model = ParametricGaussian(init_features, Config.model_noise_amount)

    # Create graphic representation of the model.
    weight_space_UI = WeightSpaceGUI(model, fig, ax_weight)
    function_space_plot = FunctionSpacePlot(model, ax_func)

    # Create interface to build the feature function.
    feature_controller = FeatureVectorController(
        area=(0.02, 0.05, 0.2, 0.9),
        gaussian=model
    )

    # Random generator for training data, configured such that the
    # target function lies within the displayed weight space limits.
    data_gen = FeatureSampleGenerator(
        n_samples=Config.number_target_samples,
        noise_amount=Config.target_noise_amount,
        xlim=Config.function_space_xlim,
        max_weight=Config.weight_space_xlim[1] * 0.8
    )

    # Create interface controlling the training of the model.
    trainer_x = ax_weight.get_position().x0
    trainer = InteractiveTrainer(
        area=(trainer_x, 0.85, 0.98-trainer_x, 0.1),
        model=model,
        data_generator=data_gen,
        fig=fig,
        ax=ax_func
    )

    plt.show()
