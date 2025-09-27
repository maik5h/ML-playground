import matplotlib.pyplot as plt

from .model import ConditionalGaussianProcess
from .model_gui import GPKernelPlot, GPFunctionSpacePlot, GPFunctionSamples
from .gp_controls import GPController
from ..math_utils import InteractiveTrainer
from ..math_utils import FeatureSampleGenerator
from ..config import Config, load_config


def run_gp_regression() -> None:
    load_config()
    plt.rcParams['font.size'] = 5

    # Initialize and customize matplotlib figure.
    fig, (ax_controls, ax_kernel, ax_func) = plt.subplots(
        nrows=1,
        ncols=3,
        dpi=200,
        figsize=(6, 3),
        gridspec_kw={'width_ratios': [1, 2, 2]}
        )
    plt.subplots_adjust(wspace=0.3, left=0.06, right=0.98, top=0.75)
    ax_func.set_xlim(Config.function_space_xlim)
    ax_func.set_ylim(Config.function_space_ylim)
    ax_controls.axis('off')

    # Model, trainer and model visuals.
    model = ConditionalGaussianProcess()
    data_gen = FeatureSampleGenerator(
        n_samples=10,
        noise_amount=0.3,
        xlim=[-4, 4],
        max_weight=8
    )
    trainer_x = ax_kernel.get_position().x0
    trainer = InteractiveTrainer(
        (trainer_x, 0.85, 0.98-trainer_x, 0.1),
        model,
        data_gen,
        fig,
        ax_func
        )
    controls = GPController((0.02, 0.05, 0.2, 0.9), model)
    model_gui = GPKernelPlot(ax_kernel, model)
    func_plot = GPFunctionSpacePlot(ax_func, model)
    samples = GPFunctionSamples(ax_func, model)

    plt.show()
