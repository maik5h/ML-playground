import matplotlib.pyplot as plt

from .model import ConditionalGaussianProcess
from .model_gui import GPKernelPlot, GPFunctionSpacePlot, GPFunctionSamples
from .gp_controls import GPController
from ..math_utils import InteractiveTrainer
from ..math_utils import FeatureSampleGenerator
from ..config import GPRegressionConfig as Config
from ..config import load_gp_regression_config


def run_gp_regression() -> None:
    plt.rcParams['font.size'] = 5

    load_gp_regression_config()

    # Initialize and customize matplotlib figure.
    fig, (ax_controls, ax_kernel, ax_func) = plt.subplots(
        nrows=1,
        ncols=3,
        dpi=200,
        figsize=(6, 3),
        gridspec_kw={'width_ratios': [1, 2, 2]}
    )
    plt.subplots_adjust(wspace=0.3, left=0.06, right=0.98, top=0.75)

    # Show the kernel evaluated on a slightly larger interval than the
    # function.
    kernel_lim = (Config.function_space_xlim[0]*1.1,
                  Config.function_space_xlim[1]*1.1)
    ax_kernel.set_xlim(kernel_lim)
    ax_kernel.set_ylim(kernel_lim)
    ax_func.set_xlim(Config.function_space_xlim)
    ax_func.set_ylim(Config.function_space_ylim)
    ax_controls.axis('off')

    # Model is a conditional Gaussian process.
    model = ConditionalGaussianProcess(epsilon=Config.model_noise_amount)

    # Initialize a trainer instance with a sample generator.
    data_gen = FeatureSampleGenerator(
        n_samples=Config.number_target_samples,
        noise_amount=Config.target_noise_amount,
        xlim=Config.function_space_xlim,
        max_weight=8
    )
    trainer_x = ax_kernel.get_position().x0
    trainer = InteractiveTrainer(
        area=(trainer_x, 0.85, 0.98-trainer_x, 0.1),
        model=model,
        data_generator=data_gen,
        fig=fig,
        ax=ax_func,
        timestep=Config.time_per_learning_step,
        batch_size=Config.samples_per_learning_step
    )

    # Initialize GP control panel.
    controls = GPController((0.02, 0.05, 0.2, 0.9), model)

    # Initialize GP visuals.
    kernel_plot = GPKernelPlot(
        ax=ax_kernel,
        model=model,
        resolution=Config.function_space_resolution,
        vmax=Config.colormap_vmax
    )
    func_plot = GPFunctionSpacePlot(
        ax=ax_func,
        model=model,
        resolution=Config.function_space_resolution,
        vmax=Config.colormap_vmax
    )
    funcsamples = GPFunctionSamples(
        ax=ax_func,
        model=model,
        number_samples=Config.number_function_samples
    )

    plt.show()
