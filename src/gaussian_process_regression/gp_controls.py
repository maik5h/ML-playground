from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event
from matplotlib.widgets import Button, Slider

from ..math_utils import (
    KernelInterface,
    RBFKernel,
    PolynomialKernel,
    WienerProcessKernel,
    IntegratedWienerProcessKernel,
    KernelSum,
    KernelProduct,
)
from .model import ConditionalGaussianProcess
from ..gui_utils import add_frame


# Button colors. Colors for inactive Buttons are matplotlib default
# colors.
COLOR_ACTIVE = "#ACE57A"
COLOR_ACTIVE_HOVER = "#d1f9ae"
COLOR_INACTIVE = '0.85'
COLOR_INACTIVE_HOVER = '0.95'


class KernelController:
    """Shows a UI to change the attributes of a KernelInterface object.

    The UI consists of a title, a slider for each kernel parameter and
    a button to activate and deactivate the kernel. KernelController
    instances are stored in a GPController.
    """
    def __init__(
            self,
            position: tuple[float, float, float, float],
            kernel: KernelInterface,
            target_product: KernelProduct,
            target_sum: KernelSum,
            title: str,
            slider_labels: list[str]
        ):
        """Parameters
        ----------

        position: `tuple[float, float, float, float]`
            Initial position of the controller interface on the figure,
            in [left, lower, width, height] coordinates.
        kernel: `KernelInterface`
            KernelInterface object that is controlled by this controller.
        target_product: `KernelProduct`
            KernelProduct object to which this instance will add its
            kernel on activate() and remove it on deactivate().
        target_sum: `KernelSum`
            KernelSum object to which this instance will add its kernel
            on activate() and remove it on deactivate().
        title: `str`
            Title displayed on the figure.
        slider_labels: `list[str]`
            List of labels for each Slider.
        """
        self._kernel = kernel
        self._target_product = target_product
        self._target_sum = target_sum
        self._active = False
        self._callbacks: set[Callable[[], None]] = set()

        # Create sliders and button in the following arrangement:
        # Two columns with intersection at 60% of the width.
        #   Left column:
        #       Two rows with slider a and slider b.
        #       Using 18% of the height and being 15% of the height
        #       apart, starting from the bottom.
        #   Right column:
        #       Activate button.
        # Values are arbitrary but look nice.
        x, y, width, height = position
        self._ax_slider_a = plt.axes(
            [x, y + 0.33*height, 0.6*width, 0.18*height]
        )
        self._ax_slider_b = plt.axes(
            [x, y, 0.6*width, 0.2*height]
        )
        self._ax_button = plt.axes(
            [x + 0.75*width, y, 0.25*width, height]
        )

        # Add title and slider labels to a new axis dedicated for text.
        ax_text = plt.axes(position)
        ax_text.axis('off')
        self._title = ax_text.text(
            x=0,
            y=1,
            s=title,
            verticalalignment='top'
        )
        self._label_a = ax_text.text(
            x=0,
            y=0.6,
            s=slider_labels[0],
            verticalalignment='center'
        )
        self._label_b = ax_text.text(
            x=0,
            y=0.25,
            s=slider_labels[1],
            verticalalignment='center'
        )

        # Create activate button.
        self._activate_button = Button(
            self._ax_button,
            'Off',
            color=COLOR_INACTIVE,
            hovercolor=COLOR_INACTIVE_HOVER
        )
        self._activate_button.on_clicked(self._on_button_clicked)

        # Create Sliders for the kernel parameters. The appropriate
        # limits are returned by _kernel.get_parameter_limits().
        lims = self._kernel.get_parameter_limits()
        self._slider_a = Slider(
            self._ax_slider_a,
            label='',
            handle_style={'size': 4},
            valmin=lims[0][0],
            valmax=lims[0][1],
            valstep=lims[0][2],
            valinit=self._kernel.parameters[0]
        )
        self._slider_a.on_changed(self._on_slider_a_changed)
        self._slider_b = Slider(
            self._ax_slider_b,
            label='',
            handle_style={'size': 4},
            valmin=lims[1][0],
            valmax=lims[1][1],
            valstep=lims[1][2],
            valinit=self._kernel.parameters[1]
        )
        self._slider_b.on_changed(self._on_slider_b_changed)

    def _on_button_clicked(self, _event: Event) -> None:
        """Switches the active state of this instance.
        """
        if self._active:
            self.deactivate()
        else:
            self.activate()

    def add_callback(self, func: Callable[[], None]) -> None:
        """Add a function to call everytime the controller state changes.
        """
        self._callbacks.add(func)

    def _process_callbacks(self) -> None:
        for func in self._callbacks:
            func()

    def _on_slider_a_changed(self, val: float) -> None:
        self._kernel.set_parameter(0, val)
        if self._active:
            self._process_callbacks()

    def _on_slider_b_changed(self, val: float) -> None:
        self._kernel.set_parameter(1, val)
        if self._active:
            self._process_callbacks()

    def activate(self) -> None:
        """Sets this KernelController active.

        This will:
        - Add the corresponding kernel to the given KernelSum and
        KernelProduct.
        - Update the activate button on the plot.
        - Call the callback functions. 
        """
        self._active = True
        self._activate_button.color = COLOR_ACTIVE
        self._activate_button.hovercolor = COLOR_ACTIVE_HOVER
        self._activate_button.label.set_text('On')
        self._target_product.add_kernel(self._kernel)
        self._target_sum.add_kernel(self._kernel)
        self._process_callbacks()

    def deactivate(self) -> None:
        """Sets this KernelController inactive.

        This will:
        - Remove the corresponding kernel to the given KernelSum and
        KernelProduct.
        - Update the activate button on the plot.
        - Call the callback functions. 
        """
        # Make sure at least one kernel is active at all times.
        # _kernel_product and _kernel_sum always have the same size.
        if len(self._target_product) == 1:
            return

        self._active = False
        self._activate_button.color = COLOR_INACTIVE
        self._activate_button.hovercolor = COLOR_INACTIVE_HOVER
        self._activate_button.label.set_text('Off')
        self._target_product.remove_kernel(self._kernel)
        self._target_sum.remove_kernel(self._kernel)
        self._process_callbacks()


class GPController:
    """A graphical interface to edit kernels.

    Lets the user edit kernel parameters, switch kernels on and off and
    choose if the selected kernels are added or multiplied together.
    The kernel designed by the user is accessible as the `kernel`
    property of this class.
    """
    def __init__(
            self,
            pos: tuple[float, float, float, float],
            model: ConditionalGaussianProcess
        ):
        """Parameters
        ----------

        pos: `tuple[float, float, float, float]`
            Size and position of the GPController in
            [left, right, width, height] coordinates.
        model: `ConditionalGaussianProcess`
            A Gaussian process model to be controlled by this
            controller. Whenever the user interacts with the
            controller, the prior kernel of the model will be updated
            and the model is refreshed.
        """
        self._pos = pos
        self._model = model

        # Create Buttons to switch between summation and
        # multiplication.
        pos = self._get_layout(row=0)
        ax_sum_button = plt.axes((pos[0], pos[1], 0.5*pos[2], pos[3]))
        self._sum_button = Button(
            ax=ax_sum_button,
            label='Add Kernels',
            color=COLOR_ACTIVE,
            hovercolor=COLOR_ACTIVE,
        )
        ax_prod_button = plt.axes((pos[0] + 0.5*pos[2], pos[1], 0.5*pos[2], pos[3]))
        self._product_button = Button(
            ax=ax_prod_button,
            label='Multiply Kernels',
            color=COLOR_INACTIVE,
            hovercolor=COLOR_INACTIVE_HOVER
        )
        self._sum_button.on_clicked(lambda event: self._switch_mode(True))
        self._product_button.on_clicked(lambda event: self._switch_mode(False))

        # Initialize kernel product and sum objects without kernels.
        self._kernel_product = KernelProduct(set())
        self._kernel_sum = KernelSum(set())

        # Create KernelController instances connected to the sum and
        # product objects.
        self._controllers = [
            KernelController(
                self._get_layout(row=1),
                RBFKernel(in_scale=0.1, out_scale=1),
                self._kernel_product,
                self._kernel_sum,
                title='RBF kernel 1',
                slider_labels=['in scale', 'out scale']
            ),
            KernelController(
                self._get_layout(row=2),
                RBFKernel(in_scale=0.5, out_scale=1),
                self._kernel_product,
                self._kernel_sum,
                title='RBF kernel 2',
                slider_labels=['in scale', 'out scale']
            ),
            KernelController(
                self._get_layout(row=3),
                PolynomialKernel(power=1, offset=0.1),
                self._kernel_product,
                self._kernel_sum,
                title='Polynomial kernel',
                slider_labels=['power', 'offset']
            ),
            KernelController(
                self._get_layout(row=4),
                WienerProcessKernel(0, 1),
                self._kernel_product,
                self._kernel_sum,
                title='Wiener process kernel',
                slider_labels=['x0', 'output scale']
            ),
            KernelController(
                self._get_layout(row=5),
                IntegratedWienerProcessKernel(0, 1),
                self._kernel_product,
                self._kernel_sum,
                title='Int. Wiener p. kernel',
                slider_labels=['x0', 'output scale']
            )
        ]

        # Model must be refreshed when a KernelController state
        # changes.
        for controller in self._controllers:
            controller.add_callback(model.refresh)

        # Activate the first kernel.
        self._controllers[0].activate()

        # Add kernels by default.
        self._add = True
        self._model._prior_gp._kernel = self._kernel_sum

        # Draw a frame around all elements.
        add_frame(self._pos)

    def _get_layout(self, row: int) -> tuple[float, float, float, float]:
        """Return the coordinates of a kernel controller at `row`.

        The GPController layout is arranged in rows. The first row is
        used for buttons to switch between sum and multiply modes.
        Rows below are filled with KernelController instances.

        Returns
        -------

        Coordinates in [left, bottom, width, height].
        """
        # There is space for six rows of elements along y.
        x, y, w, h = self._pos
        return (x, y+0.9*h - 0.166*h*row, w, 0.1*h)

    def _switch_mode(self, sum_button: bool) -> None:
        """Switch between adding and multiplying kernels.

        Switches from sum mode to multiply mode, if multiply button was
        pressed. Switches from multiply to adding mode if sum button
        was pressed. `sum_button` indicates if the request comes from
        the button associated with the sum mode or multiply mode.
        """
        if self._add and not sum_button:
            self._add = False
            self._sum_button.color = COLOR_INACTIVE
            self._sum_button.hovercolor = COLOR_INACTIVE_HOVER
            self._product_button.color = COLOR_ACTIVE
            self._product_button.hovercolor = COLOR_ACTIVE
            self._model._prior_gp._kernel = self._kernel_product
            # Switching between sum and multiplication does not affect
            # the model if only one kernel is activated. Process
            # callbacks only if necessary.
            if len(self._kernel_product) > 1:
                self._model.refresh()
        elif not self._add and sum_button:
            self._add = True
            self._sum_button.color = COLOR_ACTIVE
            self._sum_button.hovercolor = COLOR_ACTIVE
            self._product_button.color = COLOR_INACTIVE
            self._product_button.hovercolor = COLOR_INACTIVE_HOVER
            if len(self._kernel_product) > 1:
                self._model._prior_gp._kernel = self._kernel_sum
                self._model.refresh()
