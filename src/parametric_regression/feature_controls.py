from typing import Union, Literal, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from .model import InteractiveGaussian
from ..math_utils import Feature, PolynomialFeature, HarmonicFeature, GaussFeature
from ..gui_utils import create_button


# The maximum number of features that the FeatureVector class can hold.
MAX_NUMBER_FEATURES = 7

# The distance relative to the full figure in which buttons are spaced in y-direction.
Y_SPACING = 0.09

def pos(element: Literal['button', 'controller'], row: int) -> tuple[float, float, float, float]:
    """
    Returns the position of an element as a list in XYWH notation.
    Elements are arranged in rows, spaced by Y_SPACING. Buttons are positioned above
    Controllers, it is assumed that a total of three buttons is placed on top,
    consequently the Controller rows are shifted by 3.

    Attributes
    ----------

    element: `Literal['button', 'controller']`
        The type of element. Buttons are on top, controllers below.
    row: `int`
        The desired row index of the element.
    """
    x = 0.02
    y = 0.85
    w = 0.2
    h = 0.08

    row_offset = 3 if element == 'controller' else 0

    return (x, y - (row + row_offset) * Y_SPACING, w, h)


def hide_widget(widget: Union[Button, Slider], callback_id: Optional[int] = None) -> None:
    """
    Hides the given widget and takes care that no inputs are accepted while it is hidden.

    Attributes
    ----------
    widget: `Union[Button, Slider]`
        The widget to be hidden, can be a Button or a Slider.
    callback_id: `Optional[int]`
        id of the function connected to this instance. Can be None if no function has been connected.
    """
    if callback_id is not None:
        widget.disconnect(callback_id)

    if isinstance(widget, Slider):
        widget.dragging = False

    widget.ax.set_visible(False)

class FeatureController:
    """
    FeatureController is a collection of elements that enable editing of a single Feature object.
    It consists of a label displaying the function definition of the feature ("$\phi_i(x)=...$"),
    a slider to set the function parameter of the feature and a button to disable the feature.
    """
    def __init__(self, parent_controller, position: Sequence[float], idx: int):
        """
        Parameters
        ----------
        parent_controller: `FeatureVectorController`
            The parent FeatureVectorController instance that is holding this FeatureController.
        position: `Sequence[float]`
            Position of this FeatureController relative to the full figure in XYWH notation.
        idx: `int`
            Index of this FeatureController in the list of existing feature controllers.
        """
        self._idx = idx
        self._parent_controller = parent_controller

        # Stores a reference to the corresponding Feature. Is set once this FeatureController is
        # activated through self.set_feature().
        self._feature: Feature = None

        # Initialize required axes with arbitrary values and set correct values via
        # self.set_position later.
        ax_label    = plt.axes([0, 0, 1, 1])
        self._ax_slider_a = plt.axes([0, 0, 1, 1])
        self._ax_slider_b = plt.axes([0, 0, 1, 1])
        self._ax_button   = plt.axes([0, 0, 1, 1])

        ax_label.axis('off')

        # Initialize label and Buttons. Label text can be changed during runtime, Button does not have to be
        # changed at all.
        self._label      = ax_label.text(0, 0, '', verticalalignment='top')
        self._x_button   = Button(self._ax_button, 'X', color='#DF908F', hovercolor='#F5BDBC')

        # Do not initialize sliders, as they are specific to the feature that is displayed and can not be
        # edited during runtime. They are initialized in self.set_feature(...).
        self._slider_a: Slider = None
        self._slider_b: Slider = None

        self.set_position(position)

        # Callback ids of the methods connected to slider and button. Necessary to disconnect them on self.hide().
        self._slider_id: int = None
        self._button_id: int = None

        # Hide widgets by default and only display when needed.
        self._label.axes.set_visible(False)
        self._ax_slider_a.set_visible(False)
        self._ax_slider_b.set_visible(False)
        hide_widget(self._x_button, self._button_id)
    
    def set_position(self, new_position: Sequence[float]) -> None:
        """
        Sets this FeatureController to the given new position by updating the axes of all members
        to fit into the new position.
        """
        x       = new_position[0]
        y       = new_position[1]
        width   = new_position[2]
        height  = new_position[3]

        self._label.axes.set_position([x, y + height, 0.6*width, 0.3*height])
        self._ax_slider_a.set_position([x, y + 0.35*height, 0.8*width, 0.3*height])
        self._ax_slider_b.set_position([x, y, 0.8*width, 0.3*height])
        self._ax_button.set_position([x + 0.85*width, y, 0.15*width, height])
    
    def set_idx(self, new_idx: int) -> None:
        """
        Sets the internal index to new_idx and updates the label.
        """
        self._idx = new_idx
        if self._feature is not None:
            self._label.set_text(f'$\phi_{self._idx+1}(x) = {self._feature.get_expression()}$')
    
    def hide(self) -> None:
        """
        Hides and disables this FeatureController. Disconnects the callback function of slider
        and x_button.
        """
        self._label.axes.set_visible(False)
        hide_widget(self._slider_a, self._slider_id)
        hide_widget(self._slider_b, self._slider_id)
        hide_widget(self._x_button, self._button_id)

        self._feature = None
    
    def set_feature(self, feature: Feature) -> None:
        """
        Sets up label, slider and X-button to accomodate the given Feature.
        This also "activates" the FeatureController, turning it visible and connecting
        the callback functions regarding this feature.
        """
        self._feature = feature

        self._label.axes.set_visible(True)
        self._ax_slider_a.set_visible(True)
        self._ax_slider_b.set_visible(True)
        self._x_button.ax.set_visible(True)

        self._label.set_text(f'$\phi_{self._idx+1}(x) = {feature.get_expression()}$')

        # Setting a feature creates new Slider objects. This requires clearing the axes or else remainders
        # of the previous Sliders might be displayed.
        self._refresh_axes()

        min, max, step = feature.get_parameter_range('a')
        self._slider_a = Slider(self._ax_slider_a, '', valmin=min, valmax=max,
                                valinit=feature.parameter_a, valstep=step, handle_style={'size': 5})
        self._slider_a.valtext.set_visible(False)

        min, max, step = feature.get_parameter_range('b')
        self._slider_b = Slider(self._ax_slider_b, '', valmin=min, valmax=max,
                                valinit=feature.parameter_b, valstep=step, handle_style={'size': 5})
        self._slider_b.valtext.set_visible(False)

        self._slider_a_id = self._slider_a.on_changed(self._on_slider_a_changed)
        self._slider_b_id = self._slider_b.on_changed(self._on_slider_b_changed)
        self._slider_a.set_val(feature.parameter_a)
        self._slider_b.set_val(feature.parameter_b)
        
        self._button_id = self._x_button.on_clicked(lambda event: self._parent_controller.remove_feature(self._idx))
    
    def _refresh_axes(self) -> None:
        """
        Removes the Slider axes and replaces them with new axes at the same position. Call to clear all
        children to prevent leftover markers being displayed after creating new Sliders.
        """
        pos = self._ax_slider_a.get_position().get_points()
        pos = (pos[0, 0], pos[0, 1], pos[1, 0] - pos[0, 0], pos[1, 1] - pos[0, 1])
        self._ax_slider_a.remove()
        self._ax_slider_a = plt.axes(pos)

        pos = self._ax_slider_b.get_position().get_points()
        pos = (pos[0, 0], pos[0, 1], pos[1, 0] - pos[0, 0], pos[1, 1] - pos[0, 1])
        self._ax_slider_b.remove()
        self._ax_slider_b = plt.axes(pos)

    def _on_slider_a_changed(self, val: float) -> None:
        self._feature.parameter_a = val
        self._parent_controller.gauss.update_feature_parameter()
        self._label.set_text(f'$\phi_{self._idx+1}(x) = {self._feature.get_expression()}$')

    def _on_slider_b_changed(self, val: float) -> None:
        self._feature.parameter_b = val
        self._parent_controller.gauss.update_feature_parameter()
        self._label.set_text(f'$\phi_{self._idx+1}(x) = {self._feature.get_expression()}$')
    
class FeatureVectorController:
    """
    Controlls the feature vector of a given InteractiveGaussian.
    Draws buttons that let the user dynamically add new features, and a FeatureController instance
    for every active feature to a figure.
    """
    def __init__(self, fig: plt.figure, gaussian: InteractiveGaussian):
        # Reference to target figure and InteractiveGaussian object.
        self.fig = fig
        self.gauss = gaussian

        self._number_active_features = 0

        self._control_buttons: list[Button] = []
        self._create_control_buttons()

        self._feature_controllers: list[FeatureController] = []
        for i in range(MAX_NUMBER_FEATURES):
            self._feature_controllers.append(FeatureController(self, pos('controller', row=i), idx=i))

        # Show the features that are already present in the gaussians phi attribute.
        for feature in self.gauss.phi.features:
            self._feature_controllers[self._number_active_features].set_feature(feature)
            self._number_active_features += 1

    def _create_control_buttons(self) -> None:
        """
        Creates three buttons that add a polynomial, sine or cosine feature respectively when clicked.
        """
        create_button(pos           = pos('button', row=0),
                      label         = 'Add power feature',
                      on_clicked    = lambda event: self.add_feature(PolynomialFeature(0, 0)),
                      target_list   = self._control_buttons)
        
        create_button(pos           = pos('button', row=1),
                      label         = 'Add harmonic feature',
                      on_clicked    = lambda event: self.add_feature(HarmonicFeature(1, 0)),
                      target_list   = self._control_buttons)

        create_button(pos           = pos('button', row=2),
                      label         = 'Add Gauss feature',
                      on_clicked    = lambda event: self.add_feature(GaussFeature(0.2, 0)),
                      target_list   = self._control_buttons)
                
    def add_feature(self, feature: Feature) -> None:
        """
        Adds a feature to the gaussians phi attribute and displays the FeatureController to control it.
        """
        if self._number_active_features == MAX_NUMBER_FEATURES: return
 
        self._feature_controllers[self._number_active_features].set_feature(feature)
        self._number_active_features += 1

        self.gauss.add_feature(feature)

    def remove_feature(self, idx: int) -> None:
        """
        Removes the feature at index idx from the gaussians phi attribute and hides and disconnects
        the FeatureController associated with it.
        """
        # The InteractiveGaussian class requires at least two active features.
        if self._number_active_features == 2: return

        self._feature_controllers[idx].hide()

        # Move elements that were positioned below the removed index upwards to close the gap.
        # The element that has just been hidden can be moved to position self.number_active_features-1.
        self._feature_controllers[idx].set_idx(self._number_active_features-1)
        self._feature_controllers[idx].set_position(pos('controller', row=self._number_active_features-1))

        for i in range(idx+1, self._number_active_features):
            # Move elements one row upwards on the GUI.
            self._feature_controllers[i].set_position(pos('controller', row=i-1))
            self._feature_controllers[i].set_idx(i-1)

            # Swap elements in list.
            self._feature_controllers[i-1], self._feature_controllers[i] = self._feature_controllers[i], self._feature_controllers[i-1]

        self._number_active_features -= 1
        self.fig.canvas.draw()
        self.gauss.remove_feature(idx)
        