from typing import Optional, Callable

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle


def add_frame(
        area: tuple[float, float, float, float],
        offset: float=0.025
    ) -> None:
    """Add a frame around the given area.

    Parameters
    ----------
    area: `tuple[float, float, float, float]`
        Inner rectangle as [left, botton, width, height].
    offset: `float`
        The distance between `area` and the frame, relative to the
        width of the current figure.
    """
    # Get the fraction figure_width/figure_height to apply an offset of
    # FRAME_OFFSET*width in both directions.
    size = plt.gcf().get_size_inches()
    y_factor = size[0] / size[1]

    offset_area = (
        area[0] - offset / 2,
        area[1] - offset * y_factor / 2,
        area[2] + offset,
        area[3] + offset * y_factor
    )
    ax = plt.axes(offset_area)
    ax.axis('off')
    rect = Rectangle(
        xy=(0, 0),
        width=1,
        height=1,
        facecolor='none',
        edgecolor='0.8',
        lw=5
    )
    ax.add_patch(rect)


def create_button(pos: tuple[float, float, float, float],
                  label: str,
                  on_clicked: Optional[Callable] = None,
                  target_list: Optional[list[Button]] = None) -> Button:
    """
    Creates a button with the specified attributes and adds it to the target_list. Returns the new button.

    Parameters
    ----------
    pos: `tuple[float, float, float, float]`
        Button position relative to full figure in XYWH notation.
    label: `str`
        Button label as string.
    on_clicked: `Optional[Callable]`
        The function to call when button is clicked.
    target_list: `Optional[list[Button]]`
        A list this button is added to.
    """
    button_ax = plt.axes(pos)
    button = Button(button_ax, label=label)
    if on_clicked is not None:
        button.on_clicked(on_clicked)
    if target_list is not None:
        target_list.append(button)

    return button
