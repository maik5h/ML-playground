from typing import Optional, Callable

import matplotlib.pyplot as plt
from matplotlib.widgets import Button


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
