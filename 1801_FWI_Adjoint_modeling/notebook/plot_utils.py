import matplotlib.pyplot as plt


def add_subplot_axes(ax, rect, fc='w'):
    """
    Facilitates the addition of a small subplot within another plot.
    From: http://stackoverflow.com/users/2309442/pablo
    License: CC-BY-SA
    Args:
        ax (axis): A matplotlib axis.
        rect (list): A rect specifying [left pos, bottom pos, with, height]
    Returns:
        axis: The sub-axis in the specified position.
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], facecolor=fc)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax
