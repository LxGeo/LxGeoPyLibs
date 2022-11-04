import numpy as np
from matplotlib import pyplot as plt

def plot_quiver(flow, spacing, margin=0, **kwargs):
    """Plots less dense quiver field.
    Args:
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    Returns:
        fig: plt figure
    """
    fig, ax = plt.subplots()
    ax.set_axis_off()
    h, w, *_ = flow.shape
    nx = int((w - 2 * margin) / spacing)
    ny = int((h - 2 * margin) / spacing)
    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)
    flow = flow[np.ix_(y, x)]
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
    ax.quiver(x, y, u, v, **kwargs)
    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")
    return fig