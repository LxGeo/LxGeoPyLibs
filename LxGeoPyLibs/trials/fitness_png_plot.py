
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

ticks_label = np.arange(-50, 50, step=10)

def plot_in_ax(ax, fp):
    """"""
    img = plt.imread(fp)/255
    ticks = np.arange(0,img.shape[0],20)
    im = ax.imshow(img, vmin=0, vmax=1, cmap="Greens")
    cbar = plt.colorbar(im)
    cbar.set_label('Fitness value')
    ax.set_xticks(ticks);
    ax.set_yticks(ticks);
    ax.set_xticklabels(ticks_label)
    ax.set_yticklabels(ticks_label)

    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='y', which='both', length=4, direction='inout')

    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='x', which='both', length=4, direction='inout')
    
    
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')

    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    ax.set_title('(b) Dense map of geometry fitness within the search space of 50 meters from initial position', y=-0.1)

def plot_in_ax_centered(ax, fp):

    img = plt.imread(fp)/255
    im = ax.imshow(img, vmin=0, vmax=1, cmap="Greens")
    height, width = img.shape[:2]
    grid_spacing = 50  # Specify the spacing between grid lines
    x_ticks = np.arange(-width/2 + grid_spacing/2, width/2, grid_spacing)
    y_ticks = np.arange(-height/2 + grid_spacing/2, height/2, grid_spacing)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(length=0, labelcolor='white')
    ax.grid(True, color='white', linewidth=0.5)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')



if __name__ == "__main__":
    fp1 = "C:/Users/geoimage/Documents/fitness_fn/al_0.200000_be_0.000000_ga_0.000000_te_0.000000/1500.jpg"
    fp2 = "C:/Users/geoimage/Documents/fitness_fn/al_0.200000_be_0.200000_ga_0.200000_te_0.000000/1500.jpg"

    fig, axes = plt.subplots(1,1)

    #plot_in_ax(axes[0], fp1)
    plot_in_ax(axes, fp2)
    
    plt.show()