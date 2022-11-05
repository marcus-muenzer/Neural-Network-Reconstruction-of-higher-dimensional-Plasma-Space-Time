"""
NAME:
    visualization

DESCRIPTION:
    Provides different visualization functions for MHD data.
"""

import pandas as pd
import numpy as np
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation, PillowWriter
from utils.color_scale import get_color_scale
from utils.data import index_to_phys_unit, convert_tensor_to_numpy
from data.data_loading import get_index_for_value
import globals.constants as const


# Set default styles
params = {
    'legend.fontsize': 'xx-large',
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 'xx-large',
}
matplotlib.rcParams.update(params)


def color_plot_for_specific_time(x, y, data, index=0, t_val=0, annotate=None, filename="color_plot.pdf", title='default', cmap='hot', scaling_type='data', dpi=100):
    """
    Creates a 2D color plot at a specifig time.

    Parameters:
        x: tensor of x space of the data
        y: tensor of y space of the data
        data: MHD data for which to create the plot
              shape: torch.Size([const.dim_mhd_state, x, y, t])
        index: index for physical unit of the MHD data to create the plot for
        t_val: real t value of the data at which the data should be evaluated
        annotate: if set: color in which the training data point should be highlighted and pointed to
                  e.g. 'white', 'black'
        filename: path/filename at which the plot should be saved to. If empty, the plot will not be saved.
                  should end with ".pdf"
        title: title of the figure
               'default' -> sets a default title
        cmap: matplotlib.colors.Colormap
        scaling_type: method to determine minimal and maximal values for mapping values to colors
                      options: 'predefined' -> use predefined default values
                               'data' -> use minimal and maximal values of the data
                               tupel (vmin, vmax) -> use given values
        dpi: dpi for saved figure

    Returns:
        None
    """

    # Convert x, y, and data to numpy
    x = convert_tensor_to_numpy(x)
    y = convert_tensor_to_numpy(y)
    data = convert_tensor_to_numpy(data)

    # Extract plot data
    t_index = get_index_for_value(const.t_red, t_val)
    data = data[index, :, :, t_index]

    # Create figure and axes
    fig, ax = plt.subplots()

    # Create color mapping
    cmap = plt.cm.get_cmap(cmap)

    # Get colorbar range
    vmin, vmax = get_color_scale(data, scaling_type)

    # Create color plot
    img = ax.pcolormesh(x, y, data.T, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Set title
    if title == 'default':
        phys_unit = index_to_phys_unit(index)
        title = phys_unit + " in Space-Time at t = " + str(t_val)
    ax.set_title(title)

    # Show ticks
    ax.tick_params(left=True, bottom=True)

    # Add colorbar
    fig.colorbar(img)

    if annotate:
        t_ind = get_index_for_value(const.st_train[:, 2], t_val)
        plt.annotate("training data\n      (point)", xy=const.st_train[t_ind][:2], xytext=(0, -6),
                     color='white', font={'weight': 'bold', 'size': 14},
                     arrowprops=dict(color=annotate, arrowstyle="->", linewidth=2))

    # Fit labels to design
    plt.tight_layout()

    # Save figure to file
    if filename and filename != "":
        fig.savefig(filename, dpi=dpi)


def color_plot_animation(x, y, t, data, index=0, filename="color_plot_animation.gif", title='default', cmap='hot', scaling_type='data', dpi=100):
    """
    Creates a 2D animation of color plots over time.

    Parameters:
        x: tensor of x space of the data
        y: tensor of y space of the data
        t: tensor of t space of the data
        data: MHD data for which to create the plot
              shape: torch.Size([const.dims_mhd_state, x, y, t])
        index: index for physical unit of the MHD data to create the plot for
        filename: path/filename at which the plot should be saved to. If empty, the plot will not be saved.
                  must end with ".gif"
        title: title of the figure
               'default' -> sets a default title
        cmap: matplotlib.colors.Colormap
        scaling_type: determine minimal and maximal values for mapping values to colors
                      options: 'predefined' -> use predefined default values
                               'data' -> use minimal and maximal values of the data
                               tupel (vmin, vmax) -> use given values
        dpi: dpi for saved figure

    Returns:
        None
    """

    # Convert x, y, and data to numpy
    x = convert_tensor_to_numpy(x)
    y = convert_tensor_to_numpy(y)
    t = convert_tensor_to_numpy(t)
    data = convert_tensor_to_numpy(data)

    # Extract plot data
    data = data[index]

    # Create figure and axis
    fig, ax = plt.subplots()

    # Create color mapping
    cmap = plt.cm.get_cmap(cmap)

    # Get colorbar range
    vmin, vmax = get_color_scale(data, scaling_type)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Physical unit for title
    phys_unit = index_to_phys_unit(index)

    # Show ticks
    ax.tick_params(left=True, bottom=True)

    # Create initial color plot
    pcm = ax.pcolormesh(x, y, data[:, :, 0].T, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    # Add color legend
    fig.colorbar(pcm)

    plt.tight_layout()

    def animate(t_ind):
        """Helper function. Needed for FuncAnimation"""

        # Set and update title
        nonlocal title
        ax.set_title(title)
        if title == 'default':
            title_t = phys_unit + " in Space-Time at t = " + str(round(t[t_ind], 2))
            ax.set_title(title_t)
        plt.tight_layout()

        # Update color plot
        pcm.set_array(data[:, :, t_ind].T.flatten())

    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(t) - 1)

    # Run and save animation
    if filename and filename != "":
        anim.save(filename, dpi=dpi, writer=PillowWriter(fps=6))


def scatter_plot(st, U, index=0, filename="scatter_plot.pdf", title='default', cmap='hot', scaling_type='data', dpi=100):
    """
    Creates a scatter plot of trajectories in 3D spacetime.

    Parameters:
        st: tensor of spacetimes (used as coordinates for 3D plot)
            shape: torch.Size([n points, const.dims_st])
        U: tensor of MHD states (used as values for points in 3D plot)
           shape: torch.Size([n points, const.dims_mhd_state])
        index: index for physical unit of the MHD data to create the plot for
        filename: path/filename at which the plot should be saved to. If empty, the plot will not be saved.
                  should end with ".pdf"
        title: title of the figure
               'default' -> sets a default title
        cmap: matplotlib.colors.Colormap
        scaling_type: determine minimal and maximal values for mapping values to colors
                      options: 'predefined' -> use predefined default values
                               'data' -> use minimal and maximal values of the data
                               tupel (vmin, vmax) -> use given values
        dpi: dpi for saved figure

    Returns:
        None
    """

    # Grid background
    seaborn.set_style('whitegrid')

    # Convert st, U to numpy
    st = convert_tensor_to_numpy(st)
    U = convert_tensor_to_numpy(U)

    # Get colorbar range
    vmin, vmax = get_color_scale(U[:, index], scaling_type)

    # Create color mapping
    cmap = plt.cm.get_cmap(cmap)
    norm = plt.Normalize(vmin, vmax)

    # Discretize colors
    colors = []
    for i in range(st.shape[0]):
        colors.append(cmap(norm(U[i][index])))

    # Create figure
    fig = plt.figure()

    # Add subplot
    ax = fig.add_subplot(111, projection='3d')

    # Scatter spacetime trajectories
    # Using MHD state values for color encoding
    ax.scatter(st[:, 2], st[:, 0], st[:, 1], c=colors, marker='.')

    # Plot trajectory in whole spacetime
    ax.set_xlim(const.t_min, const.t_max)
    ax.set_ylim(const.x_min, const.x_max)
    ax.set_zlim(const.y_min, const.y_max)

    # Label plot
    ax.set_xlabel('T')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    # Set title
    if title == 'default':
        phys_unit = index_to_phys_unit(index)
        title = phys_unit + " trajectories in Space-Time"
    ax.set_title(title)

    # Add color bar
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=.75, orientation="vertical", pad=.15)

    # Save figure to file
    if filename and filename != "":
        fig.savefig(filename, dpi=dpi)


def binned_heatmap(matrix, nth_tick=3, distribution_plots=True, cmap='viridis', backgroundcolor='#e0e0e0'):
    """
    Creates a heatmap on bin basis.

    Paramters:
        matrix: matrix that contains the bins
                must have quadratic size
                pd.DataFrame
        nth_tick: defines the tick frequency
        distribution_plots: determines if distribution plots should be added outside of the heatmap
        cmap: matplotlib.colors.Colormap
        backgroundcolor: background color of axes

    Returns:
        fig: binned heatmap
             matplotlib.figure.Figure
    """

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6.5, 4.5), sharey=True, sharex=True)

    # Set labels
    ax.set_ylabel("Prediction")
    ax.set_xlabel("Ground truth")

    # Set ticks and ticklabels
    ax.set_xticks(range(matrix.shape[0])[::nth_tick])
    ax.set_yticks(range(matrix.shape[1])[::nth_tick])
    ax.set_xticklabels(matrix.columns.values[::nth_tick], rotation=90)
    ax.set_yticklabels(matrix.index.values[::nth_tick])

    # Show ticks
    ax.tick_params(left=True, bottom=True)

    # Create color mapping
    cmap = plt.cm.get_cmap(cmap).copy()
    # Set background color for 0 counts
    cmap.set_bad(backgroundcolor)

    # Create heatmap
    im = ax.imshow(matrix, norm=colors.LogNorm(), cmap=cmap)
    ax.grid(False)

    # Add diagonal line
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    ax.plot([min_x, min_y], [max_x, max_y], ':k')

    # Add distribution plots
    if distribution_plots:
        fig.subplots_adjust(wspace=0.3)

        # Create plots as sub axes
        xdist = ax.inset_axes([0, 1.07, 1, 0.15])
        ydist = ax.inset_axes([1.07, 0, 0.15, 1])
        xdist.bar(range(matrix.shape[0]), matrix.sum(axis=0).values)
        ydist.barh(range(matrix.shape[1]), matrix.sum(axis=1).values[::-1])

        # Set backgroundcolor for distribution plots
        xdist.set_facecolor(backgroundcolor)
        ydist.set_facecolor(backgroundcolor)

        # Set ticks and ticklabels for distribution plots
        xdist.set_xticks(range(matrix.shape[0])[::nth_tick])
        ydist.set_yticks(range(matrix.shape[1])[::-nth_tick])
        xdist.set_xticklabels([])
        ydist.set_yticklabels([])

        # Show ticks
        # Rotate ticklabes at right subplot
        xdist.tick_params(left=True)
        ydist.tick_params(bottom=True, rotation=90)

        # Set log scale
        xdist.set_yscale("log")
        ydist.set_xscale("log")

        # No margins inside the plot
        xdist.margins(0)
        ydist.margins(0)

    # Create colorbar
    cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.81])

    # Set colorbar title
    cbar_ax.set_title("Count")

    # Add colorbar to figure
    fig.colorbar(im, cax=cbar_ax)

    return fig


def create_bin_matrix(x, y, n_bins=40):
    """
    Bins data.

    Parameters:
        x: data for which the bins on the x axis are created
        y: data for which the bins on the y axis are created
        n_bins: number of bins

    Returns:
        bin_matrix: matrix containing the bins
                    pd.DataFrame
    """

    # Bin values into discrete intervals
    bin_x_per_sample, bins_x = pd.cut(x, bins=n_bins, retbins=True)
    bin_y_per_sample, bins_y = pd.cut(y, bins=n_bins, retbins=True)

    # Transform into pd.Dataframes
    bin_x_per_sample = pd.DataFrame(bin_x_per_sample)
    bin_y_per_sample = pd.DataFrame(bin_y_per_sample)
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    # Concatenate Dataframes and set axis
    bins = pd.concat([bin_x_per_sample, bin_y_per_sample, x, y], axis=1)
    bins = bins.set_axis(["bins_x", "bins_y", "a", "b"], axis=1)

    # Round predicted and real values to three decimal places
    bins[["bins_x", "bins_y"]] = bins.apply({"bins_x": lambda x: np.round(x.right, 3),
                                             "bins_y": lambda x: np.round(x.right, 3)})

    # Count occurences
    matrix = bins.groupby(["bins_x", "bins_y"]).agg("count").reset_index()

    # Create bin matrix
    bin_matrix = matrix.pivot(index=matrix.columns[1],
                              columns=matrix.columns[0],
                              values=matrix.columns[2]).sort_index(ascending=False)

    return bin_matrix


def create_binned_heatmap_from_original_data(observation, prediction, index=0, n_bins=40, nth_tick=3, distribution_plots=True, filename="binned_heatmap.pdf", title='default', cmap='viridis', dpi=100):
    """
    Generates a binned heatmap visualization of the predictions vs. labels.

    Parameters:
        observation: tensor of ground truth MHD data for which to create the plot
                     numpy.ndarray
                     shape: [n_points, const.dims_mhd_state]
        prediction: tensor of predicted of MHD data for which to create the plot
                    numpy.ndarray
                    shape: [n_points, const.dims_mhd_state]
        index: index for physical unit of the MHD data to create the plot for
        n_bins: number of bins
        nth_tick: defines the tick frequency of heatmap labels
        distribution_plots: determines if distribution plots should be added outside of the heatmap
        filename: path/filename at which the plot should be saved to. If empty, the plot will not be saved.
                  should end with ".pdf"
        title: title of the figure
               'default' -> sets a default title
        cmap: matplotlib.colors.Colormap
        dpi: dpi for saved figure

    Returns:
        None
    """

    # Convert observation, prediction to numpy
    observation = convert_tensor_to_numpy(observation)
    prediction = convert_tensor_to_numpy(prediction)

    # Extract plot data
    observation = observation[:, index]
    prediction = prediction[:, index]

    # Transform data into bins
    bin_matrix = create_bin_matrix(observation, prediction, n_bins)

    # Reset background to default
    seaborn.set_style('white')

    # Generate heatmap
    fig = binned_heatmap(bin_matrix, nth_tick, distribution_plots, cmap)

    # Add physical unit label on the left
    phys_unit = index_to_phys_unit(index)
    fig.text(0.5, 1.1, phys_unit, ha='center', va='center',
             fontsize="x-large", fontweight='bold')

    # Save figure to file
    if filename and filename != "":
        fig.savefig(filename, bbox_inches='tight', dpi=dpi)


def kernel_density_plot(observation, prediction, index=0, bw_method=.5, filename="kernel_density_plot.pdf", title='default', color='darkgoldenrod', dpi=100):
    """
    Creates a residual kernel density estimation plot.
    Residuals are automatically calculated from the data.

    Parameters:
        observation: tensor of ground truth MHD data for which to create the plot
                     numpy.ndarray
                     shape: [n_points, const.dims_mhd_state]
        prediction: tensor of predicted of MHD data for which to create the plot
                    numpy.ndarray
                    shape: [n_points, const.dims_mhd_state]
        index: index for physical unit of the MHD data to create the plot for
        bw_method: method used to calculate the estimator bandwidth
        filename: path/filename at which the plot should be saved to. If empty, the plot will not be saved.
                  should end with ".pdf"
        title: title of the figure
               'default' -> sets a default title
        color: color of the plot
        dpi: dpi for saved figure

    Returns:
        None
    """

    # Grid background
    seaborn.set_style('darkgrid')

    # Convert observation, prediction to numpy
    observation = convert_tensor_to_numpy(observation)
    prediction = convert_tensor_to_numpy(prediction)

    # Calculate Residuals
    residuals = observation[:, index] - prediction[:, index]

    # Calculate mean and standard deviation metrics
    mean = np.mean(residuals)
    std = np.std(residuals)

    # Create figure and axes
    fig, ax = plt.subplots(tight_layout=True)

    # Create KDE plot
    seaborn.kdeplot(residuals, ax=ax, color=color, shade=True, bw_method=bw_method)

    # Add mean and standard deviation
    ax.text(.95, .93, 'Mean: ' + str(round(mean, 2)) + '''\n''' + 'Std: ' + str(round(std, 2)),
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes, fontsize=13,
            bbox={'facecolor': 'grey', 'alpha': .5, 'pad': 5})

    # Add vertical dotted line in the middle
    _, max_y = ax.get_ylim()
    ax.plot([0, 0], [0, max_y], ':k')

    # Show ticks
    ax.tick_params(left=True, bottom=True)

    # Set title
    if title == 'default':
        phys_unit = index_to_phys_unit(index)
        title = "Residuals " + phys_unit
    ax.set_title(title)

    # Remove y label that gets automatically set by seaborn.kdeplot
    ax.set_ylabel("")

    # Create black border
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')

    # Save figure to file
    if filename and filename != "":
        fig.savefig(filename, dpi=dpi)


def line_plot(x_data, observation, prediction, index=0, x_val=0, y_val='all', t_val=0, filename="line_plot.pdf", title='default', dpi=100):
    """
    Creates a line plot of observated and predicted values.

    Parameters:
        x_data: data along x axis
        observation: tensor of ground truth MHD data for which to create the plot
                     numpy.ndarray
                     shape: const.U_red.shape
        prediction: tensor of predicted of MHD data for which to create the plot
                    numpy.ndarray
                     shape: const.U_red.shape
        index: index for physical unit of the MHD data to create the plot for
        x_val: real x value of the data at which the data should be evaluated
               if 'all': data will include all values along the x dimension
                         y_val, t_val must be scalars in this case
        y_val: real y value of the data at which the data should be evaluated
               if 'all': data will include all values along the y dimension
                         x_val, t_val must be scalars in this case
        t_val: real t value of the data at which the data should be evaluated
               if 'all': data will include all values along the t dimension
                         x_val, y_val must be scalars in this case
        filename: path/filename at which the plot should be saved to. If empty, the plot will not be saved.
                  should end with ".pdf"
        title: title of the figure
               'default' -> sets a default title
        dpi: dpi for saved figure

    Returns:
        None
    """

    # Convert observation, prediction to numpy
    observation = convert_tensor_to_numpy(observation)
    prediction = convert_tensor_to_numpy(prediction)

    # Extract data
    # Check for bad configuration
    # Prepare label and title
    error = False
    if x_val == 'all':
        # Check for bad configuration
        if y_val == 'all' or t_val == 'all':
            error = True

        # Calculate indices and extract data
        y_index = get_index_for_value(const.y_red, y_val)
        t_index = get_index_for_value(const.t_red, t_val)
        observation = observation[index, :, y_index, t_index]
        prediction = prediction[index, :, y_index, t_index]

        # Prepare label and title
        x_label = 'X'
        title_suffix = 'y = ' + str(y_val) + ', t = ' + str(t_val)

    elif y_val == 'all':
        # Check for bad configuration
        if x_val == 'all' or t_val == 'all':
            error = True

        # Calculate indices and extract data
        x_index = get_index_for_value(const.x_red, x_val)
        t_index = get_index_for_value(const.t_red, t_val)
        observation = observation[index, x_index, :, t_index]
        prediction = prediction[index, x_index, :, t_index]

        # Prepare label and title
        x_label = 'Y'
        title_suffix = 'x = ' + str(x_val) + ', t = ' + str(t_val)

    elif t_val == 'all':
        # Check for bad configuration
        if x_val == 'all' or y_val == 'all':
            error = True

        # Calculate indices and extract data
        x_index = get_index_for_value(const.x_red, x_val)
        y_index = get_index_for_value(const.y_red, y_val)
        observation = observation[index, x_index, y_index, :]
        prediction = prediction[index, x_index, y_index, :]

        # Prepare label and title
        x_label = 'T'
        title_suffix = 'x = ' + str(x_val) + ', y = ' + str(y_val)

    # Error handling
    if error:
        err_msd = "bad configuration: either x_val, y_val, or t_val must have value 'all'. Only one parameter can have value'all'!"
        raise ValueError(err_msd)

    # Create figure and axes
    fig, ax = plt.subplots(tight_layout=True)

    # Integrate lines
    ax.plot(x_data, observation, color='k', label="exact")
    ax.plot(x_data, prediction, color='r', label='prediction')

    # Add legend
    ax.legend(loc='best', prop={'size': 12})

    # Set x-label
    ax.set_xlabel(x_label)

    # Set title
    if title == 'default':
        phys_unit = index_to_phys_unit(index)
        title = phys_unit + ' at ' + title_suffix
    ax.set_title(title)

    # Save figure to file
    if filename and filename != "":
        fig.savefig(filename, dpi=dpi)


def bar_plot(data, xticks=None, bottom=0, colors=None, ylabel='10$^3$', total_width=0.8, single_width=1, legend=True, legend_size=12, filename="bar_plot.pdf", title='Metrics', dpi=100):
    """
    Draws a bar plot with multiple bars per data point.

    Parameters:
        data: dictionary containing the data to plot
              keys are the names of the data, the items is a list of the values

              Example:
                  data = {
                      "x":[1,2,3],
                      "y":[1,2,3],
                      "z":[1,2,3],
                  }
        xticks: list of xtick labels
                if None: xticks get enumerated
        colors : list of colors which are used for the bars
                 if None: the colors will be the standard matplotlib color cyle
        ylabel: string of the label for the y axis
        total_width : the width of a bar group
                      Example:
                          0.8 means that 80% of the x-axis is covered
                          by bars and 20% will be spaces between the bars
        single_width: relative width of a single bar within a group
                      1 means the bars will touch eachother within a group
                      values less than 1 will make the bars thinner
        legend: if this is set to True, a legend will be added to the axis
        legend_size: size of the legend
        filename: path/filename at which the plot should be saved to. If empty, the plot will not be saved.
                  should end with ".pdf"
        dpi: dpi for saved figure

    Returns:
        None
    """

    # Create figure and axis
    fig, ax = plt.subplots()

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, bottom=bottom, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), prop={'size': legend_size})

    # Set x ticks
    if xticks:
        plt.xticks(range(len(xticks)), xticks)

    # Set y label
    ax.set_ylabel(ylabel)

    # Set title
    ax.set_title(title)

    # Save figure to file
    if filename and filename != "":
        plt.savefig(filename, dpi=dpi)
