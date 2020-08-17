import sys
import os
import json
from typing import List
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from cityscapesscripts.helpers.labels import name2label


def csToMplColor(class_name):
    color = name2label[class_name].color
    return [x/255. for x in color]

def create_table_row(
        axis: Axes,
        x_pos: float,
        y_pos: float,
        data_dict: dict,
        title: str,
        key: str,
        subdict_key: str = None
    ):
    """Creates a row presenting scores for all classes in category ``key`` in ``data_dict``.

    Args:
        axis (Axes): Axes-instances to use for the subplot
        x_pos (float): x-value for the left of the row relative in the given subplot
        y_pos (float): y-value for the top of the row relative in the given subplot
        data_dict (dict): dict conatining data to visualise
        title (str): title of the row / category-name
        key (str): key in ``data_dict`` to obtain data to visualise
        subdict_key (str or None):
            additional key to access data, if ``data_dict[key]`` returns again a dict,
            otherwise None
    """

    axis.text(x_pos, y_pos, title, fontdict={'weight': 'bold'})
    y_pos -= 0.1
    delta_x_pos = 0.17

    for (cat, valdict) in data_dict[key].items():
        val = valdict if subdict_key is None else valdict[subdict_key]
        axis.text(x_pos, y_pos, cat)
        axis.text(x_pos+delta_x_pos, y_pos, "%.2f" % val)
        y_pos -= 0.1

    # add Mean
    y_pos -= 0.05
    axis.text(x_pos, y_pos, "Mean", fontdict={'weight': 'bold'})
    axis.text(x_pos+delta_x_pos, y_pos, '%.2f' %
              data_dict["m"+key], fontdict={'weight': 'bold'})


def create_result_table_and_legend_plot(axis: Axes, data_to_plot: dict, handles_labels: tuple):
    """Creates the plot-section containing a table with result scores and labels.

    Args:
        axis (Axes): Axes-instances to use for the subplot
        data_to_plot (dict): Dictionary containing ``"Detection_Score"`` and ``"AP"``
            and corresponding mean values
        handles_labels (tuple[List, List]): Tuple of matplotlib handles and corresponding labels
    """

    required_keys = ["Detection_Score", "mDetection_Score", "AP", "mAP"]
    assert all([key in data_to_plot.keys() for key in required_keys])

    # Results
    axis.axis("off")
    axis.text(0, 0.95, 'Results', fontdict={'weight': 'bold', 'size': 16})

    y_pos_row = 0.75
    # 2D AP results
    create_table_row(axis, 0.00, y_pos_row, data_to_plot,
                     title="2D AP", key="AP", subdict_key="auc")

    # Detection score results
    create_table_row(axis, 0.28, y_pos_row, data_to_plot,
                     title="Detection Score", key="Detection_Score", subdict_key=None)

    # Legend
    x_pos_legend = 0.6
    y_pos_legend = 0.75
    y_pos_dot_size = 0.0
    axis.text(x_pos_legend, 0.95, 'Legend',
              fontdict={'weight': 'bold', 'size': 16})
    axis.legend(*handles_labels, frameon=True,
                loc="upper left", bbox_to_anchor=(x_pos_legend, y_pos_legend), ncol=2)

    # add data-point-marker size explanation
    dot_size_explanation = "The size of each data-point-marker indicates\n"
    dot_size_explanation += "the relative amount of samples for that data-\n"
    dot_size_explanation += "point, with large dots indicate larger samples."
    axis.text(x_pos_legend, y_pos_dot_size, dot_size_explanation)


def create_spider_chart_plot(
        axis: Axes,
        data_to_plot: dict,
        categories: List[str],
        accept_classes: List[str],
    ):
    """Creates spider-chart with ``categories`` for all classes in ``accept_classes``.

    Args:
        axis (Axes): Axes-instances to use for the spider-chart
        data_to_plot (dict): Dictionary containing ``categories`` as keys.
        categories (list of str): List of category-names to use for the spider-chart.
        accept_classes (list of str): List of class-names to use for the spider-chart.
    """

    # create labels
    lables = [category.replace("_", "-") for category in categories]

    # Calculate metrics for each class
    vals = {
        cat: [cat_vals["auc"]
              for x, cat_vals in data_to_plot[cat].items() if x in accept_classes]
        for cat in categories
    }

    # norm everything to AP
    for key in ["Center_Dist", "Size_Similarity", "OS_Yaw", "OS_Pitch_Roll"]:
        vals[key] = [v * float(ap) for (v, ap) in zip(vals[key], vals["AP"])]

    # setup axis
    num_categories = len(categories)

    angles = [n / float(num_categories) * 2 *
              np.pi for n in range(num_categories)]
    angles += angles[:1]

    axis.set_theta_offset(np.pi / 2.)
    axis.set_theta_direction(-1)
    axis.set_rlabel_position(0)
    axis.set_yticks([0.25, 0.50, 0.75])
    axis.set_yticklabels(["0.25", "0.50", "0.75"], color="grey", size=7)
    axis.tick_params(axis="x", direction="out", pad=10)
    axis.set_ylim([0, 1])
    axis.set_xticks(np.arange(0, 2.0*np.pi, np.pi/2.5))
    axis.set_xticklabels(lables)

    for idx, class_name in enumerate(accept_classes):
        values = [x[idx] for x in [vals[cat] for cat in categories]]
        values += values[:1]

        axis.plot(angles, values, linewidth=1,
                  linestyle='solid', color=csToMplColor(class_name))
        axis.fill(
            angles, values, color=csToMplColor(class_name), alpha=0.05)

    axis.plot(angles, [np.mean(x) for x in [vals[cat] for cat in categories] + [
        vals["AP"]]], linewidth=1, linestyle='solid', color="r", label="Mean")
    axis.legend(bbox_to_anchor=(0, 0))


def create_AP_plot(axis: Axes, data_to_plot: dict, accept_classes: List[str], max_depth: int):
    """Create the average precision (AP) subplot for classes in ``accept_classes``.

    Args:
        axis (Axes): Axes-instances to use for AP-plot
        data_to_plot (dict): Dictionary containing data to be visualized
            for all classes in ``accept_classes``
        accept_classes (list of str): List of class-names to use for the spider-chart
        max_depth (int): maximal encountered depth value
    """

    if not "AP_per_depth" in data_to_plot:
        raise ValueError()

    axis.set_title("AP per depth")
    axis.set_ylim([0, 1.01])
    axis.set_ylabel("AP")

    for class_name in accept_classes:
        aps = data_to_plot["AP_per_depth"][class_name]

        x_vals = [float(x) for x in list(aps.keys())]
        y_vals = [float(x["auc"]) for x in list(aps.values())]

        fill_standard_subplot(axis, x_vals, y_vals, class_name, [], max_depth)


def set_up_xaxis(axis: Axes, max_depth: int, num_ticks: int):
    """Sets up the x-Axis of given Axes-instance ``axis``.

    Args:
        axis (Axes): Axes-instances to use
        max_depth (int): max value of the x-axis is set to ``max_depth+1``
        num_ticks (int): number of ticks on the x-axis
    """
    axis.set_xlim([0, max_depth])
    axis.set_xticks(np.linspace(0, max_depth, num_ticks + 1))
    axis.set_xticklabels(["%.1f" % x for x in np.linspace(0, max_depth, num_ticks + 1)])


def set_up_PR_plot_axis(axis: Axes, min_iou: float, matching_method: str):
    """Sets up the axis for the precision plot."""
    axis.set_title("PR Curve@%.2f (%s)" % (min_iou, matching_method))
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim([0, 1.0])
    axis.set_ylim([0, 1.01])
    axis.set_xticks(np.arange(0, 1.01, 0.1))
    axis.set_xticklabels([x / 10. for x in range(11)])


def create_all_axes(max_depth: int, num_ticks: int):
    """Creates all Axes-instances of the 8 subplots.

    Arsg:
        max_depth (int): max value of the x-axis is set to ``max_depth+1``
        num_ticks (int): number of ticks on the x-axis

    Returns:
        ax_results (Axes): Axes-instance of the subplot
            containing the results-table and plot-legend
        ax_spider (Axes): Axes-instance of the subplot
            containing the spider_chart of AP-values for
        axes (List[Axes]): 6 Axes-instances for the categories.
    """

    ax_results = plt.subplot2grid((4, 2), (0, 0))
    ax_spider = plt.subplot2grid((4, 2), (0, 1), polar=True)
    ax1 = plt.subplot2grid((4, 2), (1, 0))
    ax2 = plt.subplot2grid((4, 2), (1, 1))
    ax3 = plt.subplot2grid((4, 2), (2, 0), sharex=ax2)
    ax4 = plt.subplot2grid((4, 2), (2, 1), sharex=ax2)
    ax5 = plt.subplot2grid((4, 2), (3, 0), sharex=ax2)
    ax6 = plt.subplot2grid((4, 2), (3, 1), sharex=ax2)
    axes = (ax1, ax2, ax3, ax4, ax5, ax6)

    # set up x-axes for ax2-ax6
    set_up_xaxis(ax2, max_depth, num_ticks)
    ax5.set_xlabel("Depth [m]")
    ax6.set_xlabel("Depth [m]")

    return ax_results, ax_spider, axes


def create_PR_plot(axis: Axes, data: dict, accept_classes: List[str]):
    """Fills precision-recall (PR) subplot with data and finalises ``axis``-set-up.

    Args:
        axis (Axes): Axes-instance of the subplot
        data (dict): data-dictionnary containing precision and recall values
            for all classes in ``accept_classes``
        accept_classes (list of str):
    """
    set_up_PR_plot_axis(
        axis, 
        data["eval_params"]["min_iou_to_match"], 
        data["eval_params"]["matching_method"]
    )

    for class_name in accept_classes:
        recalls_ = data['AP'][class_name]["data"]["recall"]
        precisions_ = data['AP'][class_name]["data"]["precision"]

        # sort the data ascending
        sorted_pairs = sorted(
            zip(recalls_, precisions_), key=lambda pair: pair[0])
        recalls, precisions = map(list, zip(*sorted_pairs))
        recalls = [0.] + recalls
        precisions = [0.] + precisions

        recalls += recalls[-1:] + [1.]
        precisions += [0., 0.]

        # precision values should be decreasing only
        # p(r) = max{r' > r} p(r')
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])

        axis.plot(recalls, precisions, label=class_name,
                  color=csToMplColor(class_name))


def fill_and_finalize_subplot(
        category: str,
        data_to_plot: dict,
        accept_classes: List[str],
        axis: Axes,
        max_depth: int,
    ):
    """Plot data to subplots by selecting correct data for given ``category`` and looping over
    all classes in ``accept_classes``.

    Args:
        category (str): scorce category, one of
            ["PR", "AP", "Center_Dist", "Size_Similarity", "OS_Yaw", "OS_Pitch_Roll"]
        data_to_plot (dict): Dictionary containing data to be visualised.
        accept_classes (list of str): List of class-names to use for the spider-chart.
        axis (Axes): Axes-instances to use for the subplot
        max_depth (int): maximal encountered depth value
    """

    if category == 'PR':
        create_PR_plot(axis, data_to_plot, accept_classes)

    elif category == 'AP':
        create_AP_plot(axis, data_to_plot, accept_classes, max_depth)

    elif category in ["Center_Dist", "Size_Similarity", "OS_Yaw", "OS_Pitch_Roll"]:

        axis.set_title(category.replace("_", " ") + " (DDTP Metric)")

        if category == 'Center_Dist':
            axis.set_ylim([0, 25])
            axis.set_ylabel("Distance [m]")
        else:
            axis.set_ylim([0., 1.01])
            axis.set_ylabel("Similarity")

        for class_name in accept_classes:
            x_vals, y_vals = get_x_y_vals(
                data_to_plot[category][class_name]["data"])
            available_items_scaling = get_available_items_scaling(
                data_to_plot[category][class_name]["items"])

            if category == 'Center_Dist':
                y_vals = [(1 - y) * max_depth for y in y_vals]

            fill_standard_subplot(
                axis, x_vals, y_vals, class_name, available_items_scaling, max_depth)

    else:
        raise ValueError("Unsupported category, got %s." % category)


def fill_standard_subplot(
        axis: Axes,
        x_vals: List[float],
        y_vals: List[float],
        class_name: str,
        available_items_scaling: List[float],
        max_depth: int,
    ):
    """Fills standard-subplots with data for ``class_name`` with data.

    Includes scatter-plot with size-scaled data-points, line-plot and
    a dashed line from maximal value in ``x_vals`` to ``max_depth``.

    Args:
        axis (Axes): Axes-instances to use for the subplot
        x_vals (list of float): x-values to visualize
        y_vals (list of float): y-values to visualize
        class_name (str): name of class to visualize data for
        available_items_scaling (list of float): size of data-points
        max_depth (int): maximal value of x-axis
    """
    if len(available_items_scaling) > 0:
        axis.scatter(x_vals, y_vals, s=available_items_scaling,
                     color=csToMplColor(class_name), marker="o", alpha=1.0)
    axis.plot(x_vals, y_vals, label=class_name,
                color=csToMplColor(class_name))

    if len(x_vals) >= 1:
        axis.plot([x_vals[-1], max_depth], [y_vals[-1], y_vals[-1]], label=class_name,
                  color=csToMplColor(class_name), linestyle="--", alpha=0.6)
        axis.plot([0, x_vals[0]], [y_vals[0], y_vals[0]], label=class_name,
                  color=csToMplColor(class_name), linestyle="--", alpha=0.6)


def get_available_items_scaling(data: dict, scale_fac: float = 100.):
    """Counts available items per data-point. Normalises and scales according to ``scale_fac``."""
    available_items = list(data.values())
    if len(available_items) == 0:
        return available_items

    max_num_item = max(available_items)
    available_items_scaling = [
        x / float(max_num_item) * scale_fac for x in available_items]
    return available_items_scaling


def get_x_y_vals(data: dict):
    """Reads and returns x- and y-values from dict."""
    x_vals = [float(x) for x in list(data.keys())]
    y_vals = list(data.values())
    return x_vals, y_vals


def plot_data(data_to_plot: dict):
    """Creates the visualisation of the data in ``data_to_plot``.

    Args:
        data_to_plot (dict): Dictionary containing data to be visualised.
            Has to contain the keys "AP", "Center_Dist", "Size_Similarity",
            "OS_Yaw", "OS_Pitch_Roll".
    """

    # get max depth
    max_depth = data_to_plot["eval_params"]["max_depth"]

    # setup all categories
    categories = ["AP", "Center_Dist", "Size_Similarity",
                  "OS_Yaw", "OS_Pitch_Roll"]
    subplot_categories = ["PR", *categories]
    assert all([key in data_to_plot.keys() for key in categories])

    accept_classes = []
    for cat, count in data_to_plot["GT_stats"].items():
        if count > 0:
            accept_classes.append(cat)

    plt.figure(figsize=(20, 12), dpi=100)

    # create subplot-axes
    ax_results, ax_spider, axes = create_all_axes(max_depth, 10)

    # 1st fill subplots (3-8)
    for idx, category in enumerate(subplot_categories):
        fill_and_finalize_subplot(
            category, data_to_plot, accept_classes, axes[idx], max_depth)

    # 2nd plot Spider plot
    create_spider_chart_plot(ax_spider, data_to_plot,
                             categories, accept_classes)

    # 3rd create subplot showing the table with result scores and labels
    create_result_table_and_legend_plot(
        ax_results, data_to_plot, axes[0].get_legend_handles_labels())

    plt.tight_layout()
    # plt.savefig("results.pdf")
    plt.show()


def prepare_data(json_path: str) -> dict:
    """Loads data from json-file.

    Args:
        json_path (str): Path to json-file from which data should be loaded
    """

    with open(json_path) as file_:
        data = json.load(file_)

    return data


if __name__ == "__main__":
    RESULT_PATH = sys.argv[1]

    if not os.path.exists(RESULT_PATH):
        raise Exception("File not found!")

    DATA_TO_PLOT = prepare_data(RESULT_PATH)

    plot_data(DATA_TO_PLOT)
