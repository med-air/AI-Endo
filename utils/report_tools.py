import os
import shutil
import argparse
import cv2
import math
from glob import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# from lxml import etree
import numpy as np
from PIL import Image
import math
from matplotlib.pyplot import figure


P = [252, 233, 79, 114, 159, 207, 239, 41, 41, 173, 127, 168, 138, 226, 52,
     233, 185, 110, 252, 175, 62, 211, 215, 207, 196, 160, 0, 32, 74, 135, 164, 0, 0,
     92, 53, 102, 78, 154, 6, 143, 89, 2, 206, 92, 0, 136, 138, 133, 237, 212, 0, 52,
     101, 164, 204, 0, 0, 117, 80, 123, 115, 210, 22, 193, 125, 17, 245, 121, 0, 186,
     189, 182, 85, 87, 83, 46, 52, 54, 238, 238, 236, 0, 0, 10, 252, 233, 89, 114, 159,
     217, 239, 41, 51, 173, 127, 178, 138, 226, 62, 233, 185, 120, 252, 175, 72, 211, 215,
     217, 196, 160, 10, 32, 74, 145, 164, 0, 10, 92, 53, 112, 78, 154, 16, 143, 89, 12,
     206, 92, 10, 136, 138, 143, 237, 212, 10, 52, 101, 174, 204, 0, 10, 117, 80, 133, 115,
     210, 32, 193, 125, 27, 245, 121, 10, 186, 189, 192, 85, 87, 93, 46, 52, 64, 238, 238, 246]

P = P * math.floor(255*3/len(P))
l = int(255 - len(P)/3)
P = P + P[3:(l+1)*3]
P = [0,0,0] + P

def save_indexed_png(fname, label_map, palette=P):
    if label_map.max() > 255:
        label_map = np.remainder(label_map, 255)
    label_map = np.squeeze(label_map.astype(np.uint8))
    im = Image.fromarray(label_map, 'P')
    im.putpalette(palette)
    im.save(fname, 'PNG')


def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)

# def parse_anno_file(cvat_xml, image_name):
#     root = etree.parse(cvat_xml).getroot()
#     anno = []
#
#     image_name_attr = ".//image[@name='{}']".format(image_name)
#     for image_tag in root.iterfind(image_name_attr):
#         image = {}
#         for key, value in image_tag.items():
#             image[key] = value
#         image['shapes'] = []
#         for poly_tag in image_tag.iter('polyline'):
#             polygon = {'type': 'polygon'}
#             for key, value in poly_tag.items():
#                 polygon[key] = value
#             image['shapes'].append(polygon)
#         for box_tag in image_tag.iter('box'):
#             box = {'type': 'box'}
#             for key, value in box_tag.items():
#                 box[key] = value
#             box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
#                 box['xtl'], box['ytl'], box['xbr'], box['ybr'])
#             image['shapes'].append(box)
#         image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
#         anno.append(image)
#     return anno

def create_mask_file(mask, polygon, label):
    points = [tuple(map(float, p.split(','))) for p in polygon.split(';')]
    points = np.array([(int(p[0]), int(p[1])) for p in points])
    points = points.astype(int)
    # mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255), thickness=5)
    height, width = mask.shape
    mask_array = np.zeros((height, width, 3), dtype=np.uint8)
    mask_array = cv2.fillPoly(mask_array, [points], color=(0, 0, 255))
    mask[mask_array[..., -1] != 0] = label

    return mask


def find_clips(frame_idxs):
    clip_durations = []  # [[start1, end1], ...]
    if len(frame_idxs) != 0:
        gaps = [right - left for left, right in zip(frame_idxs[:-1], frame_idxs[1:])]

        duration = [frame_idxs[0]]
        for idx, gap in enumerate(gaps):
            if gap > 1:
                duration.append(frame_idxs[idx])
                clip_durations.append(duration[1] - duration[0])
                duration = [frame_idxs[idx + 1]]
        duration.append(frame_idxs[-1])
        clip_durations.append(duration[1] - duration[0])

    return clip_durations

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# Referred to https://stackoverflow.com/questions/59638155/how-to-set-0-to-white-at-a-uneven-color-ramp
def generate_phase_band(labels, file_name="tem.png", colors=None):

    len_labels = len(labels)
    x, y = np.meshgrid(np.linspace(0, len_labels, len_labels+1), np.linspace(0, 1, 2))
    z_gt = np.array([labels])

    label_width = 0.005 * len_labels
    fig1, ax1 = plt.subplots()

    img = ax1.pcolormesh(x, y, z_gt, cmap=colors)
    img.set_clim(1, 20)
    ax1.margins(x=0)
    ax1.margins(y=0)
    plt.axis("off")
    plt.gcf().set_size_inches(label_width, 2)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.0)
    plt.clf()
    plt.close()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    kwargs.update({"cmap": "GnBu"})
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center", fontsize=20)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_pie(labels, pie_name):
    # fig1, ax1 = plt.subplots()
    label_list = [2, 3, 4, 1]
    counts = []
    for label in label_list:
        counts.append(labels.count(label))

    fig1, ax1 = plt.subplots()
    img = ax1.pie(counts, colors=[[83/255, 129/255, 186/255],
                                  [129/255, 172/255, 211/255],
                                  [170/255, 201/255, 224/255],
                                  [203/255, 218/255, 237/255]])
    ax1.margins(x=0)
    ax1.margins(y=0)
    plt.axis("off")
    plt.savefig(pie_name, bbox_inches='tight', pad_inches=0.0)
    plt.clf()
    plt.close()

def get_durations(labels):
    label_list = [2, 3, 4, 1]
    counts = []
    for label in label_list:
        counts.append(labels.count(label))
    counts.append(len(labels))

    return counts

from sklearn import metrics
from sklearn.preprocessing import normalize
def generate_transition(labels, file_name):
    phase_dict_key = [2, 3, 4, 1]
    confusion = metrics.confusion_matrix(labels[:-1], labels[1:], labels=phase_dict_key, normalize="true")
    # for idx in range(4):
    #     confusion[idx, idx] = 0
    # confusion = normalize(confusion, axis=1, norm='l1')
    fig, ax = plt.subplots()
    im, cbar = heatmap(confusion, phase_dict_key, phase_dict_key, ax)
    cbar.remove()
    frame1 = plt.gca()
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    frame1.axes.margins(x=0)
    frame1.axes.margins(y=0)

    # fig.tight_layout()
    # plt.ylabel("Annotation")
    # pstr = "Prediction {}; acc {:>10.4f}".format(base_name.split("_")[0], acc)
    # plt.xlabel("Prediction")
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def get_score_A(labels):
    injection_proporation = labels.count(2) / len(labels)
    dissection_proporation = labels.count(3) / len(labels)
    if dissection_proporation != 0:
        score_A = -2.516 * 10 ** (-5) * (injection_proporation / dissection_proporation) # + 0.1945
    else:
        score_A = 0
    return score_A

def get_score_B(labels):
    frame_idxs = [i for i, x in enumerate(labels) if x == 3]
    clips = find_clips(frame_idxs)
    if len(clips) != 0:
        ratio = len(clips) / sum(clips)
    else:
        ratio = 0
    return 0.0009609 * math.exp(-0.0001791 * ratio) - 0.0009

def combine_logs(log_files):
    pd_log_list = []
    for log_file in log_files:
        pd_data = pd.read_csv(log_file, header=0)
        pd_log_list.append(pd_data)
    pd_log = pd.concat(pd_log_list, axis=1)

    return pd_log

def get_meta(log_files):

    pd_data_list = []
    for log_file in log_files:
        pd_data = pd.read_csv(log_file, header=0)
        pd_data_list.append(pd_data)
    pd_data = pd.concat(pd_data_list, axis=0)
    pd_data = pd_data.sort_values(by=["Time"])

    pd_data = pd_data[pd_data["Status"] != "--"]
    phases = pd_data["Combine"].tolist()
    status = pd_data["Status"].tolist()
    trainees = pd_data["Trainee"].tolist()

    trainees = list(set(trainees))
    train_str = ", ".join(trainees)
    mentors = list(set(pd_data["Trainer"].tolist()))
    mentor_str = ", ".join(mentors)
    bed = str(pd_data["Bed"].tolist()[0])
    date = pd_data["Time"].tolist()[0].split("-")[0]

    phase_dict = {'idle': 1, 'marking': 2, 'injection': 3, 'dissection': 4}
    status_dict = {"Indepedent": 6, "Help": 7, "TakeOver": 8}
    trainee_dict = {}
    trainee_dict = {"A": 13, "B": 14, "C": 15, "D": 16}

    pd_data = pd_data.replace({"Combine": phase_dict, "Status": status_dict, "Trainee": trainee_dict})

    phases = pd_data["Combine"].tolist()
    status = pd_data["Status"].tolist()
    trainees = pd_data["Trainee"].tolist()

    return phases, status, trainees, train_str, mentor_str, bed, date

def generate_report(log_dir):

    if not os.path.isdir("./reports/components"):
        os.makedirs("./reports/components")

    log_files = glob(os.path.join(log_dir, "*.csv"))
    case_names = [os.path.basename(log_file).split("_")[0] for log_file in log_files]
    case_names = list(set(case_names))
    for case_name in case_names:
        log_files = glob(os.path.join(log_dir, "{}*.csv".format(case_name)))
        phases, status, trainees, train_str, mentor_str, bed, date = get_meta(log_files)

        phase_file = "./reports/components/{}_time_phase.png".format(case_name)
        status_file = "./reports/components/{}_time_status.png".format(case_name)
        trainee_file = "./reports/components/{}_time_trainee.png".format(case_name)
        pie_file = "./reports/components/{}_phase_pie.png".format(case_name)
        transition_file = "./reports/components/{}_transition.png".format(case_name)

        generate_phase_band(phases, file_name=phase_file, colors="tab20c")
        generate_phase_band(status, file_name=status_file, colors="tab20b")
        generate_phase_band(trainees, file_name=trainee_file, colors="tab20c")
        plot_pie(phases, pie_name=pie_file)
        generate_transition(phases, transition_file)


        # combine all infos
        im = plt.imread("./configs/report_template.png")
        sh, sw, d = im.shape
        figure(figsize=(sw, sh), dpi=600)
        fig, ax = plt.subplots()
        ax = plt.gca()
        ax.set_xlim(0, sw)
        ax.set_ylim(0, sh)
        plt.imshow(im, extent=[0, sw, 0, sh])

        # add header
        plt.text(790, 7945, train_str, fontsize=3)  # add text
        plt.text(790, 7750, mentor_str, fontsize=3)  # add text
        plt.text(4660, 7945, bed, fontsize=3)  # add text
        plt.text(4660, 7750, date, fontsize=3)  # add text

        # Add basic information
        phase = Image.open(phase_file)
        plt.imshow(phase, extent=[281, 5718, 7206, 7430])

        status = Image.open(status_file)
        plt.imshow(status, extent=[281, 5718, 6667, 6889])

        trainee = Image.open(trainee_file)
        plt.imshow(trainee, extent=[281, 5718, 6090, 6352])

        # Add durations
        counts = get_durations(phases)
        locations = [[1416, 5291], [1416, 5046], [1416, 4806], [1416, 4566], [1416, 4320]]
        for idx, count in enumerate(counts):
            location = locations[idx]
            plt.text(location[0], location[1], "{:>8d}".format(count), fontsize=4)  # add text

        # add proportion
        locations = [[2520, 5281], [2520, 5046], [2520, 4806], [2520, 4566]]
        total = counts[-1]
        for idx, count in enumerate(counts[:-1]):
            location = locations[idx]
            plt.text(location[0], location[1], "{:>2.2f}".format(count / total), fontsize=4)  # add text

        # add pie file
        pie_duration = Image.open(pie_file)
        plt.imshow(pie_duration, extent=[3570, 4740, 4328, 5498])

        # add transition
        transition = Image.open(transition_file)
        plt.imshow(transition, extent=[3850, 5607, 1760, 3306])

        # Add scores
        score_A = phases.count(3) / phases.count(4)
        plt.text(2660, 2877, "{:2.2f}".format(score_A), fontsize=4)  # add text
        if score_A > 0.06:  #
            score_A = "A"
        else:
            score_A = "B"
        plt.text(2460, 550, "{}".format(score_A), fontsize=4)  # add text
        # score_B = get_score_B()

        score_B = (get_score_B(phases) * 10 ** 6 - 60)*100
        plt.text(2979, 1820, "{:2.0f}".format(abs(score_B)), fontsize=4)  # add text
        if score_B > 89:  #
            score_B = "A"
        else:
            score_B = "B"
        plt.text(4400, 550, "{}".format(score_B), fontsize=4)  # add text

        plt.axis("off")
        # plt.show()
        plt.savefig("./reports/{}_report.png".format(case_name), bbox_inches='tight', dpi=800, pad_inches=0.0)
        plt.clf()
        plt.close()