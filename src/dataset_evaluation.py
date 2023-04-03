import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image


def create_dataset_chart(dataset_path, save_as, width=0.7,
                         domain_names=["NOOP", "Fire", "Right", "Left", "Right-Fire", "Left-Fire"],
                         colors=["tab:orange", "tab:blue", "tab:green"]):
    # create figure and add axis descriptions
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlabel("Action Domain")
    ax.set_ylabel("Amount of Samples")
    ax.set_xticklabels(domain_names)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    subfolders = _get_subfolders(dataset_path)
    train_folder = list(filter(lambda folder: folder.endswith("train"), subfolders))[0]

    # get total samples per domain
    x = np.array(list(map(str, range(len(domain_names)))))
    samples_per_domain = [len(os.listdir(os.path.join(train_folder, domain))) for domain in x]

    # get duplicates per domain
    duplicates_per_domain = []
    for domain in os.listdir(train_folder):
        unique_samples, nb_uniques, total = get_uniques(os.path.join(train_folder, domain))
        duplicates_per_domain.append(total - nb_uniques)
        del unique_samples

    ax.bar(domain_names, samples_per_domain, width, color=colors[1], label=f"Total Samples")
    ax.bar(domain_names, duplicates_per_domain, width, color=colors[0], label="Duplicates")

    print(f"Domain: '{train_folder}'")
    print(f"Total Number of Samples: {sum(samples_per_domain)}")
    print(f"Distribution: {dict(zip(domain_names, samples_per_domain))}\n")
    print(f"Duplicates: {dict(zip(domain_names, duplicates_per_domain))}\n\n")

    ax.legend(ncol=2, bbox_to_anchor=(0.5, 1), loc="lower center")
    plt.savefig(save_as, bbox_inches="tight")


def create_dataset_comparison_chart(save_as, width=0.35, space=0.01, colors=["tab:orange", "tab:blue", "tab:green"],
                                    data_set_names=["Pac-Man", "Pac-Man-PP"],
                                    data_set_paths=["../res/datasets/PacMan_Ingame",
                                                    "../res/datasets/PacMan_PowerPill"]):
    domain_names = ["NOOP", "Up", "Right", "Left", "Down"]
    x = np.array(list(map(str, range(len(domain_names)))))
    x_float = x.astype(float)

    # create figure and add axis descriptions
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlabel("Action Domain")
    ax.set_ylabel("Amount of Samples")
    ax.set_xticklabels([""] + domain_names)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    shifts = [- (width / 2 + space), (width / 2 + space)]

    # create bars
    bars = []
    for i in range(len(data_set_names)):
        subfolders = _get_subfolders(data_set_paths[i])
        train_folder = list(filter(lambda folder: folder.endswith("train"), subfolders))[0]

        # get total samples per domain
        samples_per_domain = [len(os.listdir(os.path.join(train_folder, domain))) for domain in x]

        # get duplicates per domain
        duplicates_per_domain = []
        for domain in os.listdir(train_folder):
            unique_samples, nb_uniques, total = get_uniques(os.path.join(train_folder, domain))
            duplicates_per_domain.append(total - nb_uniques)
            del unique_samples

        bars.append(ax.bar(x_float + shifts[i], samples_per_domain, width, color=colors[i + 1],
                           label=f"{data_set_names[i]}"))
        if i == 0:
            label = "Duplicates"
        else:
            label = None
        bars.append(ax.bar(x_float + shifts[i], duplicates_per_domain, width, color=colors[0], label=label))

        print(f"Domain: '{train_folder}'")
        print(f"Total Number of Samples: {sum(samples_per_domain)}")
        print(f"Distribution: {dict(zip(domain_names, samples_per_domain))}\n")
        print(f"Duplicates: {dict(zip(domain_names, duplicates_per_domain))}\n\n")

    ax.legend(handles=[bars[0], bars[2], bars[1]], ncol=3, bbox_to_anchor=(0.5, 1), loc="lower center")
    plt.savefig(save_as, bbox_inches="tight")


def get_uniques(domain_folder):
    byte_samples = []
    file_names = []
    dtype = np.float32
    shape = None

    for i, item in enumerate(os.listdir(domain_folder)):
        # get file name
        file_name = os.path.join(domain_folder, item)
        file_names.append(file_name)

        # load
        img = Image.open(file_name)
        sample = np.array(img)

        # encode to bytes (so that it is hashable)
        byte_samples.append(sample.tobytes())
        dtype = sample.dtype
        shape = sample.shape

    uniques = set(byte_samples)

    # decode
    uniques = list(map(lambda byte_array: Image.fromarray(np.frombuffer(byte_array, dtype=dtype).reshape(shape)),
                       uniques))

    return uniques, len(uniques), len(byte_samples)


def _get_subfolders(path):
    subfolders = []
    for item in os.listdir(path):
        subfolder = os.path.join(path, item)
        if os.path.isdir(subfolder):
            subfolders.append(subfolder)
    return subfolders
