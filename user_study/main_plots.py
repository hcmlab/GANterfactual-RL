"""
Creates plots for the main values.
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def show_and_save_plt(ax ,file_name, y_label=None, title = None, ylim =None, label_size = 22, tick_size = 18):
    """
    Shows and saves the given plot and defines the appearance of the final plot.
    :param ax: the plot to be saved.
    :param file_name: save file name where the file is saved.
    :param y_label: the y axis label displayed
    :param title: titel of displayed in the plot (currently not used)
    :param ylim: limits of the y axis.
    :param label_size: font size of the label text
    :param tick_size: font size of the tick numbers
    """

    if y_label != None:
        plt.ylabel(y_label)
    plt.xlabel(None)
    #plt.supitle=(title)
    #ax.set(title=title)
    if ylim != None:
        ax.set(ylim=ylim)

    try:
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.label.set_size(label_size)
    except:
        try:
            plt.ylabel(y_label, fontsize=label_size)
            plt.xlabel(fontsize=label_size)
        except Exception as e:
            print(e)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    file_name = os.path.join('figures', file_name)
    if not (os.path.isdir(file_name)):
        os.makedirs(file_name)
        os.rmdir(file_name)
    plt.tight_layout()
    plt.savefig(file_name)

    plt.show()


if __name__ == "__main__":

    sns.set(palette='colorblind', style="whitegrid")
    plot_params = {"ci": 95}

    data = pd.read_csv('interesting_values.csv')
    data['condition'] = data.condition.apply(lambda x: 'Control' if x == 0 else 'CSE' if x == 1 else 'GANterfactual-RL')
    data.head()

    conditions = ["Control", "CSE", "GANterfactual-RL"]

    column_name = "retroScoreTotal"
    # plot itself
    ax = sns.barplot(x='condition', y=column_name, data=data, order=conditions, **plot_params)
    # significance
    x1, x2, x3 = 0, 1, 2,  # (first column: 0, see plt.xticks())
    y1, y2, h, col = 2.2, 2.5, 0.2, 'k'
    ## Dif Control - StarGAN
    plt.plot([x1, x1, x3, x3], [y2, y2 + h, y2 + h, y2], lw=1.5, c=col)
    plt.text((x1 + x3) * .5, y2 + h, "*", ha='center', va='bottom', color=col)
    ## Dif Olson StarGAN
    plt.plot([x2, x2, x3, x3], [y1, y1 + h, y1 + h, y1], lw=1.5, c=col)
    plt.text((x2 + x3) * .5, y1 + h, "*", ha='center', va='bottom', color=col)
    show_and_save_plt(ax, column_name, y_label='Score', title="Title", ylim=(0, 3))

    column_name = "trustTotal"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=conditions)
    show_and_save_plt(ax, column_name, y_label='Score', title="Title", ylim=(0, 5))

    column_name = "explSatisfactionRetroAvg"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=conditions)
    # significance Control vs Olson
    y2, h, col = 4.1, 0.2, 'k'
    plt.plot([x1, x1, x2, x2], [y2, y2 + h, y2 + h, y2], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y2 + h, "*", ha='center', va='bottom', color=col)
    show_and_save_plt(ax, column_name, y_label='Score', title="Title", ylim=(1, 5))

    column_name = "explSatisfactionTrustAvg"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=conditions)
    # significance Control vs Olson
    plt.plot([x1, x1, x2, x2], [y2, y2 + h, y2 + h, y2], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y2 + h, "*", ha='center', va='bottom', color=col)
    show_and_save_plt(ax, column_name, y_label='Score', title="Title", ylim=(1, 5))