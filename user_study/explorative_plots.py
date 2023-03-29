import pandas as pd
import seaborn as sns
from main_plots import show_and_save_plt
import os


def save_path(column_name):
    return os.path.join("explorative", column_name)


if __name__ == "__main__":
    sns.set(palette='colorblind', style="whitegrid")
    plot_params = {"ci": 95}

    data = pd.read_csv('cleaned_data.csv')
    data['condition'] = data.condition.apply(lambda x: 'Control' if x == 0 else 'CSE' if x == 1 else 'GANterfactual-RL')
    data.head()

    conditions = ["Control", "CSE", "GANterfactual-RL"]

    column_name = "retroPacmanTotal"
    ax = sns.barplot(x='condition', y=column_name, data=data, order=conditions)
    show_and_save_plt(ax, save_path(column_name), y_label='Score', title="Title", ylim=(0, 3))

    column_name = "totalTime"
    ax = sns.boxplot(x='condition', y=column_name, data=data, order=conditions)
    show_and_save_plt(ax, save_path(column_name), y_label='Time', title="Title")
