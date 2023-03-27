"""
    This module analyzes the demographic questions about age,gender, Pacman experience and AI experience.
"""

import pandas as pd
import seaborn as sns
import numpy as np
from result_cleaning import convert_Y_to_1
from main_plots import show_and_save_plt


def analyze_distribution(data, columns):
    """
    Analyzes the distribution of the values in the given columns
    :param data: The dataframe to be analyzed
    :param columns: The name of the columns to be compared
    :return ax: a plot showing the distribution
    """
    df = data[columns]
    def get_column_number(column):
        """
        Converts the column names to their position in the original columns-array.
        Should the name not be in the columns array the name will not be converted.
        """
        if column in columns:
            return columns.index(column) + 1
        else:
            return column

    df = pd.melt(df)
    df.variable = df.variable.apply(get_column_number)
    ax = sns.barplot(x="variable", y="value", data=df)

    return ax


def analyze_demographic(data):
    """
    Helper function that inserts the AI experience columns into analyze distribution.
    """
    ax = analyze_distribution(data, ['experienceAI2[1]', 'experienceAI2[2]', 'experienceAI2[3]', 'experienceAI2[4]',
                                'experienceAI2[5]'])
    return ax


if __name__ == "__main__":
    sns.set(palette= 'colorblind', style="whitegrid")

    data = pd.read_csv('cleaned_data.csv')
    data['condition'] = data.condition.apply(lambda x: 'Control' if x==0 else 'CSE' if x==1 else 'GANterfactual-RL' )
    data.head()

    conditions = ["Control", "CSE", "GANterfactual-RL"]

    ages = data.age.values
    print('mean age:', ages.mean())
    ax = sns.barplot(x='condition', y='age', data=data, order=conditions)
    show_and_save_plt(ax, 'demographic/age', y_label='Age')


    # GENDER 1: male, 2: female, 3:prefer not to answer, 4:other
    genders = data.gender.values
    genders = genders[np.where(genders != 3)]
    genders = genders[np.where(genders != 4)]
    genders = genders - 1
    print('number females:', genders.sum())
    data_gender = data.loc[data.gender < 3]
    # set males to 0 and females to 1
    data_gender.gender = data_gender.gender.values - 1
    data_control = data_gender.loc[data_gender.condition == "Control"]
    print('Control percent females:', data_control.gender.values.mean())
    data_olson = data_gender.loc[data_gender.condition == "CSE"]
    print('Olson percent females:', data_olson.gender.values.mean())
    data_starGAN = data_gender.loc[data_gender.condition == "GANterfactual-RL"]
    print('StarGAN percent females:', data_starGAN.gender.values.mean())

    # ax = sns.barplot(x='condition', y='gender', data=data_gender, order=conditions, ci=None)
    # show_and_save_plt(ax, 'demographic/number_females', y_label='Percentage of Female Participants')

    # PACMAN EXP 1: never played, 2: <1year, 3: <5years, 4: >5years ago
    pacman_experience = data['experiencePacman'].values
    print('pacman exp median:', np.median(pacman_experience))
    print('pacman exp mean:', np.mean(pacman_experience))
    data['experiencePacman'] = data.experiencePacman.apply(lambda x: 'never' if x == 1 else '< 1 year' if x == 2 else
                                                           '< 5 years' if x == 3 else '> 5 years')
    ax = sns.catplot(x="condition", hue="experiencePacman", kind="count", hue_order=['never', '> 5 years',
                                                                                     '< 5 years', '< 1 year'],
                     data=data, order=conditions, legend=False,  aspect= 2);
    show_and_save_plt(ax, 'demographic/pacman_experience', y_label='Number of Participants')

    # AI VALUES
    attitude = data['outcomeAI[1]'].values
    attitude = attitude[np.where(attitude != 6)]
    print('mean attitude towards AI', np.mean(attitude))

    data_attiude = data.loc[data['outcomeAI[1]'] != 6]
    ax = sns.barplot(x='condition', y='outcomeAI[1]', data=data_attiude, order=conditions)
    show_and_save_plt(ax, 'demographic/Attitude_towards_AI', y_label='Attitude towards AI')

    for i in range(1,6):
        data["experienceAI2["+ str(i) + "]"] = data["experienceAI2["+ str(i) + "]"].apply(convert_Y_to_1)

    media = data['experienceAI2[1]'].values
    print('I know AI from the media:', np.sum(media))

    technology = data['experienceAI2[2]'].values
    print("technology:", np.sum(technology))
    technology_work = data['experienceAI2[3]'].values
    print('technology work:', np.sum(technology_work))
    course = data['experienceAI2[4]'].values
    print('AI related course', np.sum(course))
    research = data['experienceAI2[5]'].values
    print('research on AI', np.sum(research))

    data_random = data.loc[data.condition == "Control"]
    print('##########Condition 1##########')
    ax = analyze_demographic(data_random)
    show_and_save_plt(ax, 'demographic/AiExperience_Control', y_label='Percent of Participants', ylim=[0,1])

    data_olson = data.loc[data.condition=="CSE"]
    print('##########Condition 2##########')
    ax = analyze_demographic(data_olson)
    show_and_save_plt(ax, 'demographic/AiExperience_Olson', y_label='Percent of Participants', ylim=[0,1], label_size = 18, tick_size = 14)

    data_starGAN = data.loc[data.condition== "GANterfactual-RL"]
    print('##########Condition 3##########')
    ax = analyze_demographic(data_starGAN)
    show_and_save_plt(ax, 'demographic/AiExperience_StarGAN', y_label='Percent of Participants', ylim=[0,1],label_size = 18, tick_size = 14)

