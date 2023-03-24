"""
    Converts the original raw result csv files to one more readable csv with less unnecessary information.
    Also, calculates and combines scores, and filters participants.
"""

import pandas as pd

# The columns that we want to keep
interesting_columns = ["id", "seed", "age", "gender", "gender[other]", "experiencePacman",
                       "experienceAI2[6]", "experienceAI2[1]", "experienceAI2[2]", "experienceAI2[3]",
                       "experienceAI2[4]", "experienceAI2[5]", "experienceAI2[other]", "outcomeAI[1]", "condition",
                       "retrospection1[1]", "retrospection1[2]", "retrospection1[3]", "retrospection1[4]",
                       "retrospection1[5]", "retrospection1[6]", "Intention1", "RetrospectionConf1[predictConf1]",
                       "UsedCounterfactual1",
                       "retrospection2[1]", "retrospection2[2]", "retrospection2[3]", "retrospection2[4]",
                       "retrospection2[5]", "retrospection2[6]", "Intention2", "RetrospectionConf2[predictConf1]",
                       "UsedCounterfactual2",
                       "retrospection3[1]", "retrospection3[2]", "retrospection3[3]", "retrospection3[4]",
                       "retrospection3[5]", "retrospection3[6]", "Intention3", "RetrospectionConf3[predictConf1]",
                       "UsedCounterfactual3",
                       "slider0", "slider1", "slider2", "slider3", "slider4", "slider5", "slider6", "slider7",
                       "slider8", "slider9", "slider10", "slider11", "slider12", "slider13", "slider14",
                       "explSatisfaction1[1]", "explSatisfaction1[2]", "explSatisfaction1[3]", "explSatisfaction1[4]",
                       "comment1",
                       "explSatisfaction2[1]", "explSatisfaction2[2]", "explSatisfaction2[3]", "explSatisfaction2[4]",
                       "comment2",
                       "sliderLeft0", "sliderLeft1", "sliderLeft2", "sliderLeft3", "sliderLeft4", "sliderLeft5",
                       "sliderLeft6", "sliderLeft7", "sliderLeft8", "sliderLeft9", "sliderLeft10", "sliderLeft11",
                       "sliderLeft12", "sliderLeft13", "sliderLeft14",
                       "sliderRight0", "sliderRight1", "sliderRight2", "sliderRight3", "sliderRight4", "sliderRight5",
                       "sliderRight6", "sliderRight7", "sliderRight8", "sliderRight9", "sliderRight10", "sliderRight11",
                       "sliderRight12", "sliderRight13", "sliderRight14",
                       "trustPoints1", "trustPoints2", "trustPoints3",
                       "trustSurvive1", "trustSurvive2", "trustSurvive3",
                       "trustConfidence1[confidence1]", "trustConfidence2[confidence1]", "trustConfidence3[confidence1]",
                       "trustUsedCF1", "trustUsedCF2", "trustUsedCF3",
                       "bonus",
                       "retroTime", "trustTime", "totalTime"
                       ]


def rename_time(data_frame, successive_name, new_name):
    # In our survey tool the time is stored under a generic name in a column just before an empty column
    # with the name of the first question on that page.
    index = data_frame.columns.get_loc(successive_name)
    index -= 1
    old_column_name = data_frame.columns[index]
    data_frame = data_frame.rename(columns={old_column_name: new_name}, errors="raise")
    return data_frame


def rename_times(data_frame):
    data_frame = rename_time(data_frame, 'ageTime', 'demographicsTime')
    data_frame = rename_time(data_frame, 'retrospectionVideo1Time', 'retro1Time')
    data_frame = rename_time(data_frame, 'retrospectionVideo2Time', 'retro2Time')
    data_frame = rename_time(data_frame, 'retrospectionVideo3Time', 'retro3Time')
    data_frame = rename_time(data_frame, 'trustDescription1Time', 'trust1Time')
    data_frame = rename_time(data_frame, 'trustDescription2Time', 'trust2Time')
    data_frame = rename_time(data_frame, 'trustDescription3Time', 'trust3Time')
    return data_frame


def combine_times(data_frame):
    data = data_frame
    data["retroTime"] = data["retro1Time"] + data["retro2Time"] + data["retro3Time"]
    data["trustTime"] = data["trust1Time"] + data["trust2Time"] + data["trust3Time"]
    data["totalTime"] = data["retroTime"] + data["trustTime"]
    return data


def delete_columns(data, keep_columns=interesting_columns):
    """
    Deletes all columns form the given dataframe that are not in *keep_columns*
    :param data: the dataframe where the columns should be deleted
    :param keep_columns: the columns to be kept
    :return: the dataframe without unnecessary columns
    """
    to_delete = []
    for i in range(0, len(data.columns)):
        column_name = data.columns[i]
        if column_name not in keep_columns:
            to_delete.append(column_name)
    data = data.drop(to_delete, axis=1)
    return data


def eval_retrospection(data_frame, number):
    """
    Evaluation of the object selection during the individual retrospection tasks.
    Calculates and defines the score of this task and checks if the participants got the agent specific goals.
    :param data_frame: the dataframe containing the survey results
    :param number: the number of the agent (1 = blue ghost,2 = power pill or 3 = fear ghost)
    :return: the same data frame with two additional columns for the score and a binary value showing whether the
     participants got the agent specific goal.
    """
    def get_name(index):
        return 'retrospection' + str(number) + '[' + str(index) + ']'
    pacman = data_frame[get_name(1)].tolist()
    normal_pill = data_frame[get_name(2)].tolist()
    power_pill = data_frame[get_name(3)].tolist()
    ghost = data_frame[get_name(4)].tolist()
    blue_ghost = data_frame[get_name(5)].tolist()
    cherry = data_frame[get_name(6)].tolist()

    length = len(pacman)

    # define score calculation
    def got_goal(index):
        i = index
        # check if they selected to many items
        if (pacman[i]== "Y") + (normal_pill[i]== "Y") + (power_pill[i]== "Y") + (ghost[i]== "Y") +\
                (blue_ghost[i]== "Y") + (cherry[i]== "Y") > 2:
            points = 0
        # here the score's per object and task are defined and added together
        else:
            points = 0
            # blue ghost agent
            if number == 1:
                if blue_ghost[i] == "Y":
                    points += 1
            # power pill agent
            elif number == 2:
                if power_pill[i] == "Y":
                    points += 1
            # agent afraid of ghosts
            elif number == 3:
                if ghost[i] == "Y":
                    points += 1
            else:
                print('number', number, 'not implemented')
                points = 'Nan'
        return points

    def calculate_points(index):
        i = index
        # check if they selected to many items
        if (pacman[i]== "Y") + (normal_pill[i]== "Y") + (power_pill[i]== "Y") + (ghost[i]== "Y") +\
                (blue_ghost[i]== "Y") + (cherry[i]== "Y") > 2:
            points = 0
        # here the score's per object and task are defined and added together
        else:
            points = 0
            # blue ghost agent
            if number == 1:
                if blue_ghost[i] == "Y":
                    points += 1
                    if (normal_pill[i]== "Y") or (power_pill[i] == "Y") or (ghost[i]== "Y") or (cherry[i]== "Y"):
                        points -= 1
            # power pill agent
            elif number == 2:
                if power_pill[i] == "Y":
                    points += 1
                    if (normal_pill[i]== "Y") or (blue_ghost[i] == "Y") or (ghost[i]== "Y") or (cherry[i]== "Y"):
                        points -= 1
            # agent afraid of ghosts
            elif number == 3:
                if ghost[i] == "Y":
                    points += 1
                    if (normal_pill[i]== "Y") or (power_pill[i] == "Y") or (blue_ghost[i]== "Y") or (cherry[i]== "Y"):
                        points -= 1
            else:
                print('number', number, 'not implemented')
                points = 'Nan'
        return points

    goals = []
    points = []
    for j in range(length):
        goals.append(got_goal(j))
        points.append(calculate_points(j))

    column_name = 'retrospection' + str(number) + 'Goal'
    data_frame[column_name] = goals

    column_name = 'retrospection' + str(number) + 'Points'
    data_frame[column_name] = points

    return data_frame


def eval_retrospection_pacman(data_frame, number):
    """
    Evaluation of the object selection during the retrospection task.
    Checks if the participants got Pacman as important for the agent.
    :param data_frame: the dataframe containing the survey results
    :param number: the number of the agent (1 = power pill,2 = normal or 3 = fear ghost)
    :return: the same data frame with one additional column with a binary value showing whether the
     participants got Pacman.
    """
    def get_name(index):
        return 'retrospection' + str(number) + '[' + str(index) + ']'
    pacman = data_frame[get_name(1)].tolist()
    normal_pill = data_frame[get_name(2)].tolist()
    power_pill = data_frame[get_name(3)].tolist()
    ghost = data_frame[get_name(4)].tolist()
    blue_ghost = data_frame[get_name(5)].tolist()
    cherry = data_frame[get_name(6)].tolist()

    length = len(pacman)

    # define score calculation
    def got_pacman(index):
        i = index
        # check if they selected to many items
        if (pacman[i]== "Y") + (normal_pill[i]== "Y") + (power_pill[i]== "Y") + (ghost[i]== "Y") +\
                (blue_ghost[i]== "Y") + (cherry[i]== "Y") > 2:
            points = 0
        # did they select pacman?
        else:
            points = 0
            if pacman[i] == "Y":
                points += 1
        return points

    pacmans = []
    for j in range(length):
        pacmans.append(got_pacman(j))

    column_name = 'retrospection' + str(number) + 'Pacman'
    data_frame[column_name] = pacmans

    return data_frame


def eval_trust(data_frame, agent_number):
    """
    simple first evaluation of the trust task that checks whether the participants got the correct agent.
    :param data_frame: the dataframe containing the survey results
    :param agent_number: the number of the comparison (1,2 or 3)
    :return: the same data_frame but with an added binary column 'trust' + *agent_number* + 'correct' storing whether
    the participants where correct.
    """
    # the correct answer for the agent with the most points
    column_name = 'trustPoints' + str(agent_number)
    answer_arr = data_frame[column_name]
    correct_answers_dict_points = {1:1.0,2:1.0, 3:1.0}
    resulting_column_name = 'trustPoints' + str(agent_number) + 'correct'
    correct_answer_arr = []
    for entry in answer_arr:
        if entry == correct_answers_dict_points[agent_number]:
            correct = 1;
        else:
            correct = 0;
        correct_answer_arr.append(correct)

    data_frame[resulting_column_name] = correct_answer_arr

    # for the second agent comparison there is no correct answer for the survive question since the agents are so close
    if agent_number == 2:
        return data_frame

    # the correct answer for the agent that survives the longest
    column_name = 'trustSurvive' + str(agent_number)
    answer_arr = data_frame[column_name]
    correct_answers_dict_survive = {1: 2.0, 2: 2.0, 3: 2.0}
    resulting_column_name = 'trustSurvive' + str(agent_number) + 'correct'
    correct_answer_arr = []
    for entry in answer_arr:
        if entry == correct_answers_dict_survive[agent_number]:
            correct = 1;
        else:
            correct = 0;
        correct_answer_arr.append(correct)

    data_frame[resulting_column_name] = correct_answer_arr

    return data_frame


def filter_cf_sliders(data_frame):
    """
    Calculates how much the participants used the counterfactual explanation slider during the agent understanding task.
    Removes participants that did not watch any counterfactual explanations during the agent understanding task
    in the counterfactual conditions.
    """
    data = data_frame
    data["agent1_slider_sum"] = data["slider0"]
    for i in range(1, 5):
        column_name = "slider" + str(i)
        data["agent1_slider_sum"] += data[column_name]

    data["agent2_slider_sum"] = data["slider5"]
    for i in range(6, 10):
        column_name = "slider" + str(i)
        data["agent2_slider_sum"] += data[column_name]

    data["agent3_slider_sum"] = data["slider10"]
    for i in range(11, 15):
        column_name = "slider" + str(i)
        data["agent3_slider_sum"] += data[column_name]

    rows_to_delete = []
    for rname in data.index:
        row = data.loc[rname]

        # in the control condition there were no sliders
        if row["condition"] != 0:
            # the value of a slider that has not been clicked is -1. If they clicked the slider it goes from 0 to 1.
            # Therefore participants did not watch any counterfactuals if and only if the summed value is -5.
            if (row["agent1_slider_sum"] == -5.0) or (row["agent2_slider_sum"] == -5.0) or (row["agent3_slider_sum"] == -5.0):
                rows_to_delete.append(rname)

    print(rows_to_delete)
    data = data.drop(index=rows_to_delete)
    return data


def filter_cf_sliders_trust(data_frame):
    """
    Calculates how much the participants used the counterfactual explanation slider during each of the trust tasks.
    """
    data = data_frame
    sum_name = "trust1_slider_sum"
    data[sum_name] = data["sliderRight0"] + data["sliderLeft0"]
    for i in range(1, 5):
        column_name = "sliderLeft" + str(i)
        data[sum_name] += data[column_name]
        column_name = "sliderRight" + str(i)
        data[sum_name] += data[column_name]

    sum_name = "trust2_slider_sum"
    data[sum_name] = data["sliderRight5"] + data["sliderLeft5"]
    for i in range(6, 10):
        column_name = "sliderLeft" + str(i)
        data[sum_name] += data[column_name]
        column_name = "sliderRight" + str(i)
        data[sum_name] += data[column_name]

    sum_name = "trust3_slider_sum"
    data[sum_name] = data["sliderRight10"] + data["sliderLeft10"]
    for i in range(11, 15):
        column_name = "sliderLeft" + str(i)
        data[sum_name] += data[column_name]
        column_name = "sliderRight" + str(i)
        data[sum_name] += data[column_name]

    return data


def combine_sliders(data_frame):
    """
    Combines the slider values for all 3 agents and the values for all 3 trust comparisions.
    :param data_frame:
    :return:
    """
    data = data_frame
    data["retroSliders"] = data["agent1_slider_sum"] + data["agent2_slider_sum"] + data["agent3_slider_sum"]
    data["trustSliders"] = data["trust1_slider_sum"] + data["trust2_slider_sum"] + data["trust3_slider_sum"]
    return data


def calculate_scores(data_frame):
    """
    Combines the scores from the individual agent analyses and agent comparisons.
    """
    data = data_frame
    data['retroScoreTotal'] = (
            data['retrospection1Points'] + data['retrospection2Points'] + data['retrospection3Points'])
    data['retroGoalTotal'] = (data['retrospection1Goal'] + data['retrospection2Goal'] + data['retrospection3Goal'])
    data['retroPacmanTotal'] = (
            data['retrospection1Pacman'] + data['retrospection2Pacman'] + data['retrospection3Pacman'])

    data["trustPointsTotal"] = (
                data['trustPoints1correct'] + data['trustPoints2correct'] + data['trustPoints3correct'])
    data["trustSurviveTotal"] = (data['trustSurvive1correct'] + data['trustSurvive3correct'])
    data["trustTotal"] = data["trustPointsTotal"] + data["trustSurviveTotal"]
    return data


def satisfaction_analysis(number, data_frame):
    '''
    Helper function to analyze the participants satisfaction in in each tast (Trust and Retrospection)
    :param number: specifies the Task: 1 = Retrospection, 2= Trust
    :param data: dataframe containing the data
    :return:
    '''

    data = data_frame
    # getting the correct column name for the given task number
    column_name = 'explSatisfaction' + str(number)
    if number == 1:
        result_name =  'explSatisfaction' + 'Retro' + 'Avg'
    elif number == 2:
        result_name = 'explSatisfaction' + 'Trust' + 'Avg'
    else:
        print('number not implemented')
    # inverting the negative question 3 to be inline with the other positive questions
    data[column_name + '[3]'] = 6 - data[column_name + '[3]']

    # calculating the average satisfaction for all questions that were actually asked
    data[result_name] = data[column_name + '[' + str(1) + ']'] +  data[column_name + '[' + str(2) + ']'] \
                        + data[column_name + '[' + str(3) + ']'] + data[column_name + '[' + str(4) + ']']
    data[result_name] = data[result_name]/4

    return data


def convert_Y_to_1(x):
    """
    Used to convert answers that only show Y for yes.
    """
    if x == "Y":
        return 1.0
    else:
        return 0.0


def evaluate_used_cf(data_frame):
    """
    Counts how often the participants stated that they used the counterfactuals for each task.
    :param data_frame:
    :return:
    """
    data = data_frame

    # Retrospection Task.
    resulting_column = "used_Counterfactual_retro"
    first_column = True
    for i in range(1, 4):
        if first_column:
            data[resulting_column] = data["UsedCounterfactual" + str(i)].apply(convert_Y_to_1)
            first_column = False
        else:
            data[resulting_column] += data["UsedCounterfactual" + str(i)].apply(convert_Y_to_1)

    # Trust Task.
    resulting_column = "used_Counterfactual_trust"
    first_column = True
    for i in range(1, 4):
        if first_column:
            data[resulting_column] = data["trustUsedCF" + str(i)].apply(convert_Y_to_1)
            first_column = False
        else:
            data[resulting_column] += data["trustUsedCF" + str(i)].apply(convert_Y_to_1)

    return data


if __name__ == '__main__':
    # The following block was used to load the raw data from the survey tool and filter people who did not finish the
    # survey. After that we manually removed some columns to anonymize the data.
    # files = []
    # file_names = ["final_download.csv"]
    # for file_name in file_names:
    #     data_frame = pd.read_csv(file_name, sep=',')
    #     files.append(data_frame)
    #
    # # raw fusion of the results
    # resulting_frame = pd.concat(files, axis=0, ignore_index=True, sort=False)
    # # resulting_frame.to_csv('raw_fusion.csv')
    #
    # # remove participants that did not finish the survey
    # resulting_frame=resulting_frame[resulting_frame.lastpage == 17]
    # resulting_frame.to_csv('finishedsurvey.csv')

    # The actuall cleaning of the data starts here with the anonymized data:
    resulting_frame = pd.read_csv("finishedsurvey_anonymized.csv", sep=',')

    # combine the time taken for each main task
    resulting_frame = rename_times(resulting_frame)
    resulting_frame = combine_times(resulting_frame)

    # only keep interesting columns
    resulting_frame = delete_columns(resulting_frame, interesting_columns)

    # evaluate whether participants chose the correct items during the retrospection task
    for i in range(1,4):
        resulting_frame = eval_retrospection(resulting_frame,i)
        resulting_frame = eval_retrospection_pacman(resulting_frame, i)
    # resulting_frame.to_csv('retrospection_evaluation.csv')


    for i in range(1,4):
        resulting_frame = eval_trust(resulting_frame,i)
    # resulting_frame.to_csv('trust_evaluation.csv')

    resulting_frame = filter_cf_sliders(resulting_frame)
    # resulting_frame.to_csv('filtered_sliders.csv')

    resulting_frame = calculate_scores(resulting_frame)
    # resulting_frame.to_csv('scores.csv')

    resulting_frame = filter_cf_sliders_trust(resulting_frame)

    resulting_frame = combine_sliders(resulting_frame)
    # resulting_frame.to_csv('slider_values.csv')

    resulting_frame = satisfaction_analysis(1, resulting_frame)
    resulting_frame = satisfaction_analysis(2, resulting_frame)
    # resulting_frame.to_csv('explanation_satisfaction.csv')

    resulting_frame = evaluate_used_cf(resulting_frame)

    resulting_frame.to_csv('cleaned_data.csv')

    # generate a data frame that only contains the "interesting" aggregated values
    interesting_values = pd.DataFrame()
    column_names = ["condition", "retroScoreTotal", "retroPacmanTotal", "trustTotal",
                    "explSatisfactionRetroAvg", "explSatisfactionTrustAvg",
                    "age", "gender", "totalTime", "retroSliders", "trustSliders",
                    "used_Counterfactual_retro", "used_Counterfactual_trust",
                    "Intention1", "Intention2", "Intention3"]

    for name in column_names:
        interesting_values[name] = resulting_frame[name]

    interesting_values.to_csv('interesting_values.csv')

    # creating easy to use data frames for Mann Whitney tests with Jamovi
    column_names = ["condition", "retroScoreTotal", "trustTotal",
                    "explSatisfactionRetroAvg", "explSatisfactionTrustAvg"]

    main_values = pd.DataFrame()
    for name in column_names:
        main_values[name] = resulting_frame[name]

    main_values_control_olson = main_values[(main_values.condition == 0) | (main_values.condition == 1)]
    main_values_control_olson.to_csv('mv_control_olson.csv')

    main_values_control_olson = main_values[(main_values.condition == 0) | (main_values.condition == 2)]
    main_values_control_olson.to_csv('mv_control_starGAN.csv')

    main_values_control_olson = main_values[(main_values.condition == 1) | (main_values.condition == 2)]
    main_values_control_olson.to_csv('mv_olson_starGAN.csv')

