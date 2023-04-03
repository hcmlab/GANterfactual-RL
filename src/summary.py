import os
from os import listdir
from os.path import isfile

import numpy as np
import pandas as pd
from PIL import Image
import keras
import torch

import src.olson.model as olson_model
from src.dataset_generation import _preprocess
from src.evaluation import Evaluator
from src.util import init_environment, get_agent_prediction_from_stacked_frames, generate_counterfactual, \
    restrict_tf_memory, generate_olson_counterfactual, load_baselines_model, load_olson_models

from baselines.common.tf_util import adjust_shape
from src.star_gan.model import Generator


def generate_highlights_div_summary(env_name, agent, num_frames, num_simulations, interval_size, save_dir,
                                    power_pill_objective=False, max_noop=1, agent_type = "keras",
                                    ablate_agent=False):
    """
    Implementation of the HIGHLIGHTS-DIV algorithm from
    "HIGHLIGHTS: Summarizing Agent Behavior to People" by Amir et al.
    Implemented to only save single frames instead of trajectories (e.g. l=1 and statesAfter=0).

    :param env_name: Name of the Gym environment that the summary should be generated for.
    :param agent: The trained agent whose behavior should be summarized.
    :param num_frames: Number of frames that should be generated (parameter k in the paper).
    :param num_simulations: Number of episodes that should be executed
    :param interval_size: Amount of states that have to be between two states that are included into the summary.
    :param save_dir: Path to a directory that will be created and filled with num_frames summary frames.
    :param power_pill_objective: Whether the Power-Pill objective for Pac-Man is used.
    :param max_noop: Maximum amount of NOOPs executed at the start of each episode.
    :param agent_type: the type of agent. Accepts "keras" for keras models, "acer" for acer baseline models and "olson"
        for a pytorch model with the same architecture as in Olson et al.
    :param ablate_agent: Whether the laser canon should be hidden from frames that are input to the agent.
    :return: None
    """
    # init
    runs = 0
    summary_importances = []
    summary_frames = []
    summary_actions = []
    remaining_interval = 0
    cummulative_reward = 0
    cummulative_steps = 0

    while runs < num_simulations:
        # init environment
        env_wrapper, skip_frames = init_environment(env_name, power_pill_objective, agent_type,
                                                    ablate_agent=ablate_agent)
        stacked_frames = env_wrapper.reset(noop_max=max_noop)
        done = False
        steps = 0

        while not done:
            if steps < 4:
                action_distribution = None
                action = env_wrapper.env.action_space.sample()
            else:
                if agent_type == "keras":
                    action_distribution = get_agent_prediction_from_stacked_frames(agent, stacked_frames)
                elif agent_type == "acer":
                    sess = agent.step_model.sess
                    feed_dict = {agent.step_model.X: adjust_shape(agent.step_model.X, stacked_frames)}
                    action_distribution = sess.run(agent.step_model.pi, feed_dict)
                elif agent_type == "olson":
                    action_distribution = get_agent_prediction_from_stacked_frames(agent, stacked_frames)
                else:
                    raise NotImplementedError("Known agent-types are: keras, acer and olson")
                action = np.argmax(np.squeeze(action_distribution))

            stacked_frames, observations, reward, done, info = env_wrapper.step(action, skip_frames=skip_frames)
            cummulative_reward += reward

            if action_distribution is not None:
                if remaining_interval > 0:
                    remaining_interval -= 1

                # compute importance
                importance = np.max(action_distribution) - np.min(action_distribution)

                # check if frame should be added to summary
                if (len(summary_frames) < num_frames or importance > min(summary_importances))\
                        and remaining_interval == 0:
                    frame = _preprocess(env_wrapper.original_stacked_frame[:, :, :, -1],
                                        pacman=not env_wrapper.space_invaders)

                    add = False
                    if len(summary_frames) < num_frames:
                        add = True
                    else:
                        most_similar_frame_idx = _get_most_similar_frame_index(frame, summary_frames)
                        if summary_importances[most_similar_frame_idx] < importance:
                            # remove less important similar frame from summary
                            del summary_frames[most_similar_frame_idx]
                            del summary_importances[most_similar_frame_idx]
                            del summary_actions[most_similar_frame_idx]
                            add = True

                    if add:
                        # add frame to summary
                        summary_frames.append(frame)
                        summary_importances.append(importance)
                        summary_actions.append(action)
                        remaining_interval = interval_size
            steps += 1
            cummulative_steps += 1
            print(f"\rProcessed {runs} runs and {steps} steps. Cummulative rewards {cummulative_reward}."
                  f" Cummulative steps {cummulative_steps}", end="")
        runs += 1

    print()
    _save_summary(summary_frames, summary_importances, summary_actions, save_dir)


def _get_most_similar_frame_index(frame, other_frames):
    # calculating the similarity with the proximity
    similarities = list(map(lambda other_frame: Evaluator.proximity(np.array(frame), other_frame), other_frames))
    return np.argmax(similarities)


def _save_summary(frames, importances, actions, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        raise FileExistsError(f"Save directory '{save_dir}' already exists.")
    for i, frame in enumerate(frames):
        frame.save(os.path.join(save_dir, f"{i}_{int(importances[i])}_action{int(actions[i])}.png"))


def generate_summary_counterfactuals(summary_dir, generator, nb_domains, image_size, save_dir):
    """
    Generates counterfactuals for each target action on every summary state in the given summary directory with StarGAN.

    :param summary_dir: Path to the directory that contains the summary states that counterfactuals should be generated
        for.
    :param generator: A trained StarGAN generator.
    :param nb_domains: Number of actions/domains of the underlying environment.
    :param image_size: Size of summary and counterfactual frames (Quadratic size is assumed).
    :param save_dir: Path to a directory that will be created and filled with counterfactuals.
    :return: None
    """
    # create save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        raise FileExistsError(f"Save directory '{save_dir}' already exists.")

    summary_frame_files = [f for f in listdir(summary_dir) if isfile(os.path.join(summary_dir, f))]
    for file in summary_frame_files:
        frame = Image.open(os.path.join(summary_dir, file))
        for target_domain in range(nb_domains):
            cf, _ = generate_counterfactual(generator, frame, target_domain, nb_domains, image_size)
            cf.save(os.path.join(save_dir, f"CF_FromFile_{file.split('.')[0]}_TargetDomain_{target_domain}.png"))


def generate_olson_summary_counterfactuals(summary_dir, agent, encoder, generator, Q, P, is_pacman, nb_domains,
                                           save_dir, ablate_agent=False, max_iters=5000):
    """
    Generates counterfactuals for each target action on every summary state in the given summary directory with the
    approach of Olson et al.

    :param summary_dir: Path to the directory that contains the summary states that counterfactuals should be generated
        for.
    :param agent: The agent that the explainability approach was trained on.
    :param encoder: The trained encoder.
    :param generator: The trained generator.
    :param Q: The trained encoder Q from the Wasserstein Autoencoder.
    :param P: The trained decoder P from the Wasserstein Autoencoder.
    :param is_pacman: Whether the summary states are from Pac-Man of Space Invaders.
    :param nb_domains: Number of actions/domains of the underlying environment.
    :param save_dir: Path to a directory that will be created and filled with counterfactuals.
    :param max_iters: Maximum amount of iterations for the gradient descent in the agents latent space via the
        Wasserstein Autoencoder.
    :return: None
    """
    # create save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        raise FileExistsError(f"Save directory '{save_dir}' already exists.")

    summary_frame_files = [f for f in listdir(summary_dir) if isfile(os.path.join(summary_dir, f))]
    for file in summary_frame_files:
        frame = Image.open(os.path.join(summary_dir, file))
        for target_domain in range(nb_domains):
            cf, _ = generate_olson_counterfactual(frame, target_domain, agent, encoder, generator, Q, P, is_pacman,
                                                  ablate_agent=ablate_agent, max_iters=max_iters)
            cf.save(os.path.join(save_dir, f"CF_FromFile_{file.split('.')[0]}_TargetDomain_{target_domain}.png"))


def create_cf_summary_collage(cf_summary_dirs, original_frame_prefix, nb_domains,
                              latex_path_prefix="figures/HIGHLIGHTS-DIV"):
    """
    Creates a pandas data frame that contains latex formatted image references. This is used to automatically generate
    figures that show counterfactuals for each target action on a given summary state for the given approaches.

    :param cf_summary_dirs: List of paths to the directories that contain counterfactuals for a summary state generated
        with a certain explainability approach.
    :param original_frame_prefix: A prefix that is unique to the original summary frame (e.g. "CF_FromFile_0).
    :param nb_domains: Number of actions/domains of the underlying environment.
    :param latex_path_prefix: Path prefix to the counterfactual images within the latex document.
    :return: A pandas data frame that can be formatted as a latex table via df.to_latex(escape=False).
    """
    columns = list(map(str, np.arange(nb_domains)))
    total_df = pd.DataFrame(columns=columns)

    for cf_summary_dir in cf_summary_dirs:
        last_sub_dir = cf_summary_dir.split('/')[-1]

        # get relevant file names
        cf_files = [f for f in listdir(cf_summary_dir) if isfile(os.path.join(cf_summary_dir, f))]
        cf_files = filter(lambda file: file.startswith(original_frame_prefix), cf_files)
        # add latex prefix and includegraphics
        cf_files = map(lambda file: latex_path_prefix + "/" + last_sub_dir + "/" + file, cf_files)
        cf_files = map(lambda file: f"\includegraphics[width=\PacManCollageImageSize\\textwidth, "
                                    f"height=\\PacManCollageImageSize\\textwidth]{{{file}}}",
                       cf_files)

        new_row = pd.DataFrame([cf_files], columns=columns, index=[last_sub_dir])
        total_df = total_df.append(new_row)

    return total_df


if __name__ == "__main__":
    restrict_tf_memory()
    GENERATE_NEW_HIGHLIGHTS = True
    OLSON = True
    STARGAN = True

    # Settings
    # summary_dir = "../res/HIGHLIGHTS_DIV/Summaries/PacMan_FearGhost2_3"
    # summary_dir = "../res/HIGHLIGHTS_DIV/Summaries/Pacman_Ingame"
    # summary_dir = "../res/HIGHLIGHTS_DIV/Summaries/Pacman_PowerPill"
    # summary_dir = "../res/HIGHLIGHTS_DIV/Summaries/SpacInvaders_Abl"
    summary_dir = "../res/HIGHLIGHTS_DIV/Summaries/SpacInvaders"

    nb_actions = 6  # 5 for Pacman, 6 for SpaceInvader
    img_size = 160  # 176 for Pacman, 160 for SpaceInvader
    agent_latent = 32  # 512 for ACER, 256 for DQN, 32 for Olson Agents
    is_pacman = False
    cf_summary_dir = "../res/HIGHLIGHTS_DIV/CF_Summaries/SpacInvaders"

    # model_name = "PacMan_FearGhost2_3"
    # model_name = "PacMan_Ingame"
    # model_name = "PacMan_PowerPill"
    # model_name = "SpaceInvaders_Abl"
    model_name = "SpaceInvaders"

    # The Fear Ghost agent uses a pytorch version of the agent for the Olson CF generation
    # but the baselines model for generating HIGHLIGHTS. Thats why we have to differentiate between the agents
    # olson_agent_path = "../res/agents/ACER_PacMan_FearGhost2_cropped_5actions_40M_3.pt"
    # olson_agent_path = "../res/agents/Pacman_Ingame_cropped_5actions_5M.h5"
    # olson_agent_path = "../res/agents/Pacman_PowerPill_cropped_5actions_5M.h5"
    # olson_agent_path = "../res/agents/abl_agent.tar"
    olson_agent_path = "../res/agents/abl_none.tar"

    if GENERATE_NEW_HIGHLIGHTS:
        env_name = "SpaceInvadersNoFrameskip-v4"  # "MsPacmanNoFrameskip-v4"
        # agent_type = "acer"
        # agent_path = r"../res/agents/ACER_PacMan_FearGhost2_cropped_5actions_40M_3"
        # agent_type = "keras"
        # agent_path = r"../res/agents/Pacman_Ingame_cropped_5actions_5M.h5"
        # agent_path = r"../res/agents/Pacman_PowerPill_cropped_5actions_5M.h5"
        agent_type = "olson"
        agent_path = "../res/agents/abl_none.tar"
        num_frames = 5
        interval_size = 50
        num_simulations = 50
        # ablate_agent = False
        ablate_agent = True
        if agent_type == "acer":
            agent = load_baselines_model(agent_path, num_actions=5,
                                     num_env=1)
        if agent_type == "olson":
            agent = olson_model.Agent(6, 32)
            agent.load_state_dict(torch.load(agent_path, map_location=lambda storage, loc: storage))
        elif agent_type == "keras":
            agent = keras.models.load_model(agent_path)


        # Generate a summary that is saved in summary_dir
        generate_highlights_div_summary(env_name, agent, num_frames, num_simulations, interval_size, summary_dir,
                                        agent_type=agent_type, ablate_agent=ablate_agent)

    if STARGAN:
        # Load a StarGAN generator
        generator = Generator(c_dim=nb_actions, channels=3).cuda()
        generator.load_state_dict(torch.load("../res/models/" + model_name + "/models/200000-G.ckpt",
                                             map_location=lambda storage, loc: storage))

        # Generate CFs for that summary which are saved in cf_summary_dir
        generate_summary_counterfactuals(summary_dir, generator, nb_actions, img_size, cf_summary_dir)

    if OLSON:
        # Load all relevant models that are necessary for the CF generation of Olson et al. via load_olson_models()
        olson_agent, olson_encoder, olson_generator, olson_Q, olson_P = load_olson_models(
            olson_agent_path,
            "../res/models/" + model_name + "_Olson/enc39",
            "../res/models/" + model_name + "_Olson/gen39",
            "../res/models/" + model_name + "_Olson_wae/Q",
            "../res/models/" + model_name + "_Olson_wae/P",
            action_size=nb_actions,
            agent_latent=agent_latent,
            pac_man=is_pacman)

        # Generate CFs for that summary which are saved in cf_summary_dir
        generate_olson_summary_counterfactuals(summary_dir, olson_agent, olson_encoder, olson_generator, olson_Q, olson_P,
                                               is_pacman, nb_actions, save_dir=cf_summary_dir + "_Olson")
