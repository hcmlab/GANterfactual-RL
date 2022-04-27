import os
from os import listdir
from os.path import isfile

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
import torch

from src.dataset_generation import _preprocess
from src.evaluation import Evaluator
from src.util import init_environment, get_agent_prediction_from_stacked_frames, generate_counterfactual, \
    restrict_tf_memory, generate_olson_counterfactual, load_baselines_model

from baselines.common.tf_util import adjust_shape
from src.star_gan.model import Generator

def generate_highlights_div_summary(env_name, agent, num_frames, num_simulations, interval_size, save_dir,
                                    power_pill_objective=False, max_noop=1, agent_type = "keras"):
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
    :param agent_type: the type of agent. Accepts "keras" for keras models and "acer" for acer baseline models.
    :return: None
    """
    # init
    runs = 0
    summary_importances = []
    summary_frames = []
    remaining_interval = 0

    while runs < num_simulations:
        # init environment
        env_wrapper, skip_frames = init_environment(env_name, power_pill_objective, agent_type)
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
                else:
                    raise NotImplementedError("Known agent-types are: keras and acer")
                action = np.argmax(np.squeeze(action_distribution))

            stacked_frames, observations, reward, done, info = env_wrapper.step(action, skip_frames=skip_frames)

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
                            add = True

                    if add:
                        # add frame to summary
                        summary_frames.append(frame)
                        summary_importances.append(importance)
                        remaining_interval = interval_size
            steps += 1
            print(f"\rProcessed {runs} runs and {steps} steps", end="")
        runs += 1

    print()
    _save_summary(summary_frames, summary_importances, save_dir)


def _get_most_similar_frame_index(frame, other_frames):
    # calculating the similarity with the proximity
    similarities = list(map(lambda other_frame: Evaluator.proximity(np.array(frame), other_frame), other_frames))
    return np.argmax(similarities)


def _save_summary(frames, importances, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        raise FileExistsError(f"Save directory '{save_dir}' already exists.")
    for i, frame in enumerate(frames):
        frame.save(os.path.join(save_dir, f"{i}_{int(importances[i])}.png"))


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
                                           save_dir, max_iters=5000):
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
                                                  max_iters=max_iters)
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

    # Settings
    summary_dir = "../res/HIGHLIGHTS_DIV/Summaries/PacMan_FearGhost"
    env_name = "MsPacmanNoFrameskip-v4"
    # agent = keras.models.load_model("../res/agents/Pacman_Ingame_cropped_5actions_5M.h5")
    agent = load_baselines_model(r"../res/agents/ACER_PacMan_FearGhost_cropped_5actions_40M", num_actions=5, num_env=1)
    agent_type = "acer"
    num_frames = 5
    interval_size = 50
    num_simulations = 3

    nb_actions = 5
    img_size = 176
    # Load a StarGAN generator
    generator = Generator(c_dim=nb_actions, channels=3).cuda()
    generator.load_state_dict(torch.load("../res/models/PacMan_FearGhost_A_1/models/200000-G.ckpt",
                                         map_location=lambda storage, loc: storage))
    cf_summary_dir = "../res/HIGHLIGHTS_DIV/CF_Summaries/PacMan_FearGhost_A_1"

    # Generate a summary that is saved in summary_dir
    generate_highlights_div_summary(env_name, agent, num_frames, num_simulations, interval_size, summary_dir,
                                    agent_type="acer")

    # Generate CFs for that summary which are saved in cf_summary_dir
    generate_summary_counterfactuals(summary_dir, generator, nb_actions, img_size, cf_summary_dir)