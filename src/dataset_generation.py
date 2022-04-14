import os
import random
import shutil

import numpy as np
import torch
from PIL import Image, ImageOps
from tensorflow import keras

from src.atari_wrapper import AtariWrapper
from src.dataset_evaluation import get_uniques
from src.util import array_to_pil_format, load_baselines_model


def create_dataset(env_name, size, target_path, agent, agent_type="deepq", seed=None, noop_range=(0, 30),
                   epsilon=0.0, power_pill_objective=False, domains=None, deepq_preprocessing = True):
    """
    Creates a data set with the following structure. A directory is created at target_path with the subdirectories
    'train' and 'test'. Each of these has a subdirectory for every domain. The domain directories contain the generated
    sample images. This function puts all samples into the 'train' directory. Use split_dataset() to split the data set.

    :param env_name: Name of the gym environment to generate a data set for.
    :param size: The total amount of samples that should be generated.
    :param target_path: The path at which the data set should be saved.
    :param agent: The agent that should be used to classify samples.
    :param agent: The type of agent. "deepq" for Keras DeepQ or "acer" for gym.baselines ACER.
        For Space Invader, a pytorch agent is expected and this flag is not used.  (Keras DeepQ, Pytorch and Baselines ACER are supported).
    :param seed: Seed for random number generator.
    :param noop_range: Range (min, max) of NOOPs that are executed at the beginning of an episode to generate a random
        offset.
    :param epsilon: epsilon value in range [0, 1] for the epsilon-greedy policy that is used to reach more diverse
        states.
    :param power_pill_objective: Whether or not to use the power pill objective on Pac-Man.
    :param domains: List of domain names. If None, the amount of domains will automatically be determined by the given
        gym environment and the names will be 0, 1, 2...
    :return: None
    """
    # set seed
    if seed is not None:
        np.random.seed(seed)

    # init environment
    wrapper = AtariWrapper(env_name, power_pill_objective=power_pill_objective, deepq_preprocessing=deepq_preprocessing)

    # create domains and folder structure
    if domains is None:
        domains = list(map(str, np.arange(wrapper.env.action_space.n)))
    train_path, test_path, success = _setup_dicts(target_path, domains)
    if not success:
        raise FileExistsError(f"Target directory '{target_path}' already exists")

    # generate train and test set
    print("Generating dataset...")
    _generate_set(size, train_path, wrapper, agent, agent_type, domains, noop_range, epsilon)


def split_dataset(dataset_path, test_portion, domains):
    """
    Splits an existing data set into a train and a test set by randomly selecting samples for the test set.

    :param dataset_path: Path to the data set directory. All samples have to be in 'train'.
    :param test_portion: Proportion of the test set in range [0, 1] (e.g. 0.1 for 10% test samples).
    :param domains: List of domain names that for domains that should be effected by the split.
    :return: None
    """
    print("Splitting dataset...")
    for domain in domains:
        # get absolute split size per domain
        domain_path = os.path.join(dataset_path, "train", domain)
        sample_names = [name for name in os.listdir(domain_path) if os.path.isfile(os.path.join(domain_path, name))]
        size = len(sample_names)
        test_size = int(size * test_portion)

        test_indices = random.sample(range(size), test_size)
        for test_idx in test_indices:
            # copy sample from train to test
            train_file_path = os.path.join(domain_path, sample_names[test_idx])
            test_file_path = os.path.join(dataset_path, "test", domain, sample_names[test_idx])
            shutil.move(train_file_path, test_file_path)


def under_sample(dataset_path, min_size=None):
    """
    Under-samples the given data set by removing randomly chosen samples from domains that contain more samples until
    each domain contains an equal amount of samples or min_size is reached.

    :param dataset_path: Path to the data set directory.
    :param min_size: The minimum amount of samples that domains with more samples should keep. If None, domains will
        be under-sampled to the size of the smallest domain.
    :return: None
    """
    def _get_sample_names_and_size(domain_path_name):
        domain_sample_names = [name for name in os.listdir(domain_path_name) if
                               os.path.isfile(os.path.join(domain_path_name, name))]
        domain_size = len(domain_sample_names)
        return domain_sample_names, domain_size

    print("Down-sampling dataset...")
    for subset in ["train", "test"]:
        # get domain sizes
        domain_sizes = []
        for domain in domains:
            domain_path = os.path.join(dataset_path, subset, domain)
            sample_names, size = _get_sample_names_and_size(domain_path)
            domain_sizes.append(size)

        # get minimum size (used for down sampling)
        new_size = int(min(domain_sizes))
        if min_size is not None:
            new_size = int(max(new_size, min_size))

        # remove files to get an equal distribution
        for domain in domains:
            domain_path = os.path.join(dataset_path, subset, domain)
            sample_names, size = _get_sample_names_and_size(domain_path)
            # randomly choose (size - new_size) samples to remove
            indices_to_remove = random.sample(range(size), int(max(size - new_size, 0)))
            for idx_to_remove in indices_to_remove:
                file_to_remove = os.path.join(domain_path, sample_names[idx_to_remove])
                os.remove(file_to_remove)


def create_unique_dataset(new_dataset_path, old_dataset_path):
    """
    Creates a data set without duplicate samples on the basis of a given data set.

    :param new_dataset_path: Path for the newly created unique data set.
    :param old_dataset_path: Path to an existing data set that possibly contains duplicate samples.
    :return: None
    """
    print("Creating a dataset with unique samples...")
    domains = []
    for item in os.listdir(os.path.join(old_dataset_path, "train")):
        path = os.path.join(old_dataset_path, "train", item)
        if os.path.isdir(path):
            domains.append(item)

    train_path, test_path, success = _setup_dicts(new_dataset_path, domains)
    if not success:
        print(f"Target directory '{new_dataset_path}' already exists")
        return

    for domain in domains:
        _save_unique_samples(train_path, old_dataset_path, domain)
        _save_unique_samples(test_path, old_dataset_path, domain)
        print(f"Finished domain {domain}.")


def create_clean_test_set(dataset_path, samples_per_domain):
    """
    Creates a clean test set for a possibly dirty data set. The clean test set is generated by selecting random samples
    from the train set as test samples. Duplicates of the selected test samples within the train set are removed from the train set.

    :param dataset_path: Path to the possibly dirty data set. It is assumed that the existing data set only contains all
        samples in the train set.
    :param samples_per_domain: Amount of test samples that should be selected per domain.
    :return: None
    """
    print("Creating a clean test set...")
    for domain in os.listdir(os.path.join(dataset_path, "train")):
        domain_path = os.path.join(dataset_path, "train", domain)
        if os.path.isdir(domain_path):
            domain_file_names = []
            for i, item in enumerate(os.listdir(domain_path)):
                # get file name
                file_name = os.path.join(domain_path, item)
                domain_file_names.append(file_name)

            random_indices = random.sample(range(len(domain_file_names)), len(domain_file_names))
            tabu_list = []
            collected_samples = 0

            for i in random_indices:
                if i in tabu_list:
                    continue
                # open image sample
                img = Image.open(domain_file_names[i])
                sample = np.array(img)

                # delete from train and copy to test
                os.remove(domain_file_names[i])
                img.save(os.path.join(dataset_path, "test", domain, f"{i}.png"))

                # add to tabu
                tabu_list.append(i)
                collected_samples += 1

                # search for duplicates and delete and tabu them
                for j, other_file in enumerate(domain_file_names):
                    if j in tabu_list:
                        continue
                    other_sample = np.array(Image.open(domain_file_names[j]))
                    if (other_sample == sample).all():
                        tabu_list.append(j)
                        os.remove(other_file)

                if collected_samples >= samples_per_domain:
                    # finished
                    break
        print(f"Finished Domain {domain}.")


def _generate_set(size, path, env_wrapper, agent, agent_type, domains, noop_range, epsilon):
    stacked_frames = env_wrapper.reset(noop_min=noop_range[0], noop_max=noop_range[1])
    # not using a for loop because the step counter should only be increased
    # if the frame is actually saved as a training sample
    step = 0
    init_steps = 4

    if env_wrapper.space_invaders:
        skip_frames = 7
    else:
        skip_frames = 4

    while step < size + init_steps:
        if step < init_steps:
            action = env_wrapper.env.action_space.sample()
            step += 1
        else:
            stacked_frames = np.expand_dims(stacked_frames, axis=0)

            if np.random.uniform() < epsilon:
                # random exploration to increase state diversity (frame must not be saved as a training sample!)
                action = env_wrapper.env.action_space.sample()
            else:
                # Optimally this should be if agent_type == "torch" but I am afraid to break something else
                if env_wrapper.space_invaders:
                    torch_state = torch.Tensor(stacked_frames).cuda()
                    output = agent.pi(agent(torch_state)).detach().cpu().numpy()
                else:
                    if agent_type == "deepq":
                        output = agent.predict(stacked_frames)
                    elif agent_type == "acer":
                        output, _, _, _ = agent.step(stacked_frames)
                    if len(output) == 2:
                        output = output[0]
                if agent_type == "acer":
                    # The ACER ouput contains the action directly.
                    action = output[0]
                else:
                    action = int(np.argmax(np.squeeze(output)))

                # save frame(s) in the domain of the action that the agent chose
                file_name = os.path.join(path, domains[action], f"{step}")
                if len(observations) == skip_frames:
                    # only saves the latest frame
                    frame = _preprocess(env_wrapper.original_stacked_frame[:, :, :, -1],
                                        pacman=not env_wrapper.space_invaders)
                    _save_image(frame, file_name + ".png")
                    step += 1

        stacked_frames, observations, reward, done, info = env_wrapper.step(action, skip_frames=skip_frames)
        print(f"\rFinished {int((step + 1) / (size + init_steps) * 100)}%", end="")

    print()

    env_wrapper.env.close()


def _save_unique_samples(set_path, old_dataset_path, domain):
    unique_samples, _, _ = get_uniques(os.path.join(old_dataset_path, set_path.split("\\")[-1], domain))
    for i, unique_sample in enumerate(unique_samples):
        _save_image(unique_sample, os.path.join(set_path, domain, f"{i}.png"))


def _setup_dicts(target_path, domains):
    try:
        # creating the directory structure
        os.mkdir(target_path)
        # train dir
        train_path = os.path.join(target_path, "train")
        os.mkdir(train_path)
        for domain in domains:
            os.mkdir(os.path.join(train_path, domain))
        # test dir
        test_path = os.path.join(target_path, "test")
        os.mkdir(test_path)
        for domain in domains:
            os.mkdir(os.path.join(test_path, domain))
        return train_path, test_path, True
    except FileExistsError:
        return None, None, False


def _preprocess(observation, pacman=False):
    observation = array_to_pil_format(np.array(observation))
    if pacman:
        observation = observation[0:173, :, :]
    else:
        observation = observation[35:195, :, :]
    img = Image.fromarray(observation, "RGB")
    if pacman:
        # squares the 173x160 image to 176x176
        img = ImageOps.expand(img, (8, 1, 8, 2))
    return img


def _save_image(frame, file_name):
    frame.save(file_name)


if __name__ == "__main__":
    # Settings
    env_name = "MsPacmanNoFrameskip-v4"
    # agent = keras.models.load_model("../res/agents/PacMan_Ingame_cropped_5actions_5M.h5")
    agent = load_baselines_model(r"../res/agents/ACER_PacMan_FearGhost_cropped_5actions_40M", num_actions=5, num_env=1)
    agent_type = "acer"
    nb_domains = 5
    nb_samples = 400000
    dataset_path = "../res/datasets/PacMan_FearGhost_cropped_5actions"
    unique_dataset_path = dataset_path + "_Unique"
    domains = list(map(str, np.arange(nb_domains)))

    deepq_preprocessing = True
    if agent_type == "acer":
        deepq_preprocessing = False

    # Data set generation
    create_dataset(env_name, nb_samples, dataset_path, agent, agent_type=agent_type, seed=42, epsilon=0.2,
                   domains=domains, deepq_preprocessing = deepq_preprocessing)
    # Additional down-sampling to reduce memory cost for removing duplicates.
    # In the end, this should in most cases not minder the amount of total samples, since min_size is set.
    under_sample(dataset_path, min_size=nb_samples / nb_domains)
    create_unique_dataset(unique_dataset_path, dataset_path)
    under_sample(unique_dataset_path)
    split_dataset(unique_dataset_path, 0.1, domains)
