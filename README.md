# GANterfactual-RL: Understanding Reinforcement Learning Agents' Strategies through Visual Counterfactual Explanations

This repository contains the official source code for the AAMAS 2023 paper [GANterfactual-RL: Understanding Reinforcement Learning Agents' Strategies through Visual Counterfactual Explanations](https://arxiv.org/abs/2302.12689).
This Readme gives an overview on how to install the requirements and use the code.

## Installation

We used python 3.7 with cuDNN 7.6.4 and CUDA 10.1.

First, install requirements from requirements.txt:
```
pip install -r requirements.txt
```
Then you can install the pytorch version we used as follows:
```
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 --index-url https://download.pytorch.org/whl/cu101
```

To properly install baselines you have to use their git repository, not pip. We used the commit *ea25b9e8b234e6ee1bca43083f8f3cf974143998*
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

Finally, we only tested on windows 10. To get atari-py to run on windows, we used the following repository with release 1.2.2:
```
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

## Code Structure & Usage
The following sections describe how the code is organized and provide examples for the usage of core functionalities.

### Data Set Generation

The module [src/dataset_generation.py](../main/src/dataset_generation.py) contains functions for the generation
of data sets to train our Counterfactual Explanation models. create_dataset() generates a data set and create_unique_dataset() creates a new duplicate-free version of it. under_sample() under-samples the data set so that each domain has the same amount of samples and split_dataset() generates a test set with a given train/test-split.

As an example, the following settings of [src/dataset_generation.py](../main/src/dataset_generation.py) were used to generate the dataset for the Blue Ghost Pac-Man agent in our paper:
```python
if __name__ == "__main__":
  # Settings
  env_name = "MsPacmanNoFrameskip-v4"
  agent = keras.models.load_model("../res/agents/PacMan_Ingame_cropped_5actions_5M.h5")
  nb_domains = 5
  nb_samples = 400000
  dataset_path = "../res/datasets/PacMan_Ingame"
  unique_dataset_path = dataset_path + "_Unique"
  domains = list(map(str, np.arange(nb_domains)))

  # Data set generation
  create_dataset(env_name, nb_samples, dataset_path, agent, seed=42, epsilon=0.2, domains=domains)
  # Additional down-sampling to reduce memory cost for removing duplicates.
  # In the end, this should in most cases not minder the amount of total samples, since min_size is set.
  under_sample(dataset_path, min_size=nb_samples / nb_domains)
  create_unique_dataset(unique_dataset_path, dataset_path)
  under_sample(unique_dataset_path)
  split_dataset(unique_dataset_path, 0.1, domains)
```


### Training the GANterfactual-RL models

The StarGAN source code (located in [src/star_gan](../main/src/star_gan)) within our GANterfactul-RL implementation is based on the source code from the
[official StarGAN implementation](https://github.com/yunjey/stargan). We only extended and modified the code where necessary. StarGAN can either be trained via the console by executing [src/star_gan/main.py](../main/src/star_gan/main.py) with the parameters described in the module or via the
train_star_gan() function in [src/train.py](../main/src/train.py).

For example, the following settings of [src/train.py](../main/src/train.py) were used to train StarGAN for the Blue Ghost Pacman agent in our paper:
```python
if __name__ == "__main__":
  train_star_gan("PacMan_Ingame_Unique", "PacMan_Ingame", image_size=176, image_channels=3, c_dim=5,
                 batch_size=16, agent_file= None)
```

### Training the CSE models

The source code for the approach
"Counterfactual State Explanations for Reinforcement Learning Agents via Generative Deep Learning"
by Olson et al. (2021) (located in [src/olson](../main/src/olson)) is based on their published [source code](https://github.com/mattolson93/counterfactual-state-explanations/).
As for StarGAN, we only extended and modified the source code where necessary. The script [src/olson/create_new_agent.py](../main/src/olson/create_new_agent.py) (the script name is a bit misleading)
can be used to train a Wasserstein Autoencoder (WAE) and the script [src/olson/main.py](../main/src/olson/main.py) can be used
to train the encoder, generator and discriminator.


### HIGHLIGHT-DIV Summaries wit Counterfactuals

The module [src/summary.py](../main/src/summary.py) contains an implementation of the 
[HIGHLIGHTS-DIV](https://scholar.harvard.edu/files/oamir/files/highlightsmain.pdf) algorithm introduced in
"Highlights: Summarizing agent behavior to people" by Amir et al. (2018), as well as utility functions for the generation of
counterfactuals for summary states.

For example, the following settings [src/summary.py](../main/src/summary.py) generates a directory with 5 HIGHLIGHTS-DIV summary states for the Blue Ghost Pacman agent and generates Counterfactual explanations for all those states.
```python

if __name__ == "__main__":
    restrict_tf_memory()
    GENERATE_NEW_HIGHLIGHTS = True
    OLSON = True
    STARGAN = True

    # Settings
    summary_dir = "../res/HIGHLIGHTS_DIV/Summaries/Pacman_Ingame"

    nb_actions = 5  # 5 for Pacman, 6 for SpaceInvader
    img_size = 176  # 176 for Pacman, 160 for SpaceInvader
    agent_latent = 256  # 512 for ACER, 256 for DQN, 32 for Olson Agents
    is_pacman = True
    cf_summary_dir = "../res/HIGHLIGHTS_DIV/CF_Summaries/SpacInvaders"

    model_name = "PacMan_Ingame"

    # The Fear Ghost agent uses a pytorch version of the agent for the Olson CF generation
    # but the baselines model for generating HIGHLIGHTS. Thats why we have to differentiate between the agents
    olson_agent_path = "../res/agents/Pacman_Ingame_cropped_5actions_5M.h5"

    if GENERATE_NEW_HIGHLIGHTS:
        env_name = "MsPacmanNoFrameskip-v4"
        agent_type = "keras"
        agent_path = r"../res/agents/Pacman_Ingame_cropped_5actions_5M.h5"
        num_frames = 5
        interval_size = 50
        num_simulations = 50
        ablate_agent = False
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
```

### Evaluation

The module [src/evaluation.py](../main/src/evaluation.py) contains the class "Evaluator" that can be used for
quantitative evaluations on a test set. Evaluations are performed by generating a counterfactual for each
possible target action on every sample from the test set.

For example, the following settings for [src/evaluation.py](../main/src/evaluation.py) can be used to evalute the CSE counterfactual model for the Fear Ghost Pacman agent in our paper:
```python
if __name__ == "__main__":
    restrict_tf_memory()
    GENERATE_NEW_RESULTS = True

    if GENERATE_NEW_RESULTS:
        # Settings
        ## Pacman
        pacman = True
        nb_actions = 5
        env_name = "MsPacmanNoFrameskip-v4"
        img_size = 176
        agent_file = "../res/agents/ACER_PacMan_FearGhost2_cropped_5actions_40M_3"
        agent_type = "acer"
        model_type = "olson"
        ablate_agent = False
        agent_latent = 512
        if agent_type == "deepq":
            agent = keras.models.load_model(agent_file)
        elif agent_type == "acer":
            agent = load_baselines_model(agent_file, num_actions=5, num_env=1)
        elif agent_type == "olson":
            # Loads a torch model with the specific architecture that Olson et al. used
            agent = olson_model.Agent(6, 32).cuda()
            agent.load_state_dict(torch.load(agent_file, map_location=lambda storage, loc: storage))
        elif agent_type == "torch_acer":
            # diry numbers for 5 actions for pacman and latent size 512
            agent = olson_model.ACER_Agent(num_actions=5, latent_size=512).cuda()
            agent.load_state_dict(torch.load(agent_file))
        elif agent_type == "torch":
            # TODO
            raise NotImplementedError("not yet implemented")

        # Create the Evaluator
        evaluator = Evaluator(agent, "../res/datasets/ACER_PacMan_FearGhost2_cropped_5actions_40M_3_Unique/test", env_name,
                              img_size=img_size, agent_type=agent_type, ablate_agent=ablate_agent)

        if model_type == "stargan":
            # Load a StarGAN generator
            generator = Generator(c_dim=nb_actions, channels=3).cuda()
            generator.load_state_dict(torch.load("../res/models/SpaceInvaders_Abl/models/200000-G.ckpt",
                                                 map_location=lambda storage, loc: storage))

            # Evaluate StarGAN
            cm, df = evaluator.evaluate_stargan(generator)
            evaluator.save_results("../res/results/Space_Invaders_Abl")

        if model_type == "olson":
            # Load all relevant models that are necessary for the CF generation of Olson et al. via load_olson_models()
            olson_agent, olson_encoder, olson_generator, olson_Q, olson_P = load_olson_models(
                "../res/agents/ACER_PacMan_FearGhost2_cropped_5actions_40M_3.pt",
                "../res/models/PacMan_FearGhost2_3_Olson/enc39",
                "../res/models/PacMan_FearGhost2_3_Olson/gen39",
                "../res/models/PacMan_FearGhost2_3_Olson_wae/Q",
                "../res/models/PacMan_FearGhost2_3_Olson_wae/P",
                action_size=nb_actions,
                agent_latent=agent_latent,
                pac_man=pacman)

    # To reload old evaluation results
    else:
        pd.set_option('display.max_columns', None)
        results = Evaluator.get_results_comparison(["../res/results/SpaceInvaders_Abl", "../res/results/SpaceInvaders_Abl_Olson"])
        print(results)
```



## Resources
The folder [res/agents](../main/res/agents) contains the agents that were used for our paper. 
The Pacman agents were trained with our [fork](https://github.com/hcmlab/baselines/tree/customTraining) of the openAI baselines repository.
The Space Invaders agents were copied from the [repository](https://github.com/mattolson93/counterfactual-state-explanations/) of Olson et al..

The folder [res/results](../main/res/results) contains the evaluation results, as well as figures of dataset statistics. 

The folder [res/HIGHLIGHTS_DIV](../main/res/HIGHLIGHTS_DIV) contains HIGHLIGHTS_DIV summaries
(in [res/HIGHLIGHTS_DIV/Summaries](../main/res/HIGHLIGHTS_DIV/Summaries)), as well as counterfactuals
for these summary states (in [res/HIGHLIGHTS_DIV/CF_Summaries](../main/res/HIGHLIGHTS_DIV/CF_Summaries)). 

We used the folders res/datasets and res/models to save generated datasets and trained Counterfactual Explanation approaches. However, these
folders are not added to the repository due to the high storage usage. They are available on request.
