# Generating Counterfactual Explanations for Atari Agents via Generative Adversarial Networks

This repository contains the source code for the master thesis
"Generating Counterfactual Explanations for Atari Agents via Generative Adversarial Networks". This Readme gives an
overview on how to install the requirements and use the code.

Author: Maximilian Demmler

Mentors: Tobias Huber, Silvan Mertes

## Installation

I used an anaconda environment with python 3.7. 
In order to run Pytorch and Tensorflow/Keras on a GPU, install cuDNN 7.6.4 and pytorch with CUDA 10.1:
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install cudnn==7.6.4
```

Then install requirements from requirements.txt:
```
pip install -r requirements.txt
```

The following error might occur due to issues with atari_py:
```
OSError: [WinError 126] The specified module could not be found
```

In order to fix this, uninstall atari_py and install the following version:
```
pip uninstall atari_py
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

## Code Structure & Usage
The following sections describe how my code is organized and provide examples for the usage of core functionalities.

### Data Set Generation

The module [src/dataset_generation.py](../main/src/dataset_generation.py) contains functions for the generation
of an XRL data set. The following code sequence was used to generate the Pac-Man data set for the thesis.
```python
from src.dataset_generation import create_dataset, under_sample, create_unique_dataset, split_dataset
from tensorflow import keras
import numpy as np


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

create_dataset() generates a data set and create_unique_dataset() creates a new duplicate-free version of it. under_sample() under-samples
the data set so that each domain has the same amount of samples and split_dataset() generates a test set with a given
train/test-split (here 10% test samples).


### Training StarGAN

The StarGAN source code (located in [src/star_gan](../main/src/star_gan)) is based on the source code from the
[official StarGAN implementation](https://github.com/yunjey/stargan). I only
extended and modified the code where necessary. StarGAN can either be trained via the console by executing
[src/star_gan/main.py](../main/src/star_gan/main.py) with the parameters described in the module or via the
train_star_gan() function in [src/train.py](../main/src/train.py).

The following code was used to train StarGAN with an advantage-based counterfactual loss on the data set
"PacMan_PowerPill_Unique":
```python
from src.train import train_star_gan


train_star_gan("PacMan_Ingame_Unique", "PacMan_Ingame_A_1", image_size=176, image_channels=3, c_dim=5,
               batch_size=16, lambda_counter=1, agent_file="Pacman_Ingame_cropped_5actions_5M.h5",
               counter_mode="advantage")
```

### Training Olson et al.

The source code for the approach
"Counterfactual State Explanations for Reinforcement Learning Agents via Generative Deep Learning"
by Olson et al. (2021) (located in [src/olson](../main/src/olson)) is based on their published [source code](https://github.com/mattolson93/counterfactual-state-explanations/).
As for StarGAN, I only extended and modified the source code where necessary. The script [src/olson/create_new_agent.py](../main/src/olson/create_new_agent.py) (the script name is a bit misleading)
can be used to train a Wasserstein Autoencoder (WAE) and the script [src/olson/main.py](../main/src/olson/main.py) can be used
to train the encoder, generator and discriminator.


### Generating a Counterfactual Explanation

The functions generate_counterfactual() and generate_olson_counterfactual() (see [src/util.py](../main/src/util.py)) can be used to generate a counterfactual
for a given frame with a trained generator of StarGAN or Olson et al..

At first, load the desired model:
```python
import torch
from PIL import Image
from tensorflow import keras
from src.star_gan.model import Generator
from src.util import load_olson_models, generate_counterfactual, generate_olson_counterfactual

# Settings
pacman = True
nb_actions = 5
env_name = "MsPacmanNoFrameskip-v4"
img_size = 176
agent_file = "../res/agents/Pacman_Ingame_cropped_5actions_5M.h5"
agent = keras.models.load_model(agent_file)

# Load a StarGAN generator
generator = Generator(c_dim=nb_actions, channels=3).cuda()
generator.load_state_dict(torch.load("../res/models/PacMan_Ingame/models/200000-G.ckpt",
                                     map_location=lambda storage, loc: storage))

# Load all relevant models that are necessary for the CF generation of Olson et al. via load_olson_models()
olson_agent, olson_encoder, olson_generator, olson_Q, olson_P = load_olson_models(
    agent_file,
    "../res/models/PacMan_Ingame_Olson/enc39",
    "../res/models/PacMan_Ingame_Olson/gen39",
    "../res/models/PacMan_Ingame_Olson_wae/Q",
    "../res/models/PacMan_Ingame_Olson_wae/P",
    action_size=nb_actions,
    pac_man=pacman)
```

Then generate the counterfactuals for an arbitrary source frame and target action
```python
# Load the original frame and specify the target action
original_frame = Image.open("../res/HIGHLIGHTS_DIV/Summaries/PacMan_Ingame/3_19.png")
target_action = 2  # Corresponds to "RIGHT" on Pac-Man

# Generate a counterfactual with StarGAN
star_gan_cf, star_gan_generation_time = generate_counterfactual(generator, original_frame, target_action,
                                                                nb_actions, image_size=img_size)

# Generate a counterfactual with Olson et al.
olson_cf, olson_generation_time = generate_olson_counterfactual(original_frame, target_action, olson_agent,
                                                                olson_encoder, olson_generator, olson_Q, olson_P,
                                                                pacman)

# Save CF images
star_gan_cf.save("StarGAN_CF.png")
olson_cf.save("Olson_CF.png")
```

### HIGHLIGHT-DIV Summaries

The module [src/summary.py](../main/src/summary.py) contains an implementation of the 
[HIGHLIGHTS-DIV](https://scholar.harvard.edu/files/oamir/files/highlightsmain.pdf) algorithm introduced in
"Highlights: Summarizing agent behavior to people" by Amir et al. (2018), as well as utility functions for the generation of
counterfactuals for summary states.

The following code generates a directory with 5 HIGHLIGHTS-DIV summary states for Pac-Man.
```python
from tensorflow import keras
from src.summary import generate_highlights_div_summary

# Settings
summary_dir = "../res/HIGHLIGHTS_DIV/Summaries/PacMan_Ingame"
cf_summary_dir = "../res/HIGHLIGHTS_DIV/CF_Summaries/PacMan_Ingame"
env_name = "MsPacmanNoFrameskip-v4"
agent = keras.models.load_model("../res/agents/Pacman_Ingame_cropped_5actions_5M.h5")
num_frames = 5
interval_size = 50
num_simulations = 3

# Generate a summary that is saved in cf_summary_dir
generate_highlights_div_summary(env_name, agent, num_frames, num_simulations, interval_size, cf_summary_dir)
```

### Evaluation

The module [src/evaluation.py](../main/src/evaluation.py) contains the class "Evaluator" that can be used for
quantitative evaluations on a test set. Evaluations are performed by generating a counterfactual for each
possible target action on every sample from the test set.


The Evaluator creates two different data structures to save an evaluation:
* A pandas DataFrame that stores the validity, proximity, sparsity and generation time per sample.
* A 3-dimensional confusion matrix cm, where cm\[target_domain, action_on_counterfactual, original_domain\]
corresponds to the amount of counterfactuals that were originally from original_domain, generated for target_domain and
are actually classified as action_on_counterfactual by the agent. The 2-dimensional confusion matrices that are
presented in the thesis are generated by averging the last dimension of such 3-dimensional counfusion matrices.

The first step to use the Evaluator is to load trained model(s) of approaches that should be evaluated. This can be done exactly as shown in
the section "Generating a Counterfactual Explanation". The following code example shows how to evaluate loaded models for
StarGAN and Olson et al. on the dataset "PacMan_Ingame_Unique":
```python
from src.evaluation import Evaluator

# Create the Evaluator
evaluator = Evaluator(agent, "../res/datasets/PacMan_Ingame_Unique/test", "MsPacmanNoFrameskip-v4", img_size=176)

# Evaluate StarGAN
cm, df = evaluator.evaluate_stargan(generator)
evaluator.save_results("../res/results/PacMan_Ingame_StarGAN")

# Evaluate Olson et al.
cm, df = evaluator.evaluate_olson(olson_agent, olson_encoder, olson_generator, olson_Q, olson_P)
evaluator.save_results("../res/results/PacMan_Ingame_Olson")
```
The Evaluator.save_results() method saves the DataFrame as Evaluation.csv, the 3-D confusion matrix as 
Confusion_Matrix_Numpy.npy, a figure of the 2D confusion matrix under Confusion_Matrix.png and examples of
counterfactuals in the subdirectory Examples/.

Evaluator.load_results() can be used to load saved results, Evaluator.print_results() can be used to output a result
summary and Evaluator.get_results_comparison() creates a table that
compares given saved results. I used Evaluator.get_results_comparison().to_latex() to automatically create tables
for my thesis. The following code section provides an example for the usage of these methods:
```python
# Load saved results and print a summary
cm, df = evaluator.load_results("../res/results/PacMan_Ingame_StarGAN")
evaluator.print_results()

# Create a table that compares two or more approaches and convert it to latex
table_df = evaluator.get_results_comparison(["../res/results/PacMan_Ingame_StarGAN", 
                                             "../res/results/PacMan_Ingame_Olson"])
print(table_df.to_latex(float_format="%.3f"))
```

## Resources
The folder [res/agents](../main/res/agents) contains the agents
that were used for my thesis. [Pacman_Ingame_cropped_5actions_5M.h5](../main/res/agents/Pacman_Ingame_cropped_5actions_5M.h5)
is the DQN that was used for Pac-Man,
[Pacman_PowerPill_cropped_5actions_5M.h5](../main/res/agents/Pacman_PowerPill_cropped_5actions_5M.h5) is the DQN that was
used for Pac-Man-PP and [abl_none.tar](../main/res/agents/abl_none.tar) is the A3C that was used for Space Invaders
(copied from the [repository](https://github.com/mattolson93/counterfactual-state-explanations/) of Olson et al.). The folder [res/results](../main/res/results) contains the evaluation results, as
well as figures of dataset statistics. The folder [res/HIGHLIGHTS_DIV](../main/res/HIGHLIGHTS_DIV) contains HIGHLIGHTS_DIV summaries
(in [res/HIGHLIGHTS_DIV/Summaries](../main/res/HIGHLIGHTS_DIV/CF_Summaries), as well as counterfactuals
for these summary states. I used folder res/datasets and res/models to save generated datasets and trained CE approaches. However, these
folders are not added to the repository due to the high storage usage.