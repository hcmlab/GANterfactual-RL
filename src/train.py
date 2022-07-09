from src.star_gan.main import get_parser, main


def train_star_gan(dataset, name, image_size=176, image_channels=3, c_dim=5, batch_size=16, agent_file=None,
                   lambda_counter=1., counter_mode="advantage", agent_type="deepq", ablate_agent=False):
    """
    Trains StarGAN on the given data set.

    :param dataset: Data set name. The data set is assumed to be saved in res/datasets/.
    :param name: Name under which the StarGAN models are saved. This will create a directory in res/models.
    :param image_size: The size of images within the data set (quadratic images are assumed).
    :param image_channels: Amount of image channels.
    :param c_dim: Amount of domains.
    :param batch_size: Batch size.
    :param agent_file: Path to the agent that should be used for the counterfactual loss. If no agent is given, StarGAN
        will be trained without a counterfactual loss.
    :param agent_type: The type of agent. "deepq" for Keras Deep-Q, "acer" for gym baselines ACER and "olson" for a
        Pytorch Actor Critic Space Invaders model.
    :param lambda_counter: Weight for the counterfactual loss.
    :param counter_mode: Mode of the counterfactual loss. Supported modes are "raw", "softmax", "advantage" and
        "z-score".
    :param ablate_agent: Whether the laser canon should be removed from space invaders frames before they are input to
        the agent.
    :return: None
    """
    args = [
        "--mode=train",
        "--dataset=RaFD",
        f"--rafd_crop_size={image_size}",
        f"--image_size={image_size}",
        f"--image_channels={image_channels}",
        f"--c_dim={c_dim}",
        f"--batch_size={batch_size}",
        f"--rafd_image_dir=../res/datasets/{dataset}/train",
        f"--sample_dir=../res/models/{name}/samples",
        f"--log_dir=../res/models/{name}/logs",
        f"--model_save_dir=../res/models/{name}/models",
        f"--result_dir=../res/models/{name}/results",
        f"--lambda_counter={lambda_counter}",
        f"--counter_mode={counter_mode}",
        f"--agent_type={agent_type}",
        f"--ablate_agent={ablate_agent}",

        "--num_iters=200000",
        "--num_iters_decay=100000",
        "--log_step=100",
        "--sample_step=25000",
        "--model_save_step=200000",
        "--use_tensorboard=False",
    ]
    if agent_file is not None:
        args.append(f"--agent_path=../res/agents/{agent_file}")

    parser = get_parser()
    config = parser.parse_args(args)
    print(config)
    main(config)


if __name__ == "__main__":
    train_star_gan("SpaceInvaders_Abl", "SpaceInvaders_Abl", image_size=160, image_channels=3, c_dim=6,
                   batch_size=16, lambda_counter=1, agent_file="abl_agent.tar",
                   counter_mode="raw", agent_type="olson", ablate_agent=True)
