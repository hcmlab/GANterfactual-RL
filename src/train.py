from src.star_gan.main import get_parser, main


def train_star_gan(dataset, name, image_size=176, image_channels=3, c_dim=5, batch_size=16, agent_file=None,
                   lambda_counter=1., counter_mode="advantage"):
    """
    Trains StarGAN on the given data set.

    :param dataset: Data set name. The data set is assumed to be saved in res/datasets/.
    :param name: Name under which the StarGAN models are saved. This will create a directory in res/models.
    :param image_size: The size of images within the data set (quadratic images are assumed).
    :param image_channels: Amount of image channels.
    :param c_dim: Amount of domains.
    :param batch_size: Batch size.
    :param agent_file: Path to the agent that should be used for the counterfactual loss. If no agent is given, StarGAN
        will be trained without a counterfactual loss. Pytorch and Keras agents are supported.
    :param lambda_counter: Weight for the counterfactual loss.
    :param counter_mode: Mode of the counterfactual loss. Supported modes are "raw", "softmax", "advantage" and
        "z-score".
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

        "--num_iters=200000",
        "--num_iters_decay=100000",
        "--log_step=100",
        "--sample_step=25000",
        "--model_save_step=200000",
    ]
    if agent_file is not None:
        args.append(f"--agent_path=../res/agents/{agent_file}")

    parser = get_parser()
    config = parser.parse_args(args)
    print(config)
    main(config)


if __name__ == "__main__":
    train_star_gan("PacMan_Ingame_Unique", "PacMan_Ingame_A_1", image_size=176, image_channels=3, c_dim=5,
                   batch_size=16, lambda_counter=1, agent_file="Pacman_Ingame_cropped_5actions_5M.h5",
                   counter_mode="advantage")
