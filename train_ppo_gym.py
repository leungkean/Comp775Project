import json
import os

import click
import gym
import tensorflow as tf
import wandb

from afa.data import load_pet_as_numpy
import afa.environments
from afa.environments.dataset_manager import EnvironmentDatasetManager
from afa.networks.segment.unet3 import UNet

from gym.envs.registration import register

import sys
sys.path.insert(1, '../')

from afa.agents.base import WandbCallback
from afa.agents.ppo import PPOAgent

for device in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except:
        pass


@click.command()
@click.option(
    "--num_iterations",
    type=click.INT,
    default=100,
    help="The number of training iterations.",
)
@click.option(
    "--num_eval_episodes",
    type=click.INT,
    default=20,
    help="The number of episodes to roll out during the evaluation stage of "
    "each iteration.",
)
@click.option(
    "--total_rollouts_length",
    type=click.INT,
    default=1024,
    help="The total number of timesteps of experience that are collected at "
    "each iteration.",
)
@click.option(
    "--num_workers",
    type=click.INT,
    default=16,
    help="The number of parallel workers used for collecting experience.",
)
@click.option(
    "--num_sgd_epochs",
    type=click.INT,
    default=16,
    help="The number of SGD epochs to perform at each iteration.",
)
@click.option(
    "--minibatch_size",
    type=click.INT,
    default=64,
    help="The minibatch size for each SGD step.",
)
@click.option(
    "--learning_rate", type=click.FLOAT, default=5e-5, help="The learning rate."
)
@click.option(
    "--entropy_coef",
    type=click.FLOAT,
    default=0.0,
    help="The coefficient of the entropy regularizer.",
)
@click.option(
    "--vf_coef",
    type=click.FLOAT,
    default=1.0,
    help="The coefficient of the value function loss.",
)
@click.option(
    "--clip_range", type=click.FLOAT, default=0.2, help="The PPO clip parameter."
)
@click.option(
    "--vf_clip",
    type=click.FLOAT,
    default=10.0,
    help="The value function clip parameter.",
)
@click.option(
    "--max_grad_norm", type=click.FLOAT, help="The gradient clipping threshold."
)
@click.option("--gamma", type=click.FLOAT, default=0.99, help="The discount factor.")
@click.option(
    "--lambda_", type=click.FLOAT, default=0.95, help="The GAE lambda parameter."
)
@click.option(
    "--hidden_units",
    type=click.STRING,
    help="A string of comma-separate integers, where the ith integer represented "
    "the number of hidden units in the ith hidden layer. If a CNN model is being used,"
    "then this defines the post-convolutional layers in the model.",
)
@click.option(
    "--conv_layers",
    type=click.STRING,
    help="A string representation of a Python list of tuples, where the ith tuple "
    "specifies the (filters, kernel, stride) in the ith hidden layer. This is relevant "
    "only if a CNN model is being used.",
)
@click.option(
    "--activation", type=click.STRING, default="relu", help="The activation function."
)
@click.option(
    "--value_network",
    type=click.Choice(["copy", "shared"]),
    default="copy",
    help="Whether the value network should be a copy or shared with the "
    "policy network.",
)
@click.option(
    "--offline", is_flag=True, help="Whether or not to run W&B in offline mode."
)
def main(
    num_iterations,
    num_eval_episodes,
    total_rollouts_length,
    num_workers,
    num_sgd_epochs,
    minibatch_size,
    learning_rate,
    entropy_coef,
    vf_coef,
    clip_range,
    vf_clip,
    max_grad_norm,
    gamma,
    lambda_,
    hidden_units,
    conv_layers,
    activation,
    value_network,
    offline,
):
    """Trains a PPO agent in a standard Gym environment."""
    config = locals()

    run = wandb.init(
        project="active-acquisition",
        job_type="train_ppo_gym",
        config=config,
        mode="disabled" if offline else "online",
    )

    features, targets = load_pet_as_numpy('train')
    dataset_manager = EnvironmentDatasetManager.remote(features, targets)

    unet_artifact = run.use_artifact("oxford_pet_unet_classifier:v14")
    unet_artifact_dir = os.path.abspath(unet_artifact.download()) 

    env_config = {
            "dataset_manager": dataset_manager,
            "model_dir": unet_artifact_dir,
            "index_dims": 2,
            "acquisition_cost": 1e-4,
    }

    def make_env(config):
        return gym.make('PretrainedUNetEnv-v0', env_config=env_config)

    model_config = {"value_network": value_network}

    if hidden_units is not None:
        model_config["hidden_units"] = tuple(map(int, hidden_units.split(",")))

    if conv_layers is not None:
        model_config["conv_layers"] = eval(conv_layers)

    if activation is not None:
        model_config["activation"] = activation

    agent_config = {
        "model_config": model_config,
        "num_workers": num_workers,
        "total_rollouts_length": total_rollouts_length,
        "num_sgd_epochs": num_sgd_epochs,
        "minibatch_size": minibatch_size,
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "vf_coef": vf_coef,
        "clip_range": clip_range,
        "vf_clip": vf_clip,
        "max_grad_norm": max_grad_norm,
        "gamma": gamma,
        "lambda": lambda_,
    }

    with open(os.path.join(run.dir, "agent_config.json"), "w") as fp:
        json.dump(agent_config, fp)

    agent = PPOAgent(agent_config, make_env)

    try:
        agent.train(
            num_iterations,
            run.dir,
            num_eval_episodes=num_eval_episodes,
            callbacks=[WandbCallback(run)],
        )
    except KeyboardInterrupt:
        print(
            "Training interrupted. Attempting to save agent artifact "
            "using current best weights."
        )

    agent_artifact = wandb.Artifact(f"unet_pet_ppo_agent", type="agent")
    agent_artifact.add_file(os.path.join(run.dir, "best_weights.pkl"))
    agent_artifact.add_file(os.path.join(run.dir, "agent_config.json"))
    run.log_artifact(agent_artifact)


if __name__ == "__main__":
    main()
