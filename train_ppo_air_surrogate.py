import json
import os

import click
import tensorflow as tf
import wandb

import sys
sys.path.insert(1, '../')

from afa.agents.base import WandbCallback
from afa.agents.ppo import PPOAgent
from afa.environments.utils import create_surrogate_air_env_fn

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass

tf.config.set_visible_devices(gpus[:1], device_type="GPU")


@click.command()
@click.option(
    "--dataset", type=click.STRING, required=True, help="The dataset to train on."
)
@click.option(
    "--surrogate_artifact",
    type=click.STRING,
    required=True,
    help="The artifact name for the surrogate model to use.",
)
@click.option(
    "--num_iterations",
    type=click.INT,
    default=100,
    help="The number of training iterations.",
)
@click.option(
    "--num_eval_episodes",
    type=click.INT,
    help="The number of episodes to roll out during the evaluation stage of "
    "each iteration.",
)
@click.option(
    "--total_rollouts_length",
    type=click.INT,
    default=512,
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
    "--acquisition_cost",
    type=click.FLOAT,
    default=0.1,
    help="The cost of acquiring each feature.",
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
    "--learning_rate", type=click.FLOAT, default=5e-4, help="The learning rate."
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
    "--remote_surrogate",
    type=click.BOOL,
    default=False,
    help="Whether or not the surrogate model should run in a remote process.",
)
@click.option(
    "--gpus_per_surrogate",
    type=click.FLOAT,
    default=0.0,
    help="The number of (possibly fractional) GPUs to allocate per surrogate model."
    "This is only relevant if --remote_surrogate=True, and the surrogates must be"
    "run remotely in order to use GPUs.",
)
@click.option(
    "--offline", is_flag=True, help="Whether or not to run W&B in offline mode."
)
def main(
    dataset,
    surrogate_artifact,
    num_iterations,
    num_eval_episodes,
    total_rollouts_length,
    num_workers,
    acquisition_cost,
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
    remote_surrogate,
    gpus_per_surrogate,
    offline,
):
    config = locals()

    run = wandb.init(
        project="active-acquisition",
        job_type="train_ppo_air_surrogate",
        config=config,
        mode="disabled" if offline else "online",
    )

    surrogate_artifact = run.use_artifact(surrogate_artifact)
    surrogate_path = os.path.abspath(surrogate_artifact.download())

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
        "env_config": {
            "surrogate": {
                "type": surrogate_artifact.type,
                "config": {
                    "model_dir": surrogate_path,
                    "info_gains_evaluation_method": "scan_samples",
                },
                "run_remote": remote_surrogate,
                "num_gpus": gpus_per_surrogate,
            },
            "acquisition_cost": acquisition_cost,
        },
    }

    env_fn = create_surrogate_air_env_fn(dataset, "train")

    agent = PPOAgent(agent_config, env_fn)

    with open(os.path.join(run.dir, "agent_config.json"), "w") as fp:
        json.dump(agent_config, fp)

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

    agent_artifact = wandb.Artifact(
        f"{dataset}_ppo_air_surrogate_agent",
        metadata=dict(
            agent_class=PPOAgent.__name__,
            surrogate_artifact=surrogate_artifact.name,
            dataset=dataset,
        ),
        type="agent",
    )
    agent_artifact.add_file(os.path.join(run.dir, "best_weights.pkl"))
    agent_artifact.add_file(os.path.join(run.dir, "agent_config.json"))
    run.log_artifact(agent_artifact)


if __name__ == "__main__": 
    main()
