import json
import os
import pickle

import click
import tensorflow as tf
import wandb

import sys
sys.path.insert(1, '../')

from afa import agents
from afa.environments.utils import create_surrogate_air_env_fn
from afa.evaluation import summarize_episode_infos

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass

tf.config.set_visible_devices(gpus[:1], device_type="GPU")


@click.command()
@click.option(
    "--agent_artifact",
    type=click.STRING,
    required=True,
    help="The name of the agent artifact to evaluate.",
)
@click.option(
    "--num_episodes",
    type=click.INT,
    required=True,
    help="The number of episodes to complete. Note that this cannot be larger "
    "than the size of the data split.",
)
@click.option(
    "--dataset",
    type=click.STRING,
    help="The dataset to evaluate on. If not specified, then the dataset that "
    "the agent was trained on will be used.",
)
@click.option(
    "--data_split",
    type=click.STRING,
    default="test",
    help="The data split to evaluate on.",
)
@click.option(
    "--num_workers",
    type=click.INT,
    default=16,
    help="The number of rollout workers to run.",
)
@click.option(
    "--deterministic",
    type=click.BOOL,
    default=True,
    help="Whether or not to deterministically (greedily) sample from the policy.",
)
@click.option(
    "--save_trajectories",
    type=click.BOOL,
    default=False,
    help="Whether or not trajectories should be saved as an artifact.",
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
def main(
    agent_artifact,
    num_episodes,
    dataset,
    data_split,
    num_workers,
    deterministic,
    save_trajectories,
    remote_surrogate,
    gpus_per_surrogate,
):
    config = locals()

    run = wandb.init(
        project="active-acquisition",
        job_type="eval_air_surrogate",
        config=config,
    )

    agent_artifact = run.use_artifact(agent_artifact)
    agent_path = agent_artifact.download()

    with open(os.path.join(agent_path, "agent_config.json"), "r") as fp:
        agent_config = json.load(fp)

    agent_cls = getattr(agents, agent_artifact.metadata["agent_class"])

    surrogate_artifact = run.use_artifact(agent_artifact.metadata["surrogate_artifact"])
    surrogate_path = os.path.abspath(surrogate_artifact.download())

    agent_config["env_config"]["surrogate"]["config"]["model_dir"] = surrogate_path
    agent_config.setdefault("evaluation_config", {})["num_workers"] = num_workers
    agent_config["evaluation_config"]["deterministic"] = deterministic
    agent_config["evaluation_config"].setdefault("surrogate", {})[
        "run_remote"
    ] = remote_surrogate
    agent_config["evaluation_config"].setdefault("surrogate", {})[
        "num_gpus"
    ] = gpus_per_surrogate

    env_fn = create_surrogate_air_env_fn(
        dataset or agent_artifact.metadata["dataset"],
        data_split,
        error_on_new_epoch=True,
    )

    agent = agent_cls(agent_config, env_fn)
    agent.load(os.path.join(agent_path, "best_weights.pkl"))

    stats, trajectories = agent.evaluate(num_episodes)

    episode_infos = stats.pop("episode_infos")
    stats.update(summarize_episode_infos(episode_infos))

    stats["num_acquisitions"] = stats["episode_length"] - 1

    run.log(stats)

    if save_trajectories:
        with open(os.path.join(run.dir, "trajectories.pkl"), "wb") as fp:
            pickle.dump(trajectories, fp)

        trajectories_artifact = wandb.Artifact(
            f"{agent_artifact.metadata['dataset']}_air_surrogate_trajectories",
            type="trajectories",
        )
        trajectories_artifact.add_file(os.path.join(run.dir, "trajectories.pkl"))
        run.log_artifact(trajectories_artifact)


if __name__ == "__main__":
    main()
