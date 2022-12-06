#################### New ####################
import json
import os
import pickle

import click
import numpy as np
import wandb
import tensorflow as tf

import sys
sys.path.insert(1, '../')

from afa import agents
from afa.agents.ppo import PPOAgent
from afa.environments.utils import create_unet_env_fn
from afa.evaluation import summarize_episode_infos


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
    help="The number of episodes to complete.",
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
    default=True,
    help="Whether or not trajectories should be saved as an artifact.",
)
def main(
    agent_artifact, num_episodes, data_split, num_workers, deterministic, save_trajectories
):
    config = locals()

    run = wandb.init(
        project="active-acquisition",
        job_type="eval_unet_ppo_env",
        config=config,
    )

    agent_artifact = run.use_artifact(agent_artifact)
    agent_path = agent_artifact.download()

    with open(os.path.join(agent_path, "agent_config.json"), "r") as fp:
        agent_config = json.load(fp)

    classifier_artifact = run.use_artifact("oxford_pet_unet_classifier:v20")
    classifier_dir = classifier_artifact.download()

    agent_config.setdefault("evaluation_config", {})["num_workers"] = num_workers
    agent_config["evaluation_config"]["deterministic"] = deterministic

    env_fn = create_unet_env_fn(data_split, classifier_dir)

    agent = PPOAgent(agent_config, env_fn)
    agent.load(os.path.join(agent_path, "best_weights.pkl"))

    stats, trajectories = agent.evaluate(num_episodes)

    episode_infos = stats.pop("episode_infos")
    stats.update(summarize_episode_infos(episode_infos))

    stats["num_acquisitions"] = stats["episode_length"] - 1

    run.log(stats)

    if save_trajectories:
        # Add classifier predictions to saved trajectories.
        classifier = tf.keras.models.load_model(classifier_dir)

        for t in trajectories:
            x = t["obs"]["observed"]
            b = t["obs"]["mask"]
            t["classifier_preds"] = np.asarray(classifier.predict({"x": x, "b": b}))

        with open(os.path.join(run.dir, "trajectories.pkl"), "wb") as fp:
            pickle.dump(trajectories, fp)

        trajectories_artifact = wandb.Artifact(
            f"unet_ppo_trajectories", type="trajectories"
        )
        trajectories_artifact.add_file(os.path.join(run.dir, "trajectories.pkl"))
        run.log_artifact(trajectories_artifact)


if __name__ == "__main__":
    main()
