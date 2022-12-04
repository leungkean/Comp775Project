import json
import os
import pickle

import click
import numpy as np
import wandb
from tensorflow import saved_model

import sys
sys.path.insert(1, '../')

from afa.agents.dagger import DaggerAgent
from afa.environments.utils import create_pretrained_classifier_env_fn
from afa.evaluation import summarize_episode_infos


@click.command()
@click.option("--dataset", type=click.STRING, required=True, help="The dataset.")
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
    default=1,
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
    dataset, agent_artifact, num_episodes, data_split, num_workers, deterministic, save_trajectories
):
    config = locals()

    run = wandb.init(
        project="active-acquisition",
        job_type="eval_dagger_pretrained_classifier_env",
        config=config,
    )

    agent_artifact = run.use_artifact(agent_artifact)
    agent_path = agent_artifact.download()

    with open(os.path.join(agent_path, "agent_config.json"), "r") as fp:
        agent_config = json.load(fp)

    classifier_artifact = os.path.basename(
        agent_config["oracle_config"]["fitness_fn_config"]["kwargs"]["classifier_dir"]
    )
    classifier_artifact = run.use_artifact(classifier_artifact)
    classifier_dir = classifier_artifact.download()

    agent_config.setdefault("evaluation_config", {})["num_workers"] = num_workers
    agent_config["evaluation_config"]["deterministic"] = deterministic

    env_fn = create_pretrained_classifier_env_fn(dataset, data_split)

    agent = DaggerAgent(agent_config, env_fn)
    agent.load(os.path.join(agent_path, "best_weights.pkl"))

    stats, trajectories = agent.evaluate(num_episodes)

    episode_infos = stats.pop("episode_infos")
    stats.update(summarize_episode_infos(episode_infos))

    stats["num_acquisitions"] = stats["episode_length"] - 1

    run.log(stats)

    if save_trajectories:
        # Add classifier predictions to saved trajectories.
        classifier = saved_model.load(classifier_dir)

        for t in trajectories:
            x = t["obs"]["observed"]
            b = t["obs"]["mask"]
            t["classifier_preds"] = np.asarray(classifier({"x": x, "b": b}))

        with open(os.path.join(run.dir, "trajectories.pkl"), "wb") as fp:
            pickle.dump(trajectories, fp)

        trajectories_artifact = wandb.Artifact(
            f"{dataset}_dagger_trajectories", type="trajectories"
        )
        trajectories_artifact.add_file(os.path.join(run.dir, "trajectories.pkl"))
        run.log_artifact(trajectories_artifact)


if __name__ == "__main__":
    main()
