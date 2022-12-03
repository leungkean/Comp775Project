import click
import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import wandb
from keras.models import *

from ray.rllib.algorithms.ppo import PPO

from afa.data import load_pet_as_numpy
from afa.environments.dataset_manager import EnvironmentDatasetManager
from afa.environments.core import PretrainedUNetEnv
from afa.networks.segment.unet3 import UNet

import sys

def main():
    config = locals()
    # Load the dataset
    features, targets = load_pet_as_numpy('train')

    dataset_manager = EnvironmentDatasetManager(
            features, targets
    )

    run = wandb.init(
            project="active-acquisition",
            job_type="train_segment_afa",
            config=config,
            mode="online",
    )

    unet_artifact = run.use_artifact("oxford_pet_unet_classifier:v14")
    unet_artifact_dir = os.path.abspath(unet_artifact.download())

    model = UNet()
    model.load_model(unet_artifact_dir)

    model.summary()

if __name__ == "__main__":
    main()
