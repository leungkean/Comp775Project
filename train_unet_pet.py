#################### New ####################
import os

import click
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from tensorflow import keras
from wandb.integration.keras import WandbCallback

import sys
sys.path.insert(1, '../')

from afa.masking import get_add_mask_fn, UniformMaskGenerator, BernoulliMaskGenerator, ImageBernoulliMaskGenerator
from afa.networks.segment.unet3 import UNet
from afa.data import load_pet_as_numpy
from keras.layers import RandomFlip
from keras.losses import SparseCategoricalCrossentropy

def load_pet_dataset(
        batch_size,
        max_observed_percentage,
        min_observed_percentage,
        mask_generator=ImageBernoulliMaskGenerator(0.25),
        seed=123,
        repeat=True,
):
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    def normalize(input_image, input_mask): 
        input_image = tf.cast(input_image, tf.float32) / 255.0 
        input_mask -= 1 
        return input_image, input_mask

    def load_image(datapoint): 
        input_image = tf.image.resize(datapoint['image'], (64, 64)) 
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (64, 64)) 
        input_image, input_mask = normalize(input_image, input_mask) 
        return {
            'image': input_image,
            'segment': input_mask,
        }

    train = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    data_key = "image" if "image" in train.element_spec else "features"
    data_shape = train.element_spec[data_key].shape
    train = train.shuffle(info.splits['train'].num_examples, seed=seed)

    train = train.batch(batch_size)

    def augment(d):
        data_key = "image" if "image" in d else "features"
        d[data_key] = RandomFlip(mode="horizontal", seed=seed)(d[data_key])
        d["segment"] = RandomFlip(mode="horizontal", seed=seed)(d["segment"])
        return d

    if repeat: 
        train = train.repeat().map(augment)

    mask_fn = get_add_mask_fn(
        mask_generator
        or UniformMaskGenerator(
            bounds=(min_observed_percentage, max_observed_percentage)
        )
    )

    train = train.map(mask_fn)

    def separate_labels(d):
        y = d.pop("segment")
        d["x"] = d.pop("features" if "features" in d else "image")
        d["b"] = d.pop("mask")
        return d, y

    train = train.map(separate_labels)

    train = train.prefetch(tf.data.AUTOTUNE)

    return train, data_shape, info.splits['train'].num_examples // batch_size

@click.command()
@click.option(
    "--epochs", type=click.INT, default=10, help="The number of training epochs."
)
@click.option("--batch_size", type=click.INT, default=32, help="The batch size.")
@click.option("--lr", type=click.FLOAT, default=1e-3, help="The learning rate.")
@click.option(
    "--activation", type=click.STRING, default="relu", help="The activation function."
)
@click.option("--dropout", type=click.FLOAT, help="The dropout rate.")
@click.option(
    "--max_observed_percentage",
    type=click.FLOAT,
    default=0.5,
    help="The upper bound on the percentage of features that can be marked as observed.",
)
@click.option(
    "--min_observed_percentage",
    type=click.FLOAT,
    default=0.0,
    help="The lower bound on the percentage of features that can be marked as observed.",
)
@click.option(
    "--offline",
    is_flag=True,
    help="If flag is set, then run will not be tracked by W&B.",
)
def main(
    epochs,
    batch_size,
    lr,
    activation,
    dropout,
    max_observed_percentage,
    min_observed_percentage,
    offline,
):
    """Trains an MLP partially observed classifier."""
    config = locals()
    del config["offline"]

    run = wandb.init(
        project="active-acquisition",
        job_type="train_classifier",
        mode="disabled" if offline else "online",
        config=config,
        magic=True,
    )

    train, data_shape, steps_per_epoch = load_pet_dataset(
        batch_size,
        max_observed_percentage,
        min_observed_percentage,
    )

    features, targets = load_pet_as_numpy('train')

    model = UNet(input_size=data_shape)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
            optimizer, 
            loss=SparseCategoricalCrossentropy(),
            metrics = ['accuracy'],
    )
    model.summary()

    model.fit(train, epochs=epochs, callbacks=[WandbCallback()], steps_per_epoch=steps_per_epoch)

    saved_model_path = os.path.join(run.dir, "model/1/")
    model.save(saved_model_path)

    model_artifact = wandb.Artifact(f"oxford_pet_unet_classifier", type="classifier")
    model_artifact.add_dir(saved_model_path)
    run.log_artifact(model_artifact)

    accuracies = []

    for p in np.linspace(0.0, 1.0, 21):
        ds, _, _ = load_pet_dataset(
            batch_size,
            max_observed_percentage,
            min_observed_percentage,
            mask_generator=ImageBernoulliMaskGenerator(p),
            repeat=False,
        )

        _, acc = model.evaluate(ds)
        accuracies.append((p, acc))

    table = wandb.Table(data=accuracies, columns=["Percent Observed", "Accuracy"])
    plot = wandb.plot.line(
        table,
        "Percent Observed",
        "Accuracy",
        title="Accuracy vs. Missingness",
    )

    run.log({"accuracy_missingness_plot": plot})


if __name__ == "__main__":
    main()
