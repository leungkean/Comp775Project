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
from afa.networks.segment.unet2 import UNet
from keras.layers import RandomFlip


def load_datasets(
    dataset,
    validation_split,
    batch_size,
    max_observed_percentage,
    min_observed_percentage,
    mask_generator=ImageBernoulliMaskGenerator(0.8),
    seed=42,
    repeat=True,
):
    train, info = tfds.load(dataset, split="train", with_info=True)
    val = tfds.load(dataset, split=validation_split)

    if "image" in train.element_spec:

        def cast_image(d):
            img = tf.cast(d["image"], tf.float32) / 255.0
            mask = tf.cast(d["segment"], tf.float32) / 255.0
            return {
                "image": img,
                "segment": mask,
            }

        train = train.map(cast_image)
        val = val.map(cast_image)

    data_key = "image" if "image" in train.element_spec else "features"
    data_shape = train.element_spec[data_key].shape

    train = train.shuffle(20000)

    train = train.batch(batch_size)
    val = val.batch(batch_size)

    def augment(d):
        data_key = "image" if "image" in d else "features"
        d[data_key] = RandomFlip(mode="horizontal", seed=seed)(d[data_key])
        d["segment"] = RandomFlip(mode="horizontal", seed=seed)(d["segment"])
        return d

    if repeat: 
        train = train.repeat().map(augment) 
        val = val.repeat().map(augment)

    mask_fn = get_add_mask_fn(
        mask_generator
        or UniformMaskGenerator(
            bounds=(min_observed_percentage, max_observed_percentage)
        )
    )

    train = train.map(mask_fn)
    val = val.map(mask_fn)

    def separate_labels(d):
        y = d.pop("segment")
        d["x"] = d.pop("features" if "features" in d else "image")
        d["b"] = d.pop("mask")
        return d, y

    train = train.map(separate_labels)
    val = val.map(separate_labels)

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    return train, val, data_shape


@click.command()
@click.option(
    "--dataset", type=click.STRING, required=True, help="The dataset to train on."
)
@click.option(
    "--validation_split",
    type=click.STRING,
    default="validation",
    help="The data split to use for validation.",
)
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
    default=0.05,
    help="The lower bound on the percentage of features that can be marked as observed.",
)
@click.option(
    "--offline",
    is_flag=True,
    help="If flag is set, then run will not be tracked by W&B.",
)
def main(
    dataset,
    validation_split,
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

    train, val, data_shape = load_datasets(
        dataset,
        validation_split,
        batch_size,
        max_observed_percentage,
        min_observed_percentage,
    )

    model = UNet(input_size=data_shape)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
            optimizer, 
            loss = "binary_crossentropy",
            metrics = ['accuracy'],
    )
    model.summary()

    model.fit(train, epochs=epochs, validation_data=val, callbacks=[WandbCallback()], steps_per_epoch=10, validation_steps=5)

    saved_model_path = os.path.join(run.dir, "model/1/")
    tf.saved_model.save(model, saved_model_path)

    model_artifact = wandb.Artifact(f"{dataset}_unet_classifier", type="classifier")
    model_artifact.add_dir(saved_model_path)
    run.log_artifact(model_artifact)

    accuracies = []

    for p in np.linspace(0.0, 1.0, 21):
        _, ds, _ = load_datasets(
            dataset,
            validation_split,
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
