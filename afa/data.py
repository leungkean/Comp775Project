from typing import Tuple

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

def load_supervised_split_as_numpy(
    dataset: str, split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a supervised TensorFlow Dataset as a numpy array.

    Args:
        dataset: The name of the dataset.
        split: The split to load.

    Returns:
        The data as the tuple `(x, y)`.
    """
    ds = tfds.load(dataset, split=split, as_supervised=True)
    x, y = ds.batch(ds.cardinality().numpy()).get_single_element()
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x.shape) == 4:
        x = x.astype(np.float32) / 255.0
    return x, y

def load_pet_as_numpy(
        split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the Pet dataset as a numpy array.

    Returns:
        The data as the tuple `(x, y)`.
    """
    ds, info = tfds.load('oxford_iiit_pet:3.*.*', split=split, with_info=True)

    def normalize(input_image, input_mask): 
        input_image = tf.cast(input_image, tf.float32) / 255.0 
        input_mask -= 1 
        return input_image, input_mask 

    def load_image(datapoint):
          input_image = tf.image.resize(datapoint['image'], (64, 64))
          input_mask = tf.image.resize(datapoint['segmentation_mask'], (64, 64))
          input_image, input_mask = normalize(input_image, input_mask)
          return ( 
                  input_image, 
                  input_mask,
            )

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(info.splits[split].num_examples)
    x, y = ds.batch(ds.cardinality().numpy()).get_single_element()
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[:x.shape[0]//4]
    y = y[:y.shape[0]//4]
    return x, y

def load_unsupervised_split_as_numpy(dataset: str, split: str) -> np.ndarray:
    """Loads an unsupervised TensorFlow Dataset as a numpy array.

    Args:
        dataset: The name of the dataset.
        split: The split to load.

    Returns:
        The data as the tuple `(x, y)`.
    """
    ds = tfds.load(dataset, split=split)
    x = ds.batch(ds.cardinality().numpy()).get_single_element()
    data_key = "features" if "features" in x else "image"
    x = x[data_key]
    x = np.asarray(x)
    if len(x.shape) == 4:
        x = x.astype(np.float32) / 255.0
    return x
