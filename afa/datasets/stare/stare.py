import os
import pickle

import gdown
import tensorflow as tf
import tensorflow_datasets as tfds

url="https://drive.google.com/file/d/104Mzdu0a5a975D5IOSmjRpZrA80cfswS/view?usp=sharing"

class STARE(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="STructured Analysis of the Retina",
            features=tfds.features.FeaturesDict(
                {"features": tfds.features.Tensor(shape=(1024,), dtype=tf.float32)}
            ),
            supervised_keys=None,
            citation=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        data_path = gdown.download(
            url=url, output=os.path.join(dl_manager.download_dir, "stare_filter.pkl"), fuzzy=True 
        )

        with open(data_path, "rb") as fp:
            data = pickle.load(fp)

        return {
            "train": self._generate_examples(data["train"]),
            "validation": self._generate_examples(data["valid"]),
            "test": self._generate_examples(data["test"]),
        }

    def _generate_examples(self, data):
        for i, x in enumerate(data):
            yield i, dict(features=x)
