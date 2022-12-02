import os
import pickle

import gdown
import tensorflow as tf
import tensorflow_datasets as tfds

url="https://drive.google.com/file/d/1JAhPj63jo9II6Ed2TBvFi4Iet4faZwnw/view?usp=share_link"

class Retina(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Retina dataset",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(128, 128, 1)),
                    "segment": tfds.features.Image(shape=(128, 128, 1)),
                    "label": tfds.features.ClassLabel(num_classes=2),
                }
            ),
            supervised_keys=("features", "label"),
            citation=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        data_path = gdown.download(
            url=url, output=os.path.join(dl_manager.download_dir, "retina_data.pkl"), fuzzy=True 
        )

        with open(data_path, "rb") as fp:
            data = pickle.load(fp)

        return {
            "train": self._generate_examples(data["train"]),
            "validation": self._generate_examples(data["valid"]),
            "test": self._generate_examples(data["test"]),
        }

    def _generate_examples(self, data):
        for i, (x, y, z) in enumerate(zip(*data)):
            record = {
                    "image": x,
                    "segment": y,
                    "label": z,
            }
            yield i, record
