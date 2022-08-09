import os
import torch
import datasets
from datasets import DownloadManager
_URL = "URL of wikisum dataset"

class NewsHeadDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description = "wikisum multi document summarisation dataset",
            features = datasets.Features(
                {
                    "documents": datasets.Value("string"),
                    "summary": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager: DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs = {
                    "filepath": os.path.join(_URL, "train.pt"),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs = {
                    "filepath": os.path.join(_URL, "validation.pt"),
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs = {
                    "filepath": os.path.join(_URL, "test.pt"),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, filepath, split):
        data = torch.load(filepath)
        for id_, example in enumerate(data):
            yield id_, {
                "documents": "|||||".join(example['text'])+"|||||",
                "summary": example["tgt"]
            }
