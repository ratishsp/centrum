import os
import json
import datasets
from datasets import DownloadManager
_URL = "URL of NewSHead dataset"

class NewsHeadDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description = "newshead multi document summarisation dataset",
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
                    "filepath": os.path.join(_URL, "train.jsonl"),
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs = {
                    "filepath": os.path.join(_URL, "validation.jsonl"),
                    "split": "validation"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs = {
                    "filepath": os.path.join(_URL, "test.jsonl"),
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, {
                    "documents": data["text"],
                    "summary": data["summary"]
                }


