# dataset utils for local VIRA dataset
# needed keys are: `qry`, `pos_text`, `pos_image` (`neg_text`, `neg_image_{i}` not used)
import json
import os

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _process_image(image, resolution="original"):
    if image is None:
        return None
    image = Image.open(image)
    if resolution == "original":
        return image
    elif resolution == "high":
        image = image.resize((512, 512))
    else:
        image = image.resize((336, 336))
    return image


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]


class VIRADataset(Dataset):
    def __init__(
        self,
        data_dir,
        q2s_file,
        tgt_modalities,
        caption_dict,
    ):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, q2s_file), "r") as f:
            self.q2s = [json.loads(line) for line in f.readlines()]
        self.tgt_modalities = tgt_modalities.split(",")
        self.caption_dict = caption_dict

    def __len__(self):
        return len(self.q2s)

    def __getitem__(self, item):
        data_dict = self.q2s[item]
        example_dict = {"qry": data_dict["query"]}

        if "text" in self.tgt_modalities:
            example_dict["pos_text"] = self.caption_dict[data_dict["image_path"]]
        if "image" in self.tgt_modalities:
            example_dict["pos_image"] = _process_image(
                os.path.join(self.data_dir, "News", data_dict["image_path"])
            )
            example_dict["pos_image_path"] = data_dict["image_path"]

        # at least one of text and image is required
        assert (
            "pos_text" in example_dict or "pos_image" in example_dict
        ), "At least one of text and image is required"

        return example_dict


# 300k images available
class VIRATrainsetWrapper:
    def __init__(
        self,
        data_dir=None,
        train_q2s_file="train_q2s.jsonl",
        eval_q2s_file="eval_q2s.jsonl",
        caption_file="filtered_caption.jsonl",
        tgt_modalities="text,image",
    ):
        if data_dir is None:
            data_dir = os.environ.get("VIRA_DATA_DIR", "./data_dir/VIRA/News")
        self.data_dir = data_dir
        self.train_q2s_file = train_q2s_file
        self.eval_q2s_file = eval_q2s_file
        # self.caption_file = caption_file
        self.tgt_modalities = tgt_modalities
        # both train and eval share the same caption file
        with open(os.path.join(data_dir, caption_file), "r") as f:
            caption = [json.loads(line) for line in f.readlines()]
        # need a mapping from image path to caption
        self.caption_dict = {x["image_path"]: x["text"] for x in caption}

    def __call__(self):
        return (
            {
                "train": VIRADataset(
                    data_dir=self.data_dir,
                    q2s_file=self.train_q2s_file,
                    tgt_modalities=self.tgt_modalities,
                    caption_dict=self.caption_dict,
                ),
                "test": VIRADataset(
                    data_dir=self.data_dir,
                    q2s_file=self.eval_q2s_file,
                    tgt_modalities=self.tgt_modalities,
                    caption_dict=self.caption_dict,
                ),
            },
            None,
            "vira",
        )
