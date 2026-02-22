# Truncation modules of LLaMA3 Vision
# We only used it to add empty image to the data -- do not run real truncation
import torch


# Llama3Vision only provides a SafeTruncation
class SafeTruncation:
    def __init__(
        self,
        content: int = 128256,  # <|image|>
        pad: int = 128004,  # <|endoftext|>; in llama3 it should be <|finetune_right_pad_id|>
        # Llama3vision does not have start & end
        # start: int = 151652,
        # end: int = 151653,
        train: bool = False,  # if True, will add an empty image to current batch
        black_image_size=448
    ):
        self.content = content
        self.pad = pad
        self.train = train
        self.black_image_size = black_image_size

    def sanity_check(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        aspect_ratio_ids,
        aspect_ratio_mask,
        cross_attention_mask,
    ):
        # sanity check of the inputs
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask cannot be None")

        if pixel_values is None:
            if torch.sum(input_ids == self.content).item():
                raise ValueError(
                    "The input_ids contains <|image|> tokens but no pixel_values is provided"
                )

    def add_black_image(self, input_ids, attention_mask):
        # padding with zero tensors -- no need to insert <|image|>
        bs, seq_len = input_ids.shape
        device = input_ids.device
        dtype = torch.float32
        num_images, num_tiles, num_channels, h, w = 1, 4, 3, self.black_image_size, self.black_image_size

        pixel_values = torch.ones(
            (bs, num_images, num_tiles, num_channels, h, w), dtype=dtype, device=device
        )
        aspect_ratio_ids = torch.ones(
            (bs, num_images), dtype=torch.int64, device=device
        )
        aspect_ratio_mask = torch.zeros(
            (bs, num_images, num_tiles), dtype=torch.int64, device=device
        )
        cross_attention_mask = torch.zeros(
            (bs, seq_len, num_images, num_tiles), dtype=torch.int64, device=device
        )

        return (
            input_ids,
            attention_mask,
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
            cross_attention_mask,
        )

    def truncate(
        self,
        inputs,
        length=32768,
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs.get("pixel_values", None)
        aspect_ratio_ids = inputs.get("aspect_ratio_ids", None)
        aspect_ratio_mask = inputs.get("aspect_ratio_mask", None)
        cross_attention_mask = inputs.get("cross_attention_mask", None)

        self.sanity_check(
            input_ids,
            attention_mask,
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
            cross_attention_mask,
        )

        # if no images, return the original inputs or add an additional image (during training)
        if pixel_values is None:
            if self.train:
                (
                    input_ids,
                    attention_mask,
                    pixel_values,
                    aspect_ratio_ids,
                    aspect_ratio_mask,
                    cross_attention_mask,
                ) = self.add_black_image(
                    input_ids,
                    attention_mask,
                )
                self.sanity_check(
                    input_ids,
                    attention_mask,
                    pixel_values,
                    aspect_ratio_ids,
                    aspect_ratio_mask,
                    cross_attention_mask,
                )
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "aspect_ratio_ids": aspect_ratio_ids,
                    "aspect_ratio_mask": aspect_ratio_mask,
                    "cross_attention_mask": cross_attention_mask,
                }
            else:
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": None,
                    "aspect_ratio_ids": None,
                    "aspect_ratio_mask": None,
                    "cross_attention_mask": None,
                }
        else:  # when we have images, no operation is needed
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "aspect_ratio_ids": aspect_ratio_ids,
                "aspect_ratio_mask": aspect_ratio_mask,
                "cross_attention_mask": cross_attention_mask,
            }
