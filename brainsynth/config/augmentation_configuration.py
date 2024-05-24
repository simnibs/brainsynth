import torch

from brainsynth.constants.constants import mapped_input_keys as mikeys

class AugmentationConfiguration:
    def __init__(
        self,
        pipeline: str = "DefaultPipeline",
        in_res: list[float] | tuple[float, float, float] = (1.0, 1.0, 1.0),
        out_size: list[int] | tuple[int, int, int] = (192, 192, 192),
        out_center_str: str = "image",
        segmentation_labels: None | list[int] | tuple[int] = None,
        photo_mode: bool = False,
        photo_spacing_range: list[float] | tuple[float, float] = (2.0, 7.0),
        alternative_images: None | list[str] | tuple[str] = None,
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device)

        self.pipeline = pipeline

        assert len(in_res) == 3
        self.in_res = torch.tensor(in_res, device=self.device)

        assert len(out_size) == 3
        self.out_size = torch.tensor(out_size, device=self.device)

        assert out_center_str in {"brain", "image", "lh", "random", "rh"}
        self.out_center_str = out_center_str

        self.segmentation_labels = segmentation_labels
        self.segmentation_num_labels = len(segmentation_labels)

        self.photo_mode = photo_mode

        assert len(photo_spacing_range) == 2
        self.photo_spacing_range = photo_spacing_range

        # if alternative_images is not None:
            # alternative_images = tuple(f"{mikeys.image}:{i}" for i in alternative_images)
        self.alternative_images = alternative_images

    def __repr__(self):
        return "\n".join([f"{k}: {v}" for k,v in self.__dict__.items()])

