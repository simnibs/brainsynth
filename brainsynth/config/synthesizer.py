
import torch

from brainsynth.constants import IMAGE
from brainsynth.transforms.spatial import GridCentering


class SynthesizerConfig:
    def __init__(
        self,
        pipeline: str = "DefaultPipeline",
        in_res: list[float] | tuple[float, float, float] = (1.0, 1.0, 1.0),
        out_size: list[int] | tuple[int, int, int] = (192, 192, 192),
        align_corners: bool = True,
        out_center_str: str = "image",
        segmentation_labels: list[int] | tuple | str = "brainseg",
        photo_mode: bool = False,
        photo_spacing_range: list[float] | tuple[float, float] = (2.0, 7.0),
        photo_thickness: float = 0.001,
        alternative_images: None | list[str] | tuple = None,
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device)

        self.pipeline = pipeline

        assert len(in_res) == 3
        self.in_res = torch.tensor(in_res, device=self.device)

        assert len(out_size) == 3
        self.out_size = torch.tensor(out_size, device=self.device)
        assert torch.all(self.out_size % 2 == 0), "Output FOV should be divisible by 2."

        self.align_corners = align_corners
        assert (
            align_corners == True
        ), "Spatial augmentation is untested with align_corners = False and probably does not work properly."
        self.grid = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(s.item(), dtype=torch.float, device=self.device)
                    for s in self.out_size
                ],
                indexing="ij",
            ),
            dim=-1,
        )
        # center the grid
        grid_centering = GridCentering(self.out_size, self.align_corners, self.device)
        self.center = grid_centering.center
        self.centered_grid = grid_centering(self.grid)

        assert out_center_str in {"brain", "image", "lh", "random", "rh"}
        self.out_center_str = out_center_str

        if isinstance(segmentation_labels, str):
            segmentation_labels = getattr(IMAGE.labeling_scheme, segmentation_labels)
        self.segmentation_labels = segmentation_labels
        self.segmentation_num_labels = len(self.segmentation_labels)

        self.photo_mode = photo_mode

        assert len(photo_spacing_range) == 2
        self.photo_spacing_range = photo_spacing_range
        self.photo_thickness = photo_thickness

        # if alternative_images is not None:
        # alternative_images = tuple(f"{mikeys.image}:{i}" for i in alternative_images)
        self.alternative_images = alternative_images

    def __repr__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])
