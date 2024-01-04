from collections import namedtuple
import itertools

from brainsynth.constants.constants import HEMISPHERES, SURFACE_RESOLUTIONS

# Default images. These always exists
DefaultImageFiles = namedtuple(
    "DefaultImageFiles",
    ("bag", "distances", "generation", "segmentation", "synthseg", "norm")
)
default_images = DefaultImageFiles(
    bag="bag.nii",
    distances="distances.nii",
    generation="generation.nii",
    segmentation="segmentation.nii",
    synthseg="synthseg.nii",

    norm="norm.nii", # REMOVE ? REPLACE with T1 and use orig.mgz?
)

# Optional images
OptionalImageFiles = namedtuple(
    "AdditionalImageFiles", ("CT", "FLAIR", "PD", "T1", "T2")
)
optional_images = OptionalImageFiles(
    CT="ct.nii",
    FLAIR="flair.nii",
    PD="pd.nii",
    T1="t1.nii",
    T2="t2.nii",
)

surfaces = {
    (res,hemi): f"surface.{res}.{hemi}.pt" for res,hemi in itertools.product(SURFACE_RESOLUTIONS, HEMISPHERES)
}

surface_templates = {
    (res,hemi): f"surface-template.{res}.{hemi}.pt" for res,hemi in itertools.product(SURFACE_RESOLUTIONS, HEMISPHERES)
}


info = "info.pt"

