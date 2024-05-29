from collections import namedtuple
import itertools

from brainsynth.constants.constants import HEMISPHERES, SURFACE_RESOLUTIONS, SURFACES

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
    "AdditionalImageFiles", ("CT", "FLAIR", "PD", "T1", "T2", "segmentation_brain")
)
optional_images = OptionalImageFiles(
    CT="ct.nii",
    FLAIR="flair.nii",
    PD="pd.nii",
    T1="t1.nii",
    T2="t2.nii",
    segmentation_brain="segmentation_brain.nii",
)

surfaces = {
    (res,hemi): f"surface.{res}.{hemi}.pt" for res,hemi in itertools.product(SURFACE_RESOLUTIONS, HEMISPHERES)
}

surface_templates = {
    (res,hemi): f"surface-template.{res}.{hemi}.pt" for res,hemi in itertools.product(SURFACE_RESOLUTIONS, HEMISPHERES)
}


# surfaces = {
#     (hemi, surf, res): f"{hemi}.{surf}.{res}.target.pt" for hemi, res, surf in itertools.product(HEMISPHERES, SURFACE_RESOLUTIONS, SURFACES)
# }
# surface_templates = {
#     (hemi, res): f"{hemi}.{res}.template.pt" for hemi, res in itertools.product(HEMISPHERES, SURFACE_RESOLUTIONS)
# }

surface_prediction = {
    (hemi, surf, res): f"{hemi}.{surf}.{res}.prediction.pt" for hemi, res, surf in itertools.product(HEMISPHERES, SURFACE_RESOLUTIONS, SURFACES)
}


info = "info.pt"

