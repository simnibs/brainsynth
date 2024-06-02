from collections import namedtuple

import torch

# neutral labels are labels that are not lateralized
# NNeutralLabels = namedtuple("NNeutralLabels", ("incl_csf", "excl_csf", "wmgm"))
# n_neutral_labels = NNeutralLabels(incl_csf=7, excl_csf=6, wmgm=1)

# Mapped inputs

MappedInputKeys = namedtuple(
    "MappedInputKeys", ["image", "surface", "initial_vertices", "state"]
)
mapped_input_keys = MappedInputKeys("image", "surface", "initial_vertices", "state")


# ---------------------------
# IMAGES
# ---------------------------

# Labeling schemes
LabelingScheme = namedtuple(
    "LabelingScheme",
    (
        "brainseg",
        "brainseg_with_extracerebral",
    ),
)

# Generation labels
GenerationLabels = namedtuple(
    "GenerationLabels",
    ("n_labels", "label_range", "kmeans", "lesion", "white", "gray", "csf", "pv"),
)
# PV is encoded between 100 and 250 such that
#   100 = 1 lesions
#   150 = 2 white matter
#   200 = 3 gray matter
#   250 = 4 csf
# where
#   RHS = 1 + (LHS-100)/50
#   LHS = 100 + 50(RHS-1).
# For example, 151, 152, ..., 199 encode fractional steps from WM to GM.
PartialVolume = namedtuple("PartialVolume", ("lesion", "white", "gray", "csf"))


# Image name to image filename mapper
Images = namedtuple(
    "ImageFiles",
    (
        "brain_dist_map",
        "brainseg",
        "brainseg_with_extracerebral",
        "generation_labels",
        "mni_reg_x",
        "mni_reg_y",
        "mni_reg_z",
        "lp_dist_map",
        "lw_dist_map",
        "rp_dist_map",
        "rw_dist_map",
        "t1w",
        "t1w_mask",
        "t2w",
        "t2w_mask",
        "ct",
        "ct_mask",
        "flair",
        "flair_mask",
    ),
)
MetaData = namedtuple(
    "ImageData",
    ("filename", "dtype", "defacingmask"),
)
class ImageSettings:
    def __init__(self):
        self.labeling_scheme = LabelingScheme(
            brainseg=tuple(range(44)),  # brainseg.lut
            brainseg_with_extracerebral=tuple(range(57)),  # brainseg_with_extracerebral.lut
        )

        self.generation_labels = GenerationLabels(
            n_labels=256,
            label_range=(0, 100),  # the actual labels; the rest are some mix of these
            kmeans=(12, 13, 14, 15),
            lesion=1,
            white=2,
            gray=3,
            csf=4,
            pv=PartialVolume(lesion=100, white=150, gray=200, csf=250),
        )

        self.images = Images(
            brain_dist_map=MetaData("brain_dist_map.nii", torch.float, None),
            brainseg=MetaData("brainseg.nii", torch.int32, None),
            brainseg_with_extracerebral=MetaData(
                "brainseg_with_extracerebral.nii", torch.int32, None
            ),
            ct=MetaData("CT.nii", torch.float, "ct_mask"),
            ct_mask=MetaData("CT.defacingmask.nii", torch.bool, None),
            flair=MetaData("FLAIR.nii", torch.float, "ct_mask"),
            flair_mask=MetaData("FLAIR.defacingmask.nii", torch.float, None),
            generation_labels=MetaData("generation_labels.nii", torch.int32, None),
            lp_dist_map=MetaData("lp_dist_map.nii", torch.float, None),
            lw_dist_map=MetaData("lw_dist_map.nii", torch.float, None),
            rp_dist_map=MetaData("rp_dist_map.nii", torch.float, None),
            rw_dist_map=MetaData("rp_dist_map.nii", torch.float, None),
            mni_reg_x=MetaData("mni_reg.x.nii", torch.float, None),
            mni_reg_y=MetaData("mni_reg.y.nii", torch.float, None),
            mni_reg_z=MetaData("mni_reg.z.nii", torch.float, None),
            t1w=MetaData("T1w.nii", torch.float, "t1w_mask"),
            t1w_mask=MetaData("T1w.defacingmask.nii", torch.bool, None),
            t2w=MetaData("T2w.nii", torch.float, "t2w_mask"),
            t2w_mask=MetaData("T2w.defacingmask.nii", torch.bool, None),
        )
        self.dist_maps = {"lp_dist_map", "lw_dist_map", "rp_dist_map", "rw_dist_map"}

        self.default_images = [
            "brain_dist_map",
            "brainseg",
            "brainseg_with_extracerebral",
            "generation_labels",
            "mni_reg_x",
            "mni_reg_y",
            "mni_reg_z",
            "lp_dist_map",
            "lw_dist_map",
            "rp_dist_map",
            "rw_dist_map",
            "t1w",
        ]

# ---------------------------
# SURFACES
# ---------------------------

class SurfaceFiles:
    def __init__(self, hemispheres, types, resolutions):
        self.prediction, self.target, self.template = {}, {}, {}
        for h in hemispheres:
            for r in resolutions:
                for t in types:
                    k = (h,t,r)
                    self.prediction[k] = f"{h}.{t}.{r}.prediction.pt"
                    self.target[k] = f"{h}.{t}.{r}.target.pt"
                self.template[(h,r)] = f"{h}.{r}.template.pt"

class SurfaceSettings:
    def __init__(self):
        self.hemispheres: tuple[str, str] = ("lh", "rh")
        self.types: tuple = ("white", "pial")
        self.resolutions: tuple = (0, 1, 2, 3, 4, 5, 6)
        self.files = SurfaceFiles(self.hemispheres, self.types, self.resolutions)

IMAGE = ImageSettings()
SURFACE = SurfaceSettings()
