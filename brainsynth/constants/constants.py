from collections import namedtuple
import itertools

import torch


HEMISPHERES = ("lh", "rh")
SURFACE_RESOLUTIONS = (0, 1, 2, 3, 4, 5, 6)

# Labeling schemes
LabelingScheme = namedtuple(
    "LabelingScheme", ("brainseg", "brainseg_with_extracerebral")
)
labeling_scheme = LabelingScheme(
    brainseg = list(range(44)),
    brainseg_with_extracerebral = list(range(57)),
)

# neutral labels are labels that are not lateralized
NNeutralLabels = namedtuple("NNeutralLabels", ("incl_csf", "excl_csf", "wmgm"))
n_neutral_labels = NNeutralLabels(incl_csf=7, excl_csf=6, wmgm=1)

generation_labels_kmeans = (12, 13, 14, 15)

#   segmentation:
#     excl_brainstem_cerebellum: [0, 2, 3, 4,       10, 11, 12, 13,     17, 18, 26, 28, 77]
#     incl_brainstem_cerebellum: [0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28, 77]
#     n_neutral_labels: 6



# Image name to image filename mapper
Images = namedtuple(
    "ImageFiles",
    ("brain_dist_map", "brainseg", "brainseg_with_extracerebral", "generation_labels", "mni_reg_x", "mni_reg_y", "mni_reg_z", "lp_dist_map", "lw_dist_map", "rp_dist_map", "rw_dist_map", "t1w", "t1w_mask", "t2w", "t2w_mask", "ct", "ct_mask", "flair", "flair_mask",)
)
MetaData = namedtuple("ImageData", ("filename", "dtype", "defacingmask", ))
images = Images(
    brain_dist_map = MetaData("brain_dist_map.nii", torch.float, None),
    brainseg=MetaData("brainseg.nii", torch.uint8, None),
    brainseg_with_extracerebral=MetaData("brainseg_with_extracerebral.nii", torch.uint8, None),
    ct=MetaData("CT.nii", torch.float, "ct_mask"),
    ct_mask=MetaData("CT.defacingmask.nii", torch.bool, None),
    flair=MetaData("FLAIR.nii", torch.float, "ct_mask"),
    flair_mask=MetaData("FLAIR.defacingmask.nii", torch.float, None),
    generation_labels=MetaData("generation_labels.nii", torch.int, None),
    lp_dist_map=MetaData("lp_dist_map.nii", torch.float, None),
    lw_dist_map=MetaData("lw_dist_map.nii", torch.float, None),
    rp_dist_map=MetaData("rp_dist_map.nii", torch.float, None),
    rw_dist_map=MetaData("rp_dist_map.nii", torch.float, None),
    mni_reg_x=MetaData("mni_reg.x.nii", torch.float, None),
    mni_reg_y=MetaData("mni_reg.y.nii", torch.float, None),
    mni_reg_z=MetaData("mni_reg.z.nii", torch.float, None),
    t1w=MetaData("T1w.nii", torch.float, "t1w_mask"),
    t1w_mask=MetaData("T1w.defacingmask.nii", torch.bool, None),
    t2w=MetaData("T2w.nii",torch.float,"t2w_mask"),
    t2w_mask=MetaData("T2w.defacingmask.nii", torch.bool, None),
)

dist_maps = {"lp_dist_map", "lw_dist_map", "rp_dist_map", "rw_dist_map"}
segmentations = {"brainseg", "brainseg_with_extracerebral"}


surfaces = {
    (res,hemi): f"surface.{res}.{hemi}.pt" for res,hemi in itertools.product(SURFACE_RESOLUTIONS, HEMISPHERES)
}

surface_templates = {
    (res,hemi): f"surface-template.{res}.{hemi}.pt" for res,hemi in itertools.product(SURFACE_RESOLUTIONS, HEMISPHERES)
}


info = "info.pt"
