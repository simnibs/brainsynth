from collections import namedtuple

HEMISPHERES = ("lh", "rh")
SURFACE_RESOLUTIONS = (0, 1, 2, 3, 4, 5, 6)

# Labeling schemes
LabelingScheme = namedtuple(
    "LabelingScheme", ("incl_csf", "excl_csf")
)
labeling_scheme = LabelingScheme(
    incl_csf = [0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 77, 85],
    excl_csf = [0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,     26, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 77, 85],
)

# neutral labels are labels that are not lateralized
NNeutralLabels = namedtuple("NNeutralLabels", ("incl_csf", "excl_csf"))
n_neutral_labels = NNeutralLabels(incl_csf=7, excl_csf=6)

#   segmentation:
#     excl_brainstem_cerebellum: [0, 2, 3, 4,       10, 11, 12, 13,     17, 18, 26, 28, 77]
#     incl_brainstem_cerebellum: [0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28, 77]
#     n_neutral_labels: 6
