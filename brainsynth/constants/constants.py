from collections import namedtuple

HEMISPHERES = ("lh", "rh")
SURFACE_RESOLUTIONS = (0, 1, 2, 3, 4, 5, 6)
SURFACES = ("white", "pial")

# Labeling schemes
LabelingScheme = namedtuple(
    # "LabelingScheme", ("incl_csf", "excl_csf", "wmgm")
    "LabelingScheme", ("brainseg", "brainseg_with_extracerebral",)

)
# labeling_scheme = LabelingScheme(
#     incl_csf = [0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 77, 85],
#     excl_csf = [0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,     26, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 77, 85],
#     wmgm = [0, 2, 3, 41, 42],
# )

labeling_scheme = LabelingScheme(
    brainseg = tuple(range(44)),                 # brainseg.lut
    brainseg_with_extracerebral = tuple(range(57)),   # brainseg_with_extracerebral.lut
)

# neutral labels are labels that are not lateralized
NNeutralLabels = namedtuple("NNeutralLabels", ("incl_csf", "excl_csf", "wmgm"))
n_neutral_labels = NNeutralLabels(incl_csf=7, excl_csf=6, wmgm=1)

# Mapped inputs

MappedInputKeys = namedtuple("MappedInputKeys", ["image", "surface", "initial_vertices", "state"])
mapped_input_keys = MappedInputKeys("image", "surface", "initial_vertices", "state")
