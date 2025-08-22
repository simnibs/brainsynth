import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

resources_dir = Path(__file__).parent

HEMISPHERES = ("lh", "rh")
SURFACES = ("white", "sphere", "sphere.reg")
ANNOTATIONS = ("aparc",)


class Template:
    def __init__(self):
        self.subject_dir = resources_dir / "template"
        self.surfaces = {
            (h, s): self.subject_dir / "surf" / f"{h}.{s}"
            for h, s in itertools.product(HEMISPHERES, SURFACES)
        }
        self.annotations = {
            (h, atlas): self.subject_dir / "label" / f"{h}.{atlas}.annot"
            for h, atlas in itertools.product(HEMISPHERES, ANNOTATIONS)
        }

    def load_surface(self, hemi, surface="white", device=None):
        v, f = nib.freesurfer.read_geometry(self.surfaces[hemi, surface])
        v = torch.tensor(v.astype(np.float32))
        f = torch.tensor(f.astype(np.int32))
        if device is not None:
            device = torch.device(device)
            v = v.to(device)
            f = f.to(device)
        return dict(vertices=v, faces=f)

    def load_annotation(self, hemi, atlas="aparc"):
        labels, ctab, names = nib.freesurfer.read_annot(self.annotations[hemi, atlas])
        names = list(map(np.bytes_.decode, names))
        return labels, ctab, names

    def load_cortex_indices(
        self, hemi, invert=False, return_indices: bool = True, device=None
    ):
        labels, _, _ = self.load_annotation(hemi, "aparc")
        mask = labels != -1  # name of 'unknown' label
        mask = np.invert(mask) if invert else mask
        if return_indices:
            mask = np.where(mask)[0].astype(np.int32)
        return torch.tensor(mask, device=device)
