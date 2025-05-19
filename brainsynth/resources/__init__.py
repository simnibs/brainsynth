import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

resources_dir = Path(__file__).parent

hemispheres = ("lh", "rh")
surface_names = ("white", "sphere", "sphere.reg")

surfaces = {
    (h, s): resources_dir / f"{h}.{s}"
    for h, s in itertools.product(hemispheres, surface_names)
}


def load_cortical_template(hemi, surface="white", device=None):
    # lh.white.srf
    # lh.sphere.reg.srf
    # v, f = nib.freesurfer.read_geometry(resources_dir / f"cortex-int-{hemi}.srf")
    v, f = nib.freesurfer.read_geometry(surfaces[hemi, surface])
    v = torch.tensor(v.astype(np.float32))
    f = torch.tensor(f.astype(np.int32))
    if device is not None:
        device = torch.device(device)
        v = v.to(device)
        f = f.to(device)
    return dict(vertices=v, faces=f)
