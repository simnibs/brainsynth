from dataclasses import dataclass
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from brainsynth.resources import resources_dir

HEMISPHERES = ("lh", "rh")


@dataclass
class Surface:
    vertices: npt.NDArray
    faces: npt.NDArray


def read_surface_as_ras(f):
    v, f, m = nib.freesurfer.read_geometry(f, read_metadata=True)
    if "cras" in m:
        return Surface(v + m["cras"], f)
    else:
        return Surface(v, f)


def estimate_affine_hemis(template, target):
    affine = {}
    for h in HEMISPHERES:
        nv = len(template[h].vertices)

        A = np.concatenate([template[h].vertices, np.ones((nv, 1))], axis=-1)
        b = np.concatenate([target[h].vertices, np.ones((nv, 1))], axis=-1)
        X, _, _, _ = np.linalg.lstsq(A, b)

        # this way, application to the *right* of points is valid
        affine[h] = torch.tensor(np.ascontiguousarray(X.T).astype(np.float32))

    return affine


def estimate_affine_brain(template, target):
    template = np.concatenate([v.vertices for v in template.values()], axis=0)
    target = np.concatenate([v.vertices for v in target.values()], axis=0)

    nv = len(template)

    A = np.concatenate([template, np.ones((nv, 1))], axis=-1)
    b = np.concatenate([target, np.ones((nv, 1))], axis=-1)
    X, _, _, _ = np.linalg.lstsq(A, b)

    # this way, application to the *right* of points is valid
    affine = torch.tensor(np.ascontiguousarray(X.T).astype(np.float32))

    return affine


def estimate_affines(DATASET_DIR):
    DATASET_DIR = Path(DATASET_DIR)

    template = {
        h: read_surface_as_ras(resources_dir / f"{h}.white") for h in HEMISPHERES
    }

    for ds_dir in sorted(DATASET_DIR.glob("*")):
        print(ds_dir.name)
        for sub_dir in tqdm(sorted(ds_dir.glob("sub-*"))):
            target = {
                h: read_surface_as_ras(sub_dir / f"{h}.white.resample")
                for h in HEMISPHERES
            }

            affine = estimate_affine_hemis(template, target)
            for k, v in affine.items():
                torch.save(v, sub_dir / f"mni305-to-ras.{k}.pt")

            affine_brain = estimate_affine_brain(template, target)
            torch.save(affine_brain, sub_dir / "mni305-to-ras.brain.pt")


if __name__ == "__main__":
    estimate_affines(sys.argv[1])
