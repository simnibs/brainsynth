from pathlib import Path
import sys

import cortech
import numpy as np
import torch

import brainsynth

HEMISPHERES = ("lh", "rh")


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


def estimate_affines(subject_dir):
    subject_dir = Path(subject_dir)

    template = brainsynth.resources.Template()
    template = {
        h: cortech.Surface(**template.load_surface(h, "white")) for h in HEMISPHERES
    }
    # for k in template:
    #     template[k].to_scanner_ras()

    target = {
        h: cortech.Surface.from_file(subject_dir / f"{h}.white.resample")
        for h in HEMISPHERES
    }
    # for k in target:
    #     target[k].to_scanner_ras()

    affine = estimate_affine_hemis(template, target)
    for k, v in affine.items():
        torch.save(v, subject_dir / f"mni305-to-ras.{k}.pt")

    affine_brain = estimate_affine_brain(template, target)
    torch.save(affine_brain, subject_dir / "mni305-to-ras.brain.pt")


if __name__ == "__main__":
    estimate_affines(sys.argv[1])
