from pathlib import Path
import sys

import cortech
import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import torch

import brainsynth

from brainnet.mesh.topology import DeepSurferTopology


def n_vertices_at_level(i, base_nf=120):
    """Estimate number of vertices at the ith level of the template
    surface.
    """
    return (base_nf * 4**i + 4) // 2


global nv
nv = {i: n_vertices_at_level(i) for i in range(6 + 1)}


def write_resolutions(v, resolutions, name):
    v = torch.tensor(v.astype(np.float32))
    for r in resolutions:
        vv = v[: nv[r]].clone()
        torch.save(vv, str(name).format(res=r))


def align_template_to_subjects(src_subject_dir, dst_subject_dir):
    src_subject_dir = Path(src_subject_dir)
    dst_subject_dir = Path(dst_subject_dir)

    resolutions = [0]
    top0 = DeepSurferTopology()
    template = brainsynth.resources.Template()
    template = {h: template.load_surface(h, "white")["vertices"] for h in ("lh", "rh")}

    filename_t1w = src_subject_dir / "T1w.nii"
    t1w = nib.load(filename_t1w)
    ras2vox = np.linalg.inv(t1w.affine)
    mni2ras = torch.load(src_subject_dir / "mni305-to-ras.brain.pt").numpy()

    for hemi in ("lh", "rh"):
        # RAS
        v = apply_affine(mni2ras, template[hemi])
        s = cortech.Surface(v[: top0.n_vertices], top0.faces.numpy())
        s.save(dst_subject_dir / f"{hemi}.template")

        # voxel
        # v = apply_affine(ras2vox, v)
        write_resolutions(
            v, resolutions, dst_subject_dir / f"{hemi}.template.{{res}}.pt"
        )


if __name__ == "__main__":
    align_template_to_subjects(sys.argv[1], sys.argv[2])
