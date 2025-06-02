from pathlib import Path
import sys

import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import torch
from tqdm import tqdm

from brainsynth.constants import FREESURFER_VOLUME_INFO
from brainsynth.resources import load_cortical_template
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


def align_template_to_subjects(DATASET_DIR, datasets=None):
    DATASET_DIR = Path(DATASET_DIR)

    resolutions = [0]
    top0 = DeepSurferTopology()
    template = dict(
        lh=load_cortical_template("lh", "white")["vertices"],
        rh=load_cortical_template("rh", "white")["vertices"],
    )

    for ds_dir in sorted(DATASET_DIR.glob("*")):
        if ds_dir.name != "ADNI-GO2":
            continue
        print(ds_dir.name)
        for sub_dir in tqdm(sorted(ds_dir.glob("sub-*"))):
            filename_t1w = sub_dir / "T1w.nii"
            t1 = nib.load(filename_t1w)
            ras2vox = np.linalg.inv(t1.affine)
            mni2ras = torch.load(sub_dir / "mni305-to-ras.brain.pt").numpy()
            FREESURFER_VOLUME_INFO["volume"] = t1.shape
            FREESURFER_VOLUME_INFO["filename"] = str(filename_t1w)
            for hemi in ("lh", "rh"):
                v = apply_affine(mni2ras, template[hemi])
                nib.freesurfer.write_geometry(
                    sub_dir / f"{hemi}.template",
                    v[: top0.n_vertices],
                    top0.faces.numpy(),
                    volume_info=FREESURFER_VOLUME_INFO,
                )
                v = apply_affine(ras2vox, v)
                write_resolutions(
                    v, resolutions, sub_dir / f"{hemi}.template.{{res}}.pt"
                )


if __name__ == "__main__":
    align_template_to_subjects(sys.argv[1])

# for d in ./*; do; for f in $d/surf/*; do echo $f; done; done
