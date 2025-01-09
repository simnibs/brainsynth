import numpy as np
import torch
from pathlib import Path
import surfa
import nibabel as nib
from nibabel.affines import apply_affine

from brainsynth import root_dir
from brainsynth.constants import SURFACE


def n_vertices_at_level(i, base_nf=120):
    """Estimate number of vertices at the ith level of the template
    surface.
    """
    return (base_nf * 4**i + 4) // 2

mni305_to_mni152 = np.array([[0.9975 ,  -0.0073 ,   0.0176 ,  -0.0429],
          [0.0146 ,   1.0009 ,  -0.0024  ,  1.5496],
          [-0.0130,   -0.0093,    0.9971  ,  1.1840], [0,0,0,1]])



white_lh_template = surfa.load_mesh(str(root_dir / "resources" / "cortex-int-lh.srf"))
l2r = surfa.load_affine(str(root_dir / "resources" / "left-to-right.lta"))

template_mesh = dict(lh=white_lh_template, rh=white_lh_template.transform(l2r))


d = Path("/mnt/projects/CORTECH/nobackup/training_data/full")
for ds_dir in d.glob("*"):
    print(ds_dir.stem)
    for sub_dir in ds_dir.glob("sub-*"):
        print(sub_dir.stem)
        t1w = nib.load(sub_dir/"T1w.nii")
        mni2ras = np.loadtxt(sub_dir / "mni152_affine_forward.txt")
        inv_affine = np.linalg.inv(t1w.affine).astype(np.float32)

        for k,v in template_mesh.items():

            x = apply_affine(inv_affine @ mni2ras @ mni305_to_mni152, v.convert("world").vertices).astype(np.float32)

            for i in (0, ): #1, 2, 3, 4, 5, 6):
                nv = n_vertices_at_level(i)
                torch.save(
                    torch.tensor(x[:nv]),
                    sub_dir / SURFACE.files.template_niftyreg[k, i]
                )
                # print(sub_dir / SURFACE.files.template_niftyreg[k, i])
