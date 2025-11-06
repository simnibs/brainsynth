from pathlib import Path
import sys

import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import torch

import cortech

import brainsynth


def write_resolutions(v, resolutions, name):
    v = torch.tensor(v.astype(np.float32))
    for r in resolutions:
        vv = v[: nv[r]].clone()
        torch.save(vv, str(name).format(res=r))


def n_vertices_at_level(i, base_nf=120):
    """Estimate number of vertices at the ith level of the template
    surface.
    """
    return (base_nf * 4**i + 4) // 2


if __name__ == "__main__":
    src_sub_dir = Path(sys.argv[1])
    dst_sub_dir = Path(sys.argv[2])

    template = brainsynth.resources.Template()
    template = {
        h: cortech.Sphere(**template.load_surface(h, "sphere.reg"))
        for h in ("lh", "rh")
    }

    nv = {i: n_vertices_at_level(i) for i in range(6 + 1)}

    write_original_as_fs = True
    write_resampled_as_fs = True
    write_original_as_torch = False
    write_resampled_as_torch = True

    resolutions = [4, 5, 6]

    for hemi in ("lh", "rh"):
        t1w = nib.load(src_sub_dir / "T1w.nii")
        ras2vox = np.linalg.inv(t1w.affine)

        white = cortech.Surface.from_file(src_sub_dir / f"{hemi}.white")
        white.to_scanner_ras()
        white.faces = white.faces.astype(int)

        pial = cortech.Surface.from_file(src_sub_dir / f"{hemi}.pial")
        pial.to_scanner_ras()
        pial.faces = pial.faces.astype(int)

        # resample subject to template
        reg = cortech.Sphere.from_file(src_sub_dir / f"{hemi}.sphere.reg")
        reg.faces = reg.faces.astype(int)
        reg.project(template[hemi])

        # v = reg.resample(reg.vertices)
        # if write_resampled_as_fs:
        #     s = cortech.Surface(100.0 * v, template[hemi].faces)
        #     s.save(dst_sub_dir / f"{hemi}.sphere.reg.resample")
        # if write_resampled_as_torch:
        #     write_resolutions(
        #         v,
        #         resolutions,
        #         dst_sub_dir / f"{hemi}.sphere.reg.resample.{{res}}.pt",
        #     )

        v = reg.resample(white.vertices)
        if write_resampled_as_fs:
            # Surface is written in *scanner RAS*
            s = cortech.Surface(v, template[hemi].faces)
            s.save(dst_sub_dir / f"{hemi}.white.resample")
        if write_resampled_as_torch:
            write_resolutions(
                # apply_affine(ras2vox, v),
                v,
                resolutions,
                dst_sub_dir / f"{hemi}.white.resample.{{res}}.pt",
            )

        v = reg.resample(pial.vertices)
        if write_resampled_as_fs:
            # Surface is written in *scanner RAS*
            s = cortech.Surface(v, template[hemi].faces)
            s.save(dst_sub_dir / f"{hemi}.pial.resample")
        if write_resampled_as_torch:
            write_resolutions(
                # apply_affine(ras2vox, v),
                v,
                resolutions,
                dst_sub_dir / f"{hemi}.pial.resample.{{res}}.pt",
            )
