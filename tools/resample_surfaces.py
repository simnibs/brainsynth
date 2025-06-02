from pathlib import Path
import sys

import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import scipy.io
import torch

from cortech import Surface, SphericalRegistration

from brainsynth.resources import load_cortical_template


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

    template = dict(
        lh=SphericalRegistration(**load_cortical_template("lh", "sphere.reg")),
        rh=SphericalRegistration(**load_cortical_template("rh", "sphere.reg")),
    )

    nv = {i: n_vertices_at_level(i) for i in range(6 + 1)}

    write_original_as_fs = True
    write_resampled_as_fs = True
    write_original_as_torch = False
    write_resampled_as_torch = True

    resolutions = [4, 5, 6]

    # out_dir = Path("/mnt/scratch/personal/jesperdn/surface_data")

    # m = scipy.io.loadmat(subject_dir / "surfaces.mat")

    for hemi in ("lh", "rh"):
        # out_subject = out_dir.joinpath(*subject_dir.parts[-2:])

        # h0 = hemi[0]
        # if not out_subject.exists():
        #     out_subject.mkdir(parents=True)

        t1 = nib.load(src_sub_dir / "T1w.nii")
        ras2vox = np.linalg.inv(t1.affine)

        # # White
        # v = m[f"V{h0}w"]
        # f = m[f"F{h0}w"]
        # white = Surface(v, f)
        # if write_original_as_fs:
        #     nib.freesurfer.write_geometry(out_subject / f"{hemi}.white", v, f)
        # if write_original_as_torch:
        #     torch.save(
        #         torch.tensor(apply_affine(ras2vox, v).astype(np.float32)),
        #         out_subject / f"{hemi}.white.pt",
        #     )

        # # Pial
        # v = m[f"V{h0}p"]
        # f = m[f"F{h0}p"]
        # pial = Surface(v, f)
        # if write_original_as_fs:
        #     nib.freesurfer.write_geometry(out_subject / f"{hemi}.pial", v, f)
        # if write_original_as_torch:
        #     torch.save(
        #         torch.tensor(apply_affine(ras2vox, v).astype(np.float32)),
        #         out_subject / f"{hemi}.pial.pt",
        #     )

        # # sphere.reg
        # v = m[f"R{h0}"]
        # f = m[f"F{h0}w"]
        # if write_original_as_fs:
        #     nib.freesurfer.write_geometry(out_subject / f"{hemi}.sphere.reg", v, f)
        # if write_original_as_torch:
        #     v = v / np.linalg.norm(v, axis=1, keepdims=True)
        #     torch.save(
        #         torch.tensor(v.astype(np.float32)),
        #         out_subject / f"{hemi}.sphere.reg.pt",
        #     )

        white = Surface.from_freesurfer(src_sub_dir / f"{hemi}.white")
        pial = Surface.from_freesurfer(src_sub_dir / f"{hemi}.pial")
        # surface RAS to scanner RAS
        white.vertices += white.metadata.geometry.cras
        pial.vertices += pial.metadata.geometry.cras

        white.faces = white.faces.astype(int)
        pial.faces = pial.faces.astype(int)

        v, f = nib.freesurfer.read_geometry(src_sub_dir / f"{hemi}.sphere.reg")

        # resample subject to template
        reg = SphericalRegistration(v, f.astype(int))
        reg.project(template[hemi])

        v = reg.resample(reg.vertices)
        if write_resampled_as_fs:
            nib.freesurfer.write_geometry(
                dst_sub_dir / f"{hemi}.sphere.reg.resample",
                100.0 * v,
                template[hemi].faces,
            )
        if write_resampled_as_torch:
            write_resolutions(
                v,
                resolutions,
                dst_sub_dir / f"{hemi}.sphere.reg.resample.{{res}}.pt",
            )

        v = reg.resample(white.vertices)
        if write_resampled_as_fs:
            # Surface is written in *scanner RAS*
            nib.freesurfer.write_geometry(
                dst_sub_dir / f"{hemi}.white.resample", v, template[hemi].faces
            )
        if write_resampled_as_torch:
            write_resolutions(
                apply_affine(ras2vox, v),
                resolutions,
                dst_sub_dir / f"{hemi}.white.resample.{{res}}.pt",
            )

        v = reg.resample(pial.vertices)
        if write_resampled_as_fs:
            # Surface is written in *scanner RAS*
            nib.freesurfer.write_geometry(
                dst_sub_dir / f"{hemi}.pial.resample", v, template[hemi].faces
            )
        if write_resampled_as_torch:
            write_resolutions(
                apply_affine(ras2vox, v),
                resolutions,
                dst_sub_dir / f"{hemi}.pial.resample.{{res}}.pt",
            )
