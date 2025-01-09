import os
from pathlib import Path

import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import subprocess

import torch
import surfa

from brainsynth.constants import SURFACE
from brainsynth.prepare_freesurfer_data import Surface
from brainsynth import resources_dir


# ref_img = "/home/jesperdn/INN_JESPER/nobackup/projects/brainnet/4dk/maps/OASIS3.sub-0002.T1w.nii"
# fs_sub_dir = "/home/jesperdn/INN_JESPER/nobackup/projects/brainnet/OASIS3/derivatives/freesurfer741/sub-0002"
# out_dir = "/mnt/projects/CORTECH/nobackup/synth_train_surfaces"


# def n_vertices_at_level(i, base_nf=120):
#     """Estimate number of vertices at the ith level of the template
#     surface.
#     """
#     return (base_nf * 4**i + 4) // 2



# def prep_mgh_data_surface(ref_img, fs_sub_dir, out_dir, n_threads=4):
#     # FS must be sourced

#     # example:
#     # ref_img = "/home/jesperdn/INN_JESPER/nobackup/projects/brainnet/4dk/maps/HCP.sub-001.generation_labels.nii"
#     # fs_sub_dir = "/mnt/projects/INN/oula/nobackup/for_jesper/sub-001"
#     # out_dir = "/mnt/scratch/personal/jesperdn/test"

#     # with whatever prefix you guys use, e.g., dataset and subject id
#     # fname_info = "info.pt"

#     fs_sub_dir = Path(fs_sub_dir)
#     mri_dir = fs_sub_dir / "mri"
#     surf_dir = fs_sub_dir / "surf"
#     out_dir = Path(out_dir)

#     norm = nib.load(mri_dir / "norm.mgz")
#     cras = norm.header["Pxyz_c"]  # Surface RAS <--> RAS offset

#     aligned_image = nib.load(ref_img)

#     fs_home = Path(os.environ["FREESURFER_HOME"])

#     folding_atlas = (
#         fs_home / "average" / "lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif"
#     )

#     sphere_reg_target = root_dir / "resources" / "sphere-reg.srf"
#     white_lh_template = surfa.load_mesh(str(root_dir / "resources" / "cortex-int-lh.srf"))
#     l2r = surfa.load_affine(str(root_dir / "resources" / "left-to-right.lta"))

#     template_mesh = dict(lh=white_lh_template, rh=white_lh_template.transform(l2r))

#     # We need this transformation to convert the template mesh to subject space
#     tal = surfa.load_affine(str(mri_dir / "transforms" / "talairach.lta"))
#     tal = tal.convert(space="world").inv()

#     # resample fs surfaces to topofit template
#     surfaces = resample_surfaces(
#         surf_dir, out_dir, folding_atlas, sphere_reg_target, n_threads
#     )

#     # Convert surfaces from surface RAS to voxel space
#     inv_affine = np.linalg.inv(aligned_image.affine).astype(np.float32)
#     for k, v in surfaces.items():
#         surfaces[k].vertices = apply_affine(inv_affine, v.vertices + cras)

#     # Convert template surfaces from surface RAS to voxel space
#     for k,v in template_mesh.items():
#         template_mesh[k] = v.transform(tal)
#     template_mesh = {
#         k: Surface(apply_affine(inv_affine, v.vertices.astype(np.float32) + cras), v.faces) for k,v in template_mesh.items()
#     }

#     for i in SURFACE.resolutions: # constants.SURFACE_RESOLUTIONS:
#         files = SURFACE.get_files(SURFACE.hemispheres, SURFACE.types, i, "target-fsavg")
#         nv = n_vertices_at_level(i)
#         for h in SURFACE.hemispheres:
#             for s in SURFACE.types:
#                 torch.save(
#                     torch.tensor(surfaces[h, s].vertices[:nv]),
#                     out_dir / files[h, s],
#                 )

#     for i in SURFACE.resolutions:
#         files = SURFACE.get_files(SURFACE.hemispheres, None, i, "template-fsavg")
#         nv = n_vertices_at_level(i)
#         for h in SURFACE.hemispheres:
#             torch.save(
#                 torch.tensor(template_mesh[h].vertices[:nv]),
#                 out_dir / files[h]
#             )


#     # write auxilliary information
#     # bounding_boxes = {
#     #     h: torch.stack(
#     #         (
#     #             torch.from_numpy(surfaces[h, "pial"].vertices).amin(0).floor().int(),
#     #             torch.from_numpy(surfaces[h, "pial"].vertices).amax(0).ceil().int() + 1,
#     #         ),
#     #     )
#     #     for h in SURFACE.hemispheres
#     # }
#     # both = torch.stack([v for v in bounding_boxes.values()])
#     # bounding_boxes["brain"] = torch.stack((both[:, 0].amin(0), both[:, 1].amax(0)))

#     # labels = dict(
#     #     # apply round as generation may include partial volume labels
#     #     generation=torch.tensor(np.unique(np.round(gen_labels)).astype(np.int32)),
#     #     segmentation=torch.tensor(np.unique(seg_labels).astype(np.int32)),
#     # )
#     # torch.save(
#     #     dict(
#     #         # affine = torch.from_numpy(affine),
#     #         resolution = torch.from_numpy(aligned_image.affine[:3, :3]).norm(dim=0).float(),
#     #         shape = torch.tensor(aligned_image.shape[:3]).to(torch.int),
#     #         bbox = bounding_boxes,
#     #         # labels = labels,
#     #     ),
#     #     out_dir / fname_info,
#     # )

def n_vertices_at_level(i):
    """Estimate number of vertices at the ith level of the template
    surface.
    """
    return 10 * 4**i + 2



def resample_surfaces_fsavg(surf_dir, out_dir, n_threads):
    """Resample white and pial surfaces from freesurfer to fsaverage template."""

    fs_home = Path(os.environ["FREESURFER_HOME"])

    # Compute spherical registrations
    sphere_reg = {}
    meta = {}
    for h in SURFACE.hemispheres:
        folding_atlas = (
            fs_home / "average" / f"{h}.average.curvature.filled.buckner40.tif"
        )
        sphere = surf_dir / f"{h}.sphere"
        sphere_reg[h] = out_dir / f"{h}.sphere.reg"

        _, _, meta[h] = nib.freesurfer.read_geometry(sphere, read_metadata=True)

        cmd = f"mris_register -curv {sphere} {folding_atlas} {sphere_reg[h]} -threads {n_threads}"

        subprocess.run(cmd.split())

    # Apply registrations (resample original FS surface to fsaverage)
    surf_buffer = {}
    for h in SURFACE.hemispheres:
        source_reg = surfa.load_mesh(str(sphere_reg[h]))
        target_reg = surfa.load_mesh(
            str(fs_home / "subjects" / "fsaverage" / "surf" / f"{h}.sphere")
        )
        for s in SURFACE.types:
            surf_in = surf_dir / f"{h}.{s}"
            surf_out = out_dir / surf_in.name

            source = surfa.load_mesh(str(surf_in))

            vertices = surfa.sphere.SphericalResamplingBarycentric(
                source_reg, target_reg
            ).sample(source.vertices)
            resampled = surfa.Mesh(vertices, target_reg.faces, geometry=source.geom)

            vertices = resampled.vertices.astype(np.float32)
            faces = resampled.faces.astype(np.int32)

            nib.freesurfer.write_geometry(
                surf_out, vertices, faces, volume_info=meta[h]
            )
            surf_buffer[(h, s)] = Surface(vertices, faces)

    return surf_buffer


def prep_mgh_data_surface_fsavg(
        ref_img, fs_sub_dir, out_dir, n_threads=4, write_fs_surfaces: bool = False
    ):
    # FS must be sourced

    # example:
    # ref_img = "/home/jesperdn/INN_JESPER/nobackup/projects/brainnet/4dk/maps/HCP.sub-001.generation_labels.nii"
    # fs_sub_dir = "/mnt/projects/INN/oula/nobackup/for_jesper/sub-001"
    # out_dir = "/mnt/scratch/personal/jesperdn/test"

    # with whatever prefix you guys use, e.g., dataset and subject id
    # fname_info = "info.pt"

    n_res = 7
    resolutions = range(n_res + 1)

    if write_fs_surfaces:
        from brainnet.mesh.topology import get_fsaverage_topology
        tops = get_fsaverage_topology(n_res)

    fs_sub_dir = Path(fs_sub_dir)
    mri_dir = fs_sub_dir / "mri"
    surf_dir = fs_sub_dir / "surf"
    out_dir = Path(out_dir)

    print(f"Reading data from : {fs_sub_dir}")
    print(f"Writing data to   : {out_dir}")

    norm = nib.load(mri_dir / "norm.mgz")
    cras = norm.header["Pxyz_c"]  # Surface RAS <--> RAS offset

    aligned_image = nib.load(ref_img)
    inv_affine = np.linalg.inv(aligned_image.affine).astype(np.float32)

    # TARGET SURFACE
    # ==============
    print("Resampling target surfaces")

    # resample subject surfaces to fsaverage
    surfaces = resample_surfaces_fsavg(surf_dir, out_dir, n_threads)

    # Convert surfaces from surface RAS to voxel space
    for k, v in surfaces.items():
        surfaces[k].vertices = apply_affine(inv_affine, v.vertices + cras)

    for i in resolutions:  # constants.SURFACE_RESOLUTIONS:
        files = SURFACE.get_files(SURFACE.hemispheres, SURFACE.types, i, "target-fsavg")
        nv = n_vertices_at_level(i)
        for (h, s), v in surfaces.items():
            torch.save(torch.tensor(v.vertices[:nv]), out_dir / files[h, s])
            if write_fs_surfaces:
                nib.freesurfer.write_geometry(
                    str(out_dir / (files[h,s] + "_fs")),
                    v.vertices[:nv],
                    tops[i].faces.numpy(),
                )

    print("Transforming template surfaces")

    # INITIAL SURFACE (template)
    # ==========================
    # Convert template surfaces from surface RAS to voxel space
    template_mesh = {
        h: surfa.load_mesh(str(resources_dir / f"{h}.white.smooth"))
        for h in SURFACE.hemispheres
    }

    # We need this transformation to convert the template mesh to subject space
    tal = surfa.load_affine(str(mri_dir / "transforms" / "talairach.lta"))
    tal = tal.convert(space="world").inv()

    # to subject RAS
    for k, v in template_mesh.items():
        template_mesh[k] = v.transform(tal)

    # to subject voxel space
    template_mesh = {\
        k: Surface(
            apply_affine(inv_affine, v.vertices.astype(np.float32) + cras), v.faces
        )
        for k, v in template_mesh.items()
    }

    for i in resolutions:
        files = SURFACE.get_files(SURFACE.hemispheres, None, i, "template-fsavg")
        nv = n_vertices_at_level(i)
        for h, v in template_mesh.items():
            torch.save(torch.tensor(v.vertices[:nv]), out_dir / files[h])
            if write_fs_surfaces:
                nib.freesurfer.write_geometry(
                    str(out_dir / (files[h] + "_fs")),
                    v.vertices[:nv],
                    tops[i].faces.numpy(),
                )


def make_smooth_template_surface():
    from cortech import Surface

    for h in ("lh", "rh"):
        v,f,m = nib.freesurfer.read_geometry(
            f"/mnt/depot64/freesurfer/freesurfer.7.4.1/subjects/fsaverage/surf/{h}.white", read_metadata=True
        )
        s = Surface(v,f)
        s = Surface(s.gaussian_smooth(a=1.0, n_iter=100), f)
        # nib.freesurfer.write_geometry("s0", s0.vertices, s0.faces)
        nib.freesurfer.write_geometry(f"/mrhome/jesperdn/repositories/brainsynth/brainsynth/resources/{h}.white.smooth", s.vertices, s.faces, volume_info=m)



# v = []
# for i in range(8):
#     a,b,c = nib.freesurfer.read_geometry(
#         f"/mnt/depot64/freesurfer/freesurfer.7.4.1/average/surf/lh.sphere.ico{i}.reg", read_metadata=True
#     )
#     v.append(a)

# d = []
# for i in range(8):
#     tree = cKDTree(v[7][:len(v[i])])
#     a,b=tree.query(v[i])
#     d.append(b)
#     # print(i, b)

#     x = torch.tensor(d)[tops[i].faces]
#     print(x)

# pv.make_tri_mesh(v[7][:tops[i].n_vertices],x.numpy()).plot()

# vx,fx = nib.freesurfer.read_geometry("/mrhome/jesperdn/repositories/brainsynth/brainsynth/resources/rh.white.smooth")

# for i in range(8):
#     m = pv.make_tri_mesh(vx[:tops[i].n_vertices],tops[i].faces.numpy())
#     m.save(f"test_{i}.vtk")
#     x = torch.tensor(d[i])[tops[i].faces]
#     m = pv.make_tri_mesh(vx[:tops[i].n_vertices],x.numpy())
#     m.save(f"testx_{i}.vtk")

#     .plot()
