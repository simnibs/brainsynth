import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nibabel as nib
import nibabel.processing
from nibabel.affines import apply_affine

import numpy as np
import numpy.typing as npt
import scipy.ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import torch

import surfa

from brainsynth import root_dir
from brainsynth.constants.FreeSurferLUT import FreeSurferLUT as lut
from brainsynth.constants import constants, filenames


def prepare_freesurfer_data(
    subject_dir: Path | str,
    out_dir: Path | str,
    write_brain_distance_image: bool = False,
    write_bag_image: bool = False,
    remove_synthseg_image: bool = True,
    additional_images: None | dict = None,
    coreg_additional_images: bool = True,
    n_threads: int = 2,
    force: bool = False,
) -> None:
    """Prepare data for the synthesizer. This will produce the following files

        t1.nii              normalized norm.mgz with modified skull-stripping.
        segmentation.nii    segmentation labels.
        generation.nii      generation labels (includes extracerebral tissues
                            classified using k-means).
        synthseg.nii        result of running SynthSeg on nu.mgz.

        surface.{res}.{hemi}.pt
                            Cortical surfaces resampled to TopoFit template.
                            Note that the surfaces are saved in (scanner) RAS
                            and *not* surface RAS.

    and optionally

        distances.nii       Voxel-wise distances to white and pial surfaces.
        bag.nii


    Parameters
    ----------
    subject_dir : str | Path
        Path to FreeSurfer run for the subject.
    out_dir : str | Path
        Directory to store outputs.
    images : None | dict
        Dictionary containing filenames of additional images to be conformed
        and transformed aligned to RAS. Filenames are keys and values are
        sequence types (e.g., T1) such that files are mapping to the proper
        filename and appropriate processing options are used.
    folding_atlas : str | Path, optional
        Path to folding atlas. If None, we grab
        `lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif` from the average
        directory in FREESURFER_HOME. This is used with `mris_register` to
        register the surface of the subject to the TopoFit template surface.
        (In the deepsurfer package this is called `folding-atlas.tif`.)
    n_threads : int, optional
        Number of threads to use, by default 2.

    Notes
    -----
    Adapted from Eugenio's original MATLAB script by Jesper Duemose Nielsen
    with some modifications.
    """
    LABEL_DTYPE = np.uint16
    IMAGE_DTYPE = np.float32

    subject_dir = Path(subject_dir)
    mri_dir = subject_dir / "mri"
    surf_dir = subject_dir / "surf"

    out_dir = Path(out_dir) / subject_dir.name
    fname_synthseg = out_dir / filenames.default_images.synthseg

    print(f"Saving data to {out_dir}")

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    fs_home = Path(os.environ["FREESURFER_HOME"])

    # if folding_atlas is None:
    folding_atlas = (
        fs_home / "average" / "lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif"
    )
    # else:
    # folding_atlas = Path(folding_atlas)

    sphere_reg_target = root_dir / "resources" / "sphere-reg.srf"
    white_lh_template = surfa.load_mesh(str(root_dir / "resources" / "cortex-int-lh.srf"))
    l2r = surfa.load_affine(str(root_dir / "resources" / "left-to-right.lta"))

    template_mesh = dict(lh=white_lh_template, rh=white_lh_template.transform(l2r))

    # MNI305 to subject mapping

    # We need this transformation to convert the template mesh to subject space

    # This is a voxel to voxel transformation between nu.mgz and mni305.cor.mgz
    # (or whatever it says in the lta file)
    tal = surfa.load_affine(str(mri_dir / "transforms" / "talairach.lta"))
    # Convert to world-to-world trans instead, i.e.,
    #   tal[ras] = vox2ras[target] @ tal[voxel] @ ras2vox[source]
    # and invert so we get mni305[RAS] -> subject[RAS]
    tal = tal.convert(space="world").inv()

    # Get norm.mgz, nu.mgz, aparc+aseg.mgz
    norm = nib.load(mri_dir / "norm.mgz")
    cras = norm.header["Pxyz_c"]  # Surface RAS <--> RAS offset
    norm = align_with_identify_affine(norm)
    norm_dat = norm.get_fdata()
    norm_dat /= np.percentile(norm_dat, 99.9)  # /= 128 in Eugenio's original version
    np.clip(norm_dat, 0, 1, out=norm_dat)

    nu = nib.load(mri_dir / "nu.mgz")
    nu = align_with_identify_affine(nu)
    nu_dat = nu.get_fdata()

    aparc = nib.load(mri_dir / "aparc+aseg.mgz")
    aparc = align_with_identify_affine(aparc)
    aparc_dat = aparc.get_fdata().astype(int)
    aparc_dat[aparc_dat >= 2000] = lut.Right_Cerebral_Cortex
    aparc_dat[aparc_dat >= 1000] = lut.Left_Cerebral_Cortex

    if force or not fname_synthseg.exists():
        print("Running SynthSeg")
        cmd = f"mri_synthseg --i {mri_dir / 'nu.mgz'} --o {fname_synthseg} --threads {n_threads}"  # >/dev/null
        subprocess.run(cmd.split(), check=True)

    synthseg = nibabel.processing.resample_from_to(
        nib.load(fname_synthseg), aparc, order=0
    )
    synthseg_dat = synthseg.get_fdata().astype(int)

    # Prepare APARC labeling
    aparc_dat = correct_aparc(aparc_dat, norm_dat, synthseg_dat)

    # APARC+ASEG -> generation
    remap = aparc_labels_to_generation_labels()
    gen_labels = estimate_extracerebral_tissues(remap[aparc_dat], nu_dat).astype(float)

    # APARC+ASEG -> segmentation
    remap = aparc_labels_to_segmentation_labels()
    seg_labels = remap[aparc_dat]

    surface_files = {
        (h, s): out_dir / f"{h}.{s}" for h in {"lh", "rh"} for s in {"white", "pial"}
    }
    if force or not all(v.exists() for v in surface_files.values()):
        print("Resampling surfaces")
        surfaces = resample_surfaces(
            surf_dir, out_dir, folding_atlas, sphere_reg_target, n_threads
        )
    else:
        print("Reading surfaces")
        surfaces = {}
        for k, v in surface_files.items():
            vertex, faces = nib.freesurfer.read_geometry(v)
            surfaces[k] = Surface(vertex.astype(np.float32), faces.astype(np.int32))

    # Convert surfaces from surface RAS to voxel space
    inv_affine = np.linalg.inv(norm.affine).astype(np.float32)
    for k, v in surfaces.items():
        surfaces[k].vertices = apply_affine(inv_affine, v.vertices + cras)

        # the below is also valid as norm.affine[:3,:3] = I
        # s.vertices += cras - norm.affine[:3, 3]

    # Convert template surfaces from surface RAS to voxel space
    for k,v in template_mesh.items():
        template_mesh[k] = v.transform(tal)
    template_mesh = {
        k: Surface(apply_affine(inv_affine, v.vertices.astype(np.float32) + cras), v.faces) for k,v in template_mesh.items()
    }

    print("Computing distance map  : bag")
    # NOTE Here we use an isotropic structuring element!
    se = scipy.ndimage.generate_binary_structure(3, 3)
    bag_mask = scipy.ndimage.binary_fill_holes(
        scipy.ndimage.binary_opening(
            (gen_labels > lut.Unknown) & (gen_labels < 500), se
        ),
        se,
    )
    bag = scipy.ndimage.distance_transform_edt(~bag_mask)

    print("Computing distance maps : white and pial surfaces")
    dist_maps = compute_distance_maps(norm, surfaces)
    sign_distance_maps_(dist_maps, surface_files, norm)

    # Now that we have the distance maps, we can refine the generation labels
    print("Adding partial volume information to generation labels")
    refine_generation_labels_(gen_labels, seg_labels, dist_maps)

    # refine skull-stripping of norm.mgz using the segmentation
    mask = ~scipy.ndimage.binary_dilation(
        (seg_labels > lut.Unknown) & (seg_labels != lut.CSF)
    )
    norm_dat[mask] = lut.Unknown

    # Write

    keys = [("lh", "white"), ("lh", "pial"), ("rh", "white"), ("rh", "pial")]
    affine = norm.affine.astype(np.float32)

    print("Writing volumes")

    print(filenames.default_images.norm)
    nib.Nifti1Image(norm_dat.astype(IMAGE_DTYPE), affine).to_filename(
        out_dir / filenames.default_images.norm
    )

    print(filenames.default_images.generation)
    nib.Nifti1Image(gen_labels.astype(IMAGE_DTYPE), affine).to_filename(
        out_dir / filenames.default_images.generation
    )

    print(filenames.default_images.segmentation)
    nib.Nifti1Image(seg_labels, affine, dtype=LABEL_DTYPE).to_filename(
        out_dir / filenames.default_images.segmentation
    )

    if write_bag_image:
        print(filenames.default_images.bag)
        nib.Nifti1Image(bag.astype(IMAGE_DTYPE), affine).to_filename(
            out_dir / filenames.default_images.bag
        )

    if write_brain_distance_image:
        print(filenames.default_images.distances)
        nib.Nifti1Image(
            np.stack([dist_maps[k].astype(IMAGE_DTYPE) for k in keys], -1), affine
        ).to_filename(out_dir / filenames.default_images.distances)

    process_additional_images(
        subject_dir,
        additional_images,
        affine,
        out_dir,
        IMAGE_DTYPE,
        coreg_additional_images,
    )

    print("Writing surfaces")

    def n_vertices_at_level(i, base_nf=120):
        """Estimate number of vertices at the ith level of the template
        surface.
        """
        return (base_nf * 4**i + 4) // 2

    for i in constants.SURFACE_RESOLUTIONS:
        nv = n_vertices_at_level(i)
        for h in constants.HEMISPHERES:
            torch.save(
                dict(
                    white=torch.tensor(surfaces[h, "white"].vertices[:nv]),
                    pial=torch.tensor(surfaces[h, "pial"].vertices[:nv]),
                ),
                out_dir / filenames.surfaces[i, h],
            )
            torch.save(
                torch.tensor(template_mesh[h].vertices[:nv]),
                out_dir / filenames.surface_templates[i, h]
            )

    print("Writing auxilliary information")
    bounding_boxes = {
        h: torch.stack(
            (
                torch.from_numpy(surfaces[h, "pial"].vertices).amin(0).floor().int(),
                torch.from_numpy(surfaces[h, "pial"].vertices).amax(0).ceil().int() + 1,
            ),
        )
        for h in constants.HEMISPHERES
    }
    both = torch.stack([v for v in bounding_boxes.values()])
    bounding_boxes["brain"] = torch.stack((both[:, 0].amin(0), both[:, 1].amax(0)))

    labels = dict(
        # apply round as generation may include partial volume labels
        generation=torch.tensor(np.unique(np.round(gen_labels)).astype(np.int32)),
        segmentation=torch.tensor(np.unique(seg_labels).astype(np.int32)),
    )
    torch.save(
        dict(
            affine = torch.from_numpy(affine),
            resolution = torch.from_numpy(affine[:3, :3]).norm(dim=0),
            shape = torch.tensor(norm_dat.shape).to(torch.int),
            bbox = bounding_boxes,
            labels = labels,
        ),
        out_dir / filenames.info,
    )

    if remove_synthseg_image:
        print(f"Removing {fname_synthseg.name}")
        fname_synthseg.unlink()

@dataclass
class Surface:
    vertices: npt.NDArray
    faces: npt.NDArray


def process_additional_images(
    subject_dir, images, affine, out_dir, dtype, apply_coreg=True
):
    """Perform the following steps

        - Register to /mri/orig.mgz (depends on `apply_coreg`)
        - conform
        - align to identity affine.

    """
    if images is None:
        return

    reference = subject_dir / "mri" / "orig.mgz"

    with tempfile.TemporaryDirectory(dir=out_dir) as tempdir:
        d = Path(tempdir)

        tmp_reg = d / "reg.lta"
        tmp_mgz = d / "out.mgz"

        # cmd_coreg = f"mri_easyreg --affine_only --threads {n_threads} --ref {ref} --ref_seg {ref_seg} --flo {{flo}} --flo_seg {{flo_seg}} --flo_reg {{flo_reg}}"

        cmd_coreg = "mri_coreg --mov {source} --ref {reference} --reg {reg}"

        if apply_coreg:
            cmd_conform = "mri_convert --apply_transform {reg} --resample_type cubic --reslice_like {ref} {input} {output}"
        else:
            cmd_conform = "mri_convert --conform {input} {output}"

        for sequence, image in images.items():
            image = Path(image)

            if apply_coreg:
                subprocess.run(
                    cmd_coreg.format(
                        source=image, reference=reference, reg=tmp_reg
                    ).split(),
                    check=True,
                )
                kwargs = dict(
                    reg=tmp_reg, ref=reference, input=image, output=tmp_mgz
                )
            else:
                kwargs = dict(input=image, output=tmp_mgz)

            subprocess.run(cmd_conform.format(**kwargs).split(), check=True)

            conformed = nib.load(tmp_mgz)
            conformed = align_with_identify_affine(conformed)
            conformed_dat = conformed.get_fdata()

            # Contrast-specific processing
            match sequence:
                case "CT":
                    pass
                # case "FLAIR":
                #     pass
                # case "PD":
                #     pass
                # case "T1":
                #     pass
                # case "T2":
                #     pass
                case _:
                    conformed_dat /= np.percentile(conformed_dat, 99.9)
                    np.clip(conformed_dat, 0, 1, out=conformed_dat)

            nib.Nifti1Image(conformed_dat.astype(dtype), affine).to_filename(
                out_dir / getattr(filenames.optional_images, sequence)
            )


def crop_label_volume(data, margin=10, threshold=0):
    if isinstance(margin, int):
        margin = np.repeat(margin, 3)

    coords = np.stack(np.where(data > threshold))
    coords0 = np.maximum(1, coords.min(1) - margin)
    coords1 = np.minimum(data.shape, coords.max(1) + margin) + 1

    return tuple(slice(i, j) for i, j in zip(coords0, coords1))


def align_with_identify_affine(img):
    """Reorient image (and possibly flip dimensions) so as to bring as close as
    possible to having an identity affine transformation matrix.
    """
    perm, flip = (
        nib.orientations.io_orientation(np.linalg.inv(img.affine)).astype(int).T
    )

    # Construct new affine
    affine = np.identity(4)
    affine[:3, :3] = img.affine[:3, perm] * flip
    affine[:3, 3] = img.affine[:3, 3]

    # Adjust image data accordingly
    data = img.get_fdata().transpose(perm)
    shape = data.shape
    for i, f in enumerate(flip):
        if f == -1:
            affine[:3, 3] -= affine[:3, i] * (shape[i] - 1)
            data = np.flip(data, i)

    return nib.Nifti1Image(data, affine)


def paint_in(data, mask):
    """Paint in values of data[mask] with the closest value of the closest voxel.

    Parameters
    ----------
    data : _type_
        _description_
    mask : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    out = data.copy()
    labels, n_labels = scipy.ndimage.label(mask)
    for label in range(1, n_labels + 1):
        cropper = crop_label_volume(labels == label)
        mcrop = mask[cropper]
        dcrop = data[cropper]
        ocrop = out[cropper]

        label_list = np.unique(dcrop[~mcrop])
        D = np.zeros((*dcrop.shape, len(label_list)))

        for i, ll in enumerate(label_list):
            D[..., i] = scipy.ndimage.distance_transform_edt(
                ~((dcrop == ll) & (mcrop == 0))
            )
        idx = D.argmin(3)
        ocrop[mcrop] = label_list[idx][mcrop]
        out[cropper] = ocrop

    return out


def correct_aparc(
    aparc_dat: npt.NDArray, norm_dat: npt.NDArray, synthseg_dat: npt.NDArray
):
    aparc_dat = aparc_dat.copy()

    # Fix cerebellum
    wm = (aparc_dat == lut.Left_Cerebellum_White_Matter) | (
        aparc_dat == lut.Right_Cerebellum_White_Matter
    )
    gm = (aparc_dat == lut.Left_Cerebellum_Cortex) | (
        aparc_dat == lut.Right_Cerebellum_Cortex
    )
    # brain = wm | gm
    threshold = 0.5 * (np.median(norm_dat[wm]) + np.median(norm_dat[gm]))
    aparc_dat[
        aparc_dat == lut.Left_Cerebellum_White_Matter
    ] = lut.Left_Cerebellum_Cortex
    aparc_dat[
        aparc_dat == lut.Right_Cerebellum_White_Matter
    ] = lut.Right_Cerebellum_Cortex

    se = scipy.ndimage.generate_binary_structure(3, 3)

    labels, _ = scipy.ndimage.label(
        (aparc_dat == lut.Left_Cerebellum_Cortex) & (norm_dat > threshold), se
    )
    counts = np.bincount(labels[labels > lut.Unknown])
    idx = counts.argmax()
    aparc_dat[labels == idx] = lut.Left_Cerebellum_White_Matter

    labels, _ = scipy.ndimage.label(
        (aparc_dat == lut.Right_Cerebellum_Cortex) & (norm_dat > threshold), se
    )
    counts = np.bincount(labels[labels > lut.Unknown])
    idx = counts.argmax()
    aparc_dat[labels == idx] = lut.Right_Cerebellum_White_Matter

    # Fix thalamus and basal ganglia with inpainting
    labels = (
        lut.Left_Thalamus,
        lut.Right_Thalamus,
        lut.Left_Putamen,
        lut.Right_Putamen,
        lut.Left_Pallidum,
        lut.Right_Pallidum,
    )
    aparc_dat = paint_in(aparc_dat, np.isin(aparc_dat, labels))
    for label in labels:
        aparc_dat[synthseg_dat == label] = label

    # add CSF
    aparc_dat[(synthseg_dat > lut.Unknown) & (aparc_dat == lut.Unknown)] = lut.CSF

    # add corpus callosum and non_WM_hypointensities
    aparc_dat = paint_in(
        aparc_dat,
        (aparc_dat == lut.non_WM_hypointensities)
        | ((aparc_dat >= lut.Fornix) & (aparc_dat <= lut.CC_Anterior)),
    )

    return aparc_dat


def aparc_labels_to_segmentation_labels(n=1000):
    remap = np.arange(n)  # label goes to itself
    remap[lut.Left_Inf_Lat_Vent] = lut.Left_Lateral_Ventricle
    remap[lut.Left_vessel] = lut.Left_Cerebral_White_Matter
    remap[lut.Left_choroid_plexus] = lut.Left_Lateral_Ventricle
    remap[lut.Right_Inf_Lat_Vent] = lut.Right_Lateral_Ventricle
    remap[lut.Right_vessel] = lut.Right_Cerebral_White_Matter
    remap[lut.Right_choroid_plexus] = lut.Right_Lateral_Ventricle
    return remap


def aparc_labels_to_generation_labels(n=1000):
    """Create a mapping of APARC labels to synthesized labels."""
    remap = np.arange(n)  # label goes to itself

    remap[lut.Left_Inf_Lat_Vent] = lut.Left_Lateral_Ventricle
    remap[lut.Ventricle_3rd] = lut.Left_Lateral_Ventricle
    remap[lut.Ventricle_4th] = lut.Left_Lateral_Ventricle
    remap[lut.Left_Hippocampus] = lut.Left_Cerebral_Cortex
    remap[lut.Left_Amygdala] = lut.Left_Cerebral_Cortex
    remap[lut.CSF] = lut.Left_Lateral_Ventricle
    remap[lut.Left_Accumbens_area] = lut.Left_Caudate

    # map right to left
    remap[lut.Right_Cerebral_White_Matter] = lut.Left_Cerebral_White_Matter
    remap[lut.Right_Cerebral_Cortex] = lut.Left_Cerebral_Cortex
    remap[lut.Right_Lateral_Ventricle] = lut.Left_Lateral_Ventricle
    remap[lut.Right_Inf_Lat_Vent] = lut.Left_Lateral_Ventricle
    remap[lut.Right_Cerebellum_White_Matter] = lut.Left_Cerebellum_White_Matter
    remap[lut.Right_Cerebellum_Cortex] = lut.Left_Cerebellum_Cortex
    remap[lut.Right_Thalamus] = lut.Left_Thalamus
    remap[lut.Right_Caudate] = lut.Left_Caudate
    remap[lut.Right_Putamen] = lut.Left_Putamen
    remap[lut.Right_Pallidum] = lut.Left_Pallidum
    remap[lut.Right_Hippocampus] = lut.Left_Cerebral_Cortex
    remap[lut.Right_Amygdala] = lut.Left_Cerebral_Cortex
    remap[lut.Right_Accumbens_area] = lut.Left_Caudate
    remap[lut.Right_VentralDC] = lut.Left_VentralDC
    remap[lut.Right_vessel] = lut.Left_vessel
    remap[lut.Right_choroid_plexus] = lut.Left_choroid_plexus
    remap[lut.Optic_Chiasm] = lut.Left_Cerebral_White_Matter

    return remap


def make_rolled_mask(mask0, mask1):
    out = np.zeros_like(mask0)
    nn = np.arange(-1, 2)
    for ii in nn:
        for jj in nn:
            for kk in nn:
                out |= np.roll(mask1, (ii, jj, kk), (0, 1, 2))
    out &= mask0
    return out


def refine_generation_labels_(gen_labels, seg_labels, dist_maps):
    """In place update of `gen_labels` with partial volume 'labels' from the
    distance maps. We use PV labels for the white-cortex and cortex-csf
    interface, e.g., 2.25 means 75% label 2 and 25% label 3, etc.
    """
    rho = 1.5

    # Really, this is left *and* right
    WM = gen_labels == lut.Left_Cerebral_White_Matter
    GM = gen_labels == lut.Left_Cerebral_Cortex
    CSF = seg_labels == lut.CSF

    # Compute p(WM)
    pGM = np.clip(
        np.exp(rho * np.minimum(dist_maps["lh", "white"], dist_maps["rh", "white"])),
        0,
        1,
    )
    pWM = 1 - pGM
    M = make_rolled_mask(WM, GM) | make_rolled_mask(GM, WM)
    gen_labels[M] = 2 * pWM[M] + 3 * pGM[M]  # Update generation labels

    # Compute p(GM)
    pCSF = np.clip(
        np.exp(rho * np.minimum(dist_maps["lh", "pial"], dist_maps["rh", "pial"])), 0, 1
    )
    pGM = 1 - pCSF
    M = make_rolled_mask(CSF, GM) | make_rolled_mask(GM, CSF)
    gen_labels[M] = 3 * pGM[M] + 4 * pCSF[M]  # Update generation labels


def resample_surfaces(surf_dir, out_dir, folding_atlas, sphere_reg_target, n_threads):
    """Resample white and pial surfaces from freesurfer to topofit template."""
    surfaces = {"white", "pial"}

    # Generate sphere.reg to TopoFit template
    flip_faces = (0, 2, 1)
    flip_x = np.array([-1, 1, 1])

    sphere_reg = {}
    for h in constants.HEMISPHERES:
        sphere = surf_dir / f"{h}.sphere"
        sphere_reg[h] = out_dir / f"{h}.sphere.reg"
        cmd = f"mris_register -curv {{sphere}} {folding_atlas} {sphere_reg[h]} -threads {n_threads}"
        if h == "lh":
            subprocess.run(cmd.format(sphere=sphere).split())
        elif h == "rh":
            # We need to flip the x axis as the atlas is left hemisphere
            # we do this and save to a temporary directory

            # NOTE Consequently, rh.sphere.reg actually flipped so that it can
            # be used with sphere-reg shipped with topofit/deepsurfer (which is
            # lh)!
            with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
                p = Path(tmpdir)

                # sphere
                out_sphere = p / sphere.name
                v, f, m = nib.freesurfer.read_geometry(sphere, read_metadata=True)
                nib.freesurfer.write_geometry(
                    out_sphere, v * flip_x, f[:, flip_faces], volume_info=m
                )

                # smoothwm
                smoothwm = surf_dir / f"{h}.smoothwm"
                v, f, m = nib.freesurfer.read_geometry(smoothwm, read_metadata=True)
                nib.freesurfer.write_geometry(
                    p / smoothwm.name, v * flip_x, f[:, flip_faces], volume_info=m
                )

                # sulc
                sulc = surf_dir / f"{h}.sulc"
                shutil.copyfile(sulc, p / sulc.name)

                subprocess.run(cmd.format(sphere=out_sphere).split())

    # Resample from freesurfer to topofit using sphere.reg
    target_reg = surfa.load_mesh(str(sphere_reg_target))

    surf_buffer = {}
    for h in constants.HEMISPHERES:
        source_reg = surfa.load_mesh(str(sphere_reg[h]))
        for s in surfaces:
            surf_in = surf_dir / f"{h}.{s}"
            surf_out = out_dir / surf_in.name

            source = surfa.load_mesh(str(surf_in))

            vertices = surfa.sphere.SphericalResamplingBarycentric(
                source_reg, target_reg
            ).sample(source.vertices)
            resampled = surfa.Mesh(vertices, target_reg.faces, geometry=source.geom)

            vertices = resampled.vertices.astype(np.float32)
            faces = resampled.faces.astype(np.int32)
            faces = faces[:, flip_faces] if h == "rh" else faces  # flip if rh

            nib.freesurfer.write_geometry(surf_out, vertices, faces, volume_info=m)
            surf_buffer[(h, s)] = Surface(vertices, faces)

        # Delete as it is non-trivial to use as both are actually registered to
        # lh!
        sphere_reg[h].unlink()

    return surf_buffer


def estimate_extracerebral_tissues(GG, nu_dat):
    # model extracerebral tissues with k-means
    n_clusters = np.random.randint(3, 8)
    mask = GG == lut.Unknown
    x = nu_dat[mask]

    kmeans = KMeans(n_clusters, n_init="auto")
    kmeans.fit(x[:, None])
    labels = kmeans.labels_

    # detect background (smallest average intensity cluster)
    means = np.zeros(n_clusters)
    counts = np.zeros(n_clusters)
    for i in range(n_clusters):
        aux = x[labels == i]
        means[i] = aux.mean()
        counts[i] = len(aux)
    costs = means / counts
    # z = costs.argmin()
    # labels[labels == z] = 0
    # labels[labels > z] -= 1

    remap = np.zeros(n_clusters, dtype=labels.dtype)
    remap[costs.argsort()] = np.arange(n_clusters, dtype=labels.dtype)
    labels = remap[labels]

    GG[mask] = 500 + labels
    GG[GG == 500] = 0

    return GG


# def make_spherical_structure_element(radius, vox_size=None):
#     if vox_size is None:
#         vox_size = np.array([1, 1, 1])
#     d = np.ceil(2*radius / vox_size + 1)
#     d += np.mod(d+1, 2)
#     se = np.zeros(d, dtype=bool)
#     for i in range(d[0])
#         for j in range(d[1])
#             for k in range(d[3])
#                 if (((ii-(d(1)+1)/2)*pixsize(1))^2+((jj-(d(2)+1)/2)*pixsize(2))^2+((kk-(d(3)+1)/2)*pixsize(3))^2)<=r^2
#                     se[ii,jj,kk] = True


def compute_distance_maps(vol, surfaces, max_dist=5):
    # cvx = 0.5 * np.asarray(vol.header.get_zooms()) # center grid on voxels
    grid = np.stack(np.meshgrid(*[np.arange(i) for i in vol.shape], indexing="ij"))
    grid = grid.reshape(3, -1).T  # + cvx

    dist_maps = {}
    for k, s in surfaces.items():
        tree = cKDTree(s.vertices)
        dist, _ = tree.query(grid, distance_upper_bound=max_dist)
        dist_maps[k] = np.nan_to_num(dist, posinf=max_dist).reshape(vol.shape)

    return dist_maps


def sign_distance_maps_(dist_maps, surface_files, ref_vol):
    """Sign distance maps such that negative means inside and positive means
    outside."""
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        for k in dist_maps:
            cmd = f"mris_fill {surface_files[k]} {f.name}"
            subprocess.run(cmd.split(), check=True)
            vol = nib.load(f.name)
            data = (
                nibabel.processing.resample_from_to(
                    vol, ref_vol, order=0
                )  # linear interp. in Eugenio's version
                .get_fdata()
                .astype(int)
            )
            dist_maps[k][data > 0.5] *= -1


def parse_args(argv):
    parser = argparse.ArgumentParser("Prepare FreeSurfer Data")
    parser.add_argument("subject-dir", help="Path to FreeSurfer run of subject.")
    parser.add_argument(
        "out-dir",
        help=(
            "Top level directory in which to save the prepared data. "
            "Specifically, if subject-dir is /path/to/sub-01 and out-dir is "
            "/path/out/, then the data will be saved in /path/out/sub-01."
        ),
    )
    parser.add_argument(
        "--threads", type=int, default=2, help="Maximum number of threads to use."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force creation of surfaces even if they already exist.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    prepare_freesurfer_data(
        getattr(args, "subject-dir"),
        getattr(args, "out-dir"),
        n_threads=args.threads,
        force=args.force,
    )
