from pathlib import Path
import subprocess
import sys
import tempfile

import nibabel as nib

# import nibabel.processing
import numpy as np

from brainsynth.prepare_freesurfer_data import align_with_identity_affine

fs_subject_dir = "/mnt/projects/CORTECH/nobackup/jesper/IXI/freesurfer/IXI002"
filename_in = "/mnt/projects/CORTECH/nobackup/jesper/IXI/T2/IXI002-Guys-0828-T2.nii.gz"
filename_out = "/mnt/scratch/personal/jesperdn/IXI/IXI002/T2w.nii"


def process_extra_image(
    fs_subject_dir, filename_in, filename_out, dtype=np.uint8, apply_coreg=True
):
    """Perform the following steps

    - Register to /mri/orig.mgz (depends on `apply_coreg`)
    - conform
    - align to identity affine.

    """

    reference = Path(fs_subject_dir) / "mri" / "orig.mgz"
    filename_in = Path(filename_in)
    filename_out = Path(filename_out)

    with tempfile.TemporaryDirectory(dir=filename_out.parent) as tempdir:
        d = Path(tempdir)

        tmp_reg = d / "reg.lta"
        tmp_mgz = d / "out.mgz"

        cmd_coreg = "mri_coreg --mov {source} --ref {reference} --reg {reg}"

        if apply_coreg:
            cmd_conform = "mri_convert --apply_transform {reg} --resample_type cubic --reslice_like {ref} {input} {output}"
        else:
            cmd_conform = "mri_convert --conform {input} {output}"

        if apply_coreg:
            subprocess.run(
                cmd_coreg.format(
                    source=filename_in, reference=reference, reg=tmp_reg
                ).split(),
                check=True,
            )
            kwargs = dict(reg=tmp_reg, ref=reference, input=filename_in, output=tmp_mgz)
        else:
            kwargs = dict(input=filename_in, output=tmp_mgz)

        subprocess.run(cmd_conform.format(**kwargs).split(), check=True)

        conformed = nib.load(tmp_mgz)
        conformed = align_with_identity_affine(conformed)
        conformed.to_filename(filename_out)


if __name__ == "__main__":
    fs_subject_dir = Path(sys.argv[1])
    filename_in = Path(sys.argv[2])
    filename_out = Path(sys.argv[3])

    process_extra_image(fs_subject_dir, filename_in, filename_out)
