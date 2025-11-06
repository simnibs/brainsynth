from pathlib import Path
import sys

import nibabel as nib

from brainsynth.prepare_freesurfer_data import align_with_identity_affine


if __name__ == "__main__":
    fs_subject_dir = Path(sys.argv[1])
    filename_out = Path(sys.argv[2])

    img = nib.load(fs_subject_dir / "mri" / "orig.mgz")
    img = align_with_identity_affine(img)
    img.to_filename(filename_out)
