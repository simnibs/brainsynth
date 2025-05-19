from brainsynth.prepare_freesurfer_data import process_additional_images
from pathlib import Path
import tqdm

import numpy as np

if __name__ == "__main__":
    IMAGE_DTYPE = np.float32

    derivatives_dir = Path("/mrhome/jesperdn/INN_JESPER/nobackup/projects/brainnet/OASIS3/derivatives/freesurfer741")
    out_dir = Path("/mnt/scratch/personal/jesperdn/datasets/OASIS3")


    for subject_dir in tqdm.tqdm(sorted(derivatives_dir.glob("sub*"))):

        additional_images = dict(T1 = derivatives_dir / subject_dir.name / "mri" / "orig.mgz")

        process_additional_images(
            subject_dir,
            additional_images,
            out_dir / subject_dir.name,
            IMAGE_DTYPE,
            apply_coreg=False,
        )
