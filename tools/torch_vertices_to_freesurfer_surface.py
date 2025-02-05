from pathlib import Path
import sys

import nibabel as nib
import torch

from brainnet.mesh.topology import FsAverageTopology
# from brainnet.mesh.topology import StandardTopology


if __name__ == "__main__":
    filename = Path(sys.argv[1])
    hemi = sys.argv[2]
    resolution = int(sys.argv[3])

    top = FsAverageTopology.recursive_subdivision(resolution)[-1]
    # top = StandardTopology.recursive_subdivision(resolution)[-1]
    if hemi == "rh":
        top.reverse_face_orientation()
    faces = top.faces.numpy()
    v = torch.load(filename).numpy()
    nib.freesurfer.write_geometry(filename.parent / f"freesurfer.{filename.stem}", v, faces)
