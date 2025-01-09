from pathlib import Path
import sys

import nibabel as nib
import torch

from brainnet.mesh.topology import get_recursively_subdivided_topology


if __name__ == "__main__":
    filename = Path(sys.argv[1])
    hemi = sys.argv[2]

    resolution = 6

    top = get_recursively_subdivided_topology(resolution)[-1]
    if hemi == "rh":
        top.reverse_face_orientation()
    faces = top.faces.numpy()
    v = torch.load(filename).numpy()
    nib.freesurfer.write_geometry(filename.parent / f"freesurfer.{filename.stem}", v, faces)
