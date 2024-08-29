from pathlib import Path
import sys

import nibabel as nib
import torch

from brainnet.mesh.topology import get_recursively_subdivided_topology


if __name__ == "__main__":
    filename = Path(sys.argv[1])

    resolution = 5

    faces = get_recursively_subdivided_topology(resolution)[-1].faces.numpy()
    v = torch.load(filename).numpy()
    nib.freesurfer.write_geometry(filename.parent / f"freesurfer.{filename.stem}", v, faces)