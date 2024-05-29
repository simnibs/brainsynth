import torch

from brainsynth.constants.constants import mapped_input_keys as mikeys
from brainsynth.config.augmentation_configuration import AugmentationConfiguration
import brainsynth.config.pipelines
from brainsynth.transforms import EnsureDevice, InputSelectorError, Pipeline


class Synthesizer(torch.nn.Module):
    def __init__(self, config: AugmentationConfiguration):
        super().__init__()
        self.config = config
        self.ensure_device = EnsureDevice(self.config.device)

        self.pipeline = getattr(brainsynth.config.pipelines, config.pipeline)()

    def execute(self, pipelines: dict, mapped_inputs: dict, out: dict | None = None):
        out = out if out is not None else {}
        for k, pipeline in pipelines.items():
            try:
                if isinstance(pipeline, Pipeline):
                    out[k] = pipeline(mapped_inputs)
                else:
                    out[k] = pipeline()
            except InputSelectorError:
                # could not find input for this pipeline
                pass
        return out

    def forward(
        self,
        images: dict[str, torch.Tensor],
        surfaces: dict[str, dict[str, torch.Tensor]],
        initial_vertices: dict[str, torch.Tensor]
    ):
        mapped_inputs = {
            mikeys.image: self.ensure_device(images),
            mikeys.initial_vertices: self.ensure_device(initial_vertices),
            mikeys.surface: self.ensure_device(surfaces),
            mikeys.state: {},
        }

        # Build the pipelines
        state_pipeline, output_pipeline = self.pipeline.build(self.config)

        # Set the state
        mapped_inputs["state"] = self.execute(
            state_pipeline, mapped_inputs, mapped_inputs[mikeys.state]
        )

        # Generate the output
        return self.execute(output_pipeline, mapped_inputs)

from pathlib import Path
import nibabel as nib
import numpy as np
from brainnet.mesh.topology import get_recursively_subdivided_topology
top = get_recursively_subdivided_topology(5)

config = AugmentationConfiguration(
    device="cpu",
    alternative_images=("T1w", "T2w"),
    out_center_str="lh",
    segmentation_labels="brainseg",
)
self = Synthesizer(config)

root = Path("/mnt/projects/CORTECH/nobackup/training_data/")
sub="sub-0003"
images=dict(
    T1w=torch.from_numpy(np.transpose(nib.load(root / f"ABIDE.{sub}.T1w.nii").get_fdata().astype(np.float32), (3,0,1,2))),
    T1w_mask=torch.from_numpy(nib.load(root / f"ABIDE.{sub}.T1w.defacingmask.nii").get_fdata().astype(np.int32)[None]),
    # T2w=torch.rand((256, 256, 256)),
    generation=torch.from_numpy(np.transpose(nib.load(root / f"ABIDE.{sub}.generation_labels.nii").get_fdata().astype(np.int32), (3,0,1,2))),
    segmentation=torch.from_numpy(np.transpose(nib.load(root / f"ABIDE.{sub}.brainseg.nii").get_fdata().astype(np.int32), (3,0,1,2))),
    # FLAIR=torch.rand((10,10,10))
)
surfaces=dict(
    lh=dict(
        white=torch.load(root / f"ABIDE.{sub}.surf_dir" / "lh.white.5.target.pt"),
        pial=torch.load(root / f"ABIDE.{sub}.surf_dir" / "lh.white.5.target.pt"),
    ))
initial_vertices=dict(
    lh=torch.load(root / f"ABIDE.{sub}.surf_dir" / "lh.0.template.pt"))

a = self(images, surfaces, initial_vertices)

affine = np.eye(4)
nib.Nifti1Image(a["T1w"][0].numpy(), affine).to_filename("/home/jesperdn/nobackup/T1w.nii")
nib.freesurfer.write_geometry(
    "/home/jesperdn/nobackup/lh.white",
    a["surface"]["lh"]["white"].numpy(), top[5].faces.numpy(),
)

#nib.Nifti1Image(out["T1w"][0].numpy(), affine).to_filename("/home/jesperdn/nobackup/T1w_2.nii")
