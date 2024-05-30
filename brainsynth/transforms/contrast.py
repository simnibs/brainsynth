import torch

from brainsynth.constants.FreeSurferLUT import FreeSurferLUT as lut
from brainsynth.transforms.base import BaseTransform, RandomizableTransform
from brainsynth.transforms.filters import GaussianSmooth


class SynthesizeIntensityImage(BaseTransform):
    def __init__(
        self,
        mu_offset: float,
        mu_scale: float,
        sigma_offset: float,
        sigma_scale: float,
        add_partial_volume: bool = True,
        photo_mode: bool = False,
        min_cortical_contrast: float = 25.0,
        device: None | torch.device = None,
    ):
        super().__init__(device)

        self.add_partial_volume = add_partial_volume
        self.photo_mode = photo_mode
        self.min_cortical_contrast = min_cortical_contrast

        self.sample_gaussians(mu_offset, mu_scale, sigma_offset, sigma_scale)

    def sample_gaussians(self, mu_offset, mu_scale, sigma_offset, sigma_scale):
        # extracerebral tissue (from k-means clustering) are added as
        # 500 + label so we need to take this into account, hence the value of
        # n
        n = 500 + 10
        mu = mu_offset + mu_scale * torch.rand(n, device=self.device)
        sigma = sigma_offset + sigma_scale * torch.rand(n, device=self.device)

        scale_factor = mu / (mu_offset + mu_scale)
        sigma *= scale_factor

        # set the background to zero every once in a while (or always in photo mode)
        if self.photo_mode or torch.rand(1, device=self.device) < 0.5:
            mu[0] = 0

        # Ensure that there is *some* contrast between white and gray matter
        # and gray matter and CSF.
        #   20 seems OK
        #   50 always gives nice contrast everytime
        if self.min_cortical_contrast > 0.0:
            for i in (2, 3):
                d = mu[i] - mu[i + 1]
                if (ad := d.abs()) < self.min_cortical_contrast:
                    correction = self.min_cortical_contrast - ad
                    if d < 0:
                        mu[i + 1] += correction
                    else:
                        mu[i + 1] -= correction

        self.mu = mu
        self.sigma = sigma

    def sample_image_from_gaussians(self, label: torch.Tensor):
        # sample synthetic image from gaussians
        # a = mu[Gr]
        # b = sigma[Gr]
        # c = torch.randn(Gr.shape)
        # SYN = a + b * c
        # d = self._rand_buffer.normal_(a, b, generator=self._generator)
        # torch.normal(a, b, generator=self._generator, out=self._rand_buffer)
        label = label.round().long()
        return self.mu[label] + self.sigma[label] * torch.randn(
            label.shape, device=self.device
        )

    def add_partial_volume_effect(self, label: torch.Tensor, image: torch.Tensor):
        if not self.add_partial_volume:
            return image

        # That is, 2 < label < 4
        # wm = lut.Left_Cerebral_White_Matter
        # gm = lut.Left_Lateral_Ventricle
        mask = (label > lut.Left_Cerebral_White_Matter) & (
            label < lut.Left_Lateral_Ventricle
        )

        if mask.any():  # only if we have partial volume information

            Gv = label[mask]
            isv = torch.zeros(Gv.shape, device=self.device)
            # e.g., 2.25 -> 75 % WM
            pw = (Gv <= 3) * (3 - Gv)
            isv += pw * self.mu[2] + pw * self.sigma[2] * torch.randn(
                Gv.shape, device=self.device
            )
            # e.g., 2.75 -> 25 % GM and 3.25 -> 75 % GM
            pg = (Gv <= 3) * (Gv - 2) + (Gv > 3) * (4 - Gv)
            isv += pg * self.mu[3] + pg * self.sigma[3] * torch.randn(
                Gv.shape, device=self.device
            )
            # e.g., 3.25 -> 25 CSF
            pcsf = (Gv >= 3) * (Gv - 3)
            isv += pcsf * self.mu[4] + pcsf * self.sigma[4] * torch.randn(
                Gv.shape, device=self.device
            )

            # this will also kill the random gaussian noise in the PV areas
            image[mask] = isv

        return image

    def forward(self, label):
        image = self.sample_image_from_gaussians(label)
        image = self.add_partial_volume_effect(label, image)
        # image[image < lut.Unknown] = lut.Unknown # clip
        return image

class RandMaskRemove(RandomizableTransform):
    def __init__(
        self,
        mask: torch.Tensor,
        prob: float = 1.0,
        device: None | torch.device = None,
    ):
        """Randomly sets everything in mask to 0."""
        super().__init__(prob, device)
        self.mask = mask

    def forward(self, image):
        if self.should_apply_transform:
            image[self.mask] = 0
        return image


class IntensityNormalization(BaseTransform):
    def __init__(
        self,
        low: float = 0.001,
        high: float = 0.999,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.low = low
        self.high = high

    def forward(self, image):
        ql = image.quantile(self.low)
        qu = image.quantile(self.high)
        return torch.clip((image - ql) / (qu - ql), 0.0, 1.0)


class RandGaussianNoise(RandomizableTransform):
    def __init__(
        self,
        mean: float,
        std_range: tuple[float, float] | list[float],
        prob: float = 1.0,
        device: None | torch.device = None,
    ):
        super().__init__(prob, device)
        self.mean = mean
        self.std_range = std_range

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not self.should_apply_transform():
            return image
        a, b = self.std_range
        std = a + (b - a) * torch.rand(1, device=self.device)
        return image + self.mean + std * torch.randn(image.shape)


class RandGammaTransform(RandomizableTransform):
    def __init__(self, mean: float, std: float, prob: float = 1.0):
        super().__init__(prob)
        self.mean = mean
        self.std = std

    def _sample_gamma(self, device):
        return torch.exp(self.mean + self.std * torch.randn(1, device=device))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not self.should_apply_transform():
            return image
        gamma = self._sample_gamma(image.device)
        # apply transform
        ra = image.aminmax()
        r = ra.max - ra.min
        return ((image - ra.min) / r) ** gamma * r + ra.min


class RandBiasfield(RandomizableTransform):
    def __init__(
        self,
        size: tuple | list | torch.Size | torch.Tensor,
        scale_min: float,
        scale_max: float,
        std_min: float,
        std_max: float,
        photo_mode: bool = False,
        interpolate_kwargs: dict | None = None,
        prob: float = 1.0,
        device: None | torch.device = None,
    ):
        super().__init__(prob, device)

        self.size = torch.as_tensor(size, device=self.device)
        self.photo_mode = photo_mode
        self.interpolate_kwargs = interpolate_kwargs or dict(
            mode="trilinear", align_corners=True
        )

        if self.should_apply_transform():
            self.generate_biasfield(scale_min, scale_max, std_min, std_max)
        else:
            self.biasfield = None

    def generate_biasfield(
        self, scale_min: float, scale_max: float, std_min: float, std_max: float
    ):
        """Synthesize a bias field."""

        bf_scale = scale_min + torch.rand(1, device=self.device) * (
            scale_max - scale_min
        )
        bf_small_size = torch.round(bf_scale * self.size).to(torch.int)

        if self.photo_mode:
            bf_small_size[1] = torch.round(self.size[1] / self.spacing).to(torch.int)

        # reduced size
        bf_small = std_min + (std_max - std_min) * torch.rand(
            1, device=self.device
        ) * torch.randn(*bf_small_size, device=self.device)

        # Resize
        self.biasfield = torch.nn.functional.interpolate(
            input = bf_small[None, None], # add batch, channel dims
            size = tuple(self.size),
            **self.interpolate_kwargs,
        )[
            0
        ].exp()  # remove batch dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Add a log-transformed bias field to an image."""
        # factor = 300.0
        # # Gamma transform
        # gamma = torch.exp(self.config["intensity"]["gamma_std"] * torch.randn(1))
        # SYN_gamma = factor * (image / factor) ** gamma
        return image * self.biasfield if self.should_apply_transform() else image


class RandBlendImages(RandomizableTransform):
    def __init__(self, image_name: str, blend_names: tuple | list, prob: float = 1.0):
        super().__init__(prob)

        self.image_name = image_name
        self.blend_names = blend_names

    def forward(self, images: dict[str, torch.Tensor]):
        if not self.should_apply_transform:
            return images[self.image_name]

        # choose image
        i = torch.randint(0, len(self.config.intensity.blend.images), (1,))
        # choose ratio
        sr = torch.rand(1)
        ir = 1 - sr
        return sr * images[self.image_name] + ir * images[self.blend_names[i]]
