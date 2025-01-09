import torch

from brainsynth.constants import IMAGE
from brainsynth.transforms.base import BaseTransform, RandomizableTransform
from brainsynth.transforms.filters import GaussianSmooth

gl = IMAGE.generation_labels

class SynthesizeFromMultivariateNormal(BaseTransform):
    def __init__(
        self,
        mean_loc: torch.Tensor,
        mean_scale_tril: torch.Tensor,
        sigma_loc: torch.Tensor,
        sigma_scale_tril: torch.Tensor,
        pv_sigma_range: list[float] = [0.25, 0.5],
        pv_tail_length: float = 2.0,
        device: None | torch.device = None
    ) -> None:
        super().__init__(device)

        self.dist_mean = torch.distributions.MultivariateNormal(
            mean_loc.to(self.device), scale_tril=mean_scale_tril.to(self.device)
        )
        self.dist_sigma = torch.distributions.MultivariateNormal(
            sigma_loc.to(self.device), scale_tril=sigma_scale_tril.to(self.device)
        )

        # sample smoothing
        pv_sigma = pv_sigma_range[0] + (
            pv_sigma_range[1] - pv_sigma_range[0]
        ) * torch.rand(1, device=self.device)
        pv_sigma = pv_sigma.item()
        if (isinstance(pv_sigma, float) and (pv_sigma > 0.0)) or (
            isinstance(pv_sigma, (list, tuple)) and any(i > 0.0 for i in pv_sigma)
        ):
            self.gaussian_blur = GaussianSmooth(
                pv_sigma, pv_tail_length, device=self.device
            )
        else:
            self.gaussian_blur = None

    def sample_image(self, label: torch.Tensor):
        # draw mean and standard deviation of each cluster
        mean = self.dist_mean.sample()
        sigma = self.dist_sigma.sample()

        img = mean[label] + sigma[label] * torch.randn(label.shape, device=self.device)
        if self.gaussian_blur is not None:
            img = self.gaussian_blur(img)
        img.clamp_(0, 255)
        return img

    def forward(self, label):
        return self.sample_image(label)


class SynthesizeIntensityImage(BaseTransform):
    def __init__(
        self,
        mu_offset: float,
        mu_scale: float,
        sigma_offset: float,
        sigma_scale: float,
        # pv_sigma: float | list[float],
        pv_sigma_range: list[float] = [0.25, 1.0],
        pv_tail_length: float = 2.0,
        photo_mode: bool = False,
        min_cortical_contrast: float = 0.0,  # 25.0
        gen_labels_dist_encoded: bool = True,
        gen_labels_dist_max: float = 3.0,
        device: None | torch.device = None,
    ):
        super().__init__(device)
        self.photo_mode = photo_mode
        self.min_cortical_contrast = min_cortical_contrast

        self.sample_gaussians(mu_offset, mu_scale, sigma_offset, sigma_scale)

        # sample smoothing
        pv_sigma = pv_sigma_range[0] + (
            pv_sigma_range[1] - pv_sigma_range[0]
        ) * torch.rand(1, device=self.device)
        pv_sigma = pv_sigma.item()
        if (isinstance(pv_sigma, float) and (pv_sigma > 0.0)) or (
            isinstance(pv_sigma, (list, tuple)) and any(i > 0.0 for i in pv_sigma)
        ):
            self.gaussian_blur = GaussianSmooth(
                pv_sigma, pv_tail_length, device=self.device
            )
        else:
            self.gaussian_blur = None

        self.gen_labels_dist_encoded = gen_labels_dist_encoded
        self.gen_labels_dist_max = gen_labels_dist_max

    def sample_gaussians(self, mu_offset, mu_scale, sigma_offset, sigma_scale):
        # Generate Gaussians
        mu = torch.zeros(gl.n_labels, device=self.device)
        sigma = torch.zeros(gl.n_labels, device=self.device)

        n = gl.label_range[1] - gl.label_range[0]
        slc = slice(*gl.label_range)

        mu[slc] = mu_offset + mu_scale * torch.rand(n, device=self.device)
        sigma[slc] = sigma_offset + sigma_scale * torch.rand(n, device=self.device)

        # # draw GM mean around WM
        # std = 7.0 * sigma_scale
        # while True:
        #     new_mu = mu[gl.white] + torch.randn(1, device=self.device) * std
        #     if mu_offset <= new_mu <= mu_offset + mu_scale:
        #         mu[gl.gray] = new_mu
        #         break

        # scale sigma by mu but make sure we don't kill all noise
        # mu_max = mu_offset + mu_scale
        # scale_factor = (
        #     torch.minimum(
        #         mu[slc] + 3 * mu_offset, torch.tensor(mu_max, device=self.device)
        #     )
        #     / mu_max
        # )
        # sigma[slc] *= scale_factor

        # set the background to zero in photo mode
        if self.photo_mode:
            mu[0] = 0

        # Ensure that there is *some* contrast between white and gray matter
        # and gray matter and CSF.
        #   20 seems OK
        #   50 always gives nice contrast everytime
        if self.min_cortical_contrast > 0.0:
            for inner, outer in ((gl.white, gl.gray), (gl.gray, gl.csf)):
                d = mu[inner] - mu[outer]
                if (ad := d.abs()) < self.min_cortical_contrast:
                    correction = self.min_cortical_contrast - ad
                    if d < 0:
                        mu[outer] += correction
                    else:
                        mu[outer] -= correction

        # Partial volume information
        # --------------------------
        # mix parameters

        frac = torch.linspace(0, 1, 50, device=self.device)

        mu[gl.pv.lesion : gl.pv.white] = mu[gl.lesion] + frac * (
            mu[gl.white] - mu[gl.lesion]
        )
        mu[gl.pv.white : gl.pv.gray] = mu[gl.white] + frac * (
            mu[gl.gray] - mu[gl.white]
        )
        mu[gl.pv.gray : gl.pv.csf] = mu[gl.gray] + frac * (mu[gl.csf] - mu[gl.gray])
        mu[gl.pv.csf] = mu[gl.csf]

        sigma[gl.pv.lesion : gl.pv.white] = sigma[gl.lesion] + frac * (
            sigma[gl.white] - sigma[gl.lesion]
        )
        sigma[gl.pv.white : gl.pv.gray] = sigma[gl.white] + frac * (
            sigma[gl.gray] - sigma[gl.white]
        )
        sigma[gl.pv.gray : gl.pv.csf] = sigma[gl.gray] + frac * (
            sigma[gl.csf] - sigma[gl.gray]
        )
        sigma[gl.pv.csf] = sigma[gl.csf]

        self.mu = mu
        self.sigma = sigma

    # Used to generate the distance encoded labels

    # def distance_to_label(self, d, max_d, n, start_label):
    #     return torch.round((d+max_d)*n/max_d + start_label)

    def label_to_distance(self, label, n, start_label):
        return (
            label - start_label
        ) / n * self.gen_labels_dist_max - self.gen_labels_dist_max

    @staticmethod
    def distance_to_pv(x, i: float | torch.Tensor = 1.0):
        """Label to PV fraction transfer function. We use a scaled sigmoid."""
        return 1 / (1 + torch.exp(-i * x))

    @staticmethod
    def pv_to_label(f, start, end):
        return torch.round(f * (end - start) + start)

    def distance_labels_to_pv_labels(self, image: torch.Tensor):
        """Convert labels encoding distance (e.g., [-3.0, 3.0]) to labels
        encoding partial voluming (as in the original generation label image).

        We randomize the smoothness of the PV effect.
        """
        # higher rho -> steeper transfer function
        # [0.5, 3.0] 1-4
        # [2.0, 5.0]
        rhos = [
            1.0 + torch.rand(1, device=image.device) * 3.0, # WM-GM     [1.0, 4.0]
            2.0 + torch.rand(1, device=image.device) * 2.0, # GM-CSF    [2.0, 4.0]
        ]
        info = [(gl.pv.white, gl.pv.gray), (gl.pv.gray, gl.pv.csf)]

        # out = torch.clone(image)
        out = image

        for pv,rho in zip(info, rhos):
            half_size = (pv[1] - pv[0]) / 2  # halfway point
            mask = (image >= pv[0]) & (image <= pv[1])
            labels = image[mask]

            new_labels = self.pv_to_label(
                self.distance_to_pv(
                    self.label_to_distance(labels, half_size, pv[0]),
                    rho,
                ),
                *pv,
            ).to(image.dtype)

            out[mask] = new_labels

        return out

    def sample_image(self, label: torch.Tensor):
        # sample synthetic image from gaussians
        # a = mu[Gr]
        # b = sigma[Gr]
        # c = torch.randn(Gr.shape)
        # SYN = a + b * c
        # d = self._rand_buffer.normal_(a, b, generator=self._generator)
        # torch.normal(a, b, generator=self._generator, out=self._rand_buffer)

        if self.gen_labels_dist_encoded:
            label = self.distance_labels_to_pv_labels(label)

        img = self.mu[label]
        if self.gaussian_blur is not None:
            img_blur = self.gaussian_blur(img)
            # we explicitly model PV in some areas so use that instead of the
            # blurred image.

            explicit_pv_mask = (label >= gl.pv.lesion) & (label <= gl.pv.csf)
            img_blur[explicit_pv_mask] = img[explicit_pv_mask]
        else:
            img_blur = img
        img_blur += self.sigma[label] * torch.randn(label.shape, device=self.device)
        img_blur[img_blur < 0] = 0

        return img_blur
        # return self.mu[label] + self.sigma[label] * torch.randn(
        #     label.shape, device=self.device
        # )

    def forward(self, label):
        return self.sample_image(label)
        # image = self.sample_image(label)
        # image[image < lut.Unknown] = lut.Unknown # clip
        # return image


# trans = RandSaltAndPepperNoise(gl.kmeans, prob = 0.1)
# trans(image, label_image)


class RandSaltAndPepperNoise(RandomizableTransform):
    def __init__(
        self,
        mask,
        alpha=0.1,
        beta=0.1,
        scale=None,
        prob: float = 1.0,
        device: None | torch.device = None,
    ):
        super().__init__(prob, device)
        self.mask = mask
        self.mask_size = torch.Size([mask.sum()])
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.beta_dist = torch.distributions.Beta(
            torch.tensor([self.alpha], device=self.device),
            torch.tensor([self.beta], device=self.device),
        )

    def forward(self, image):
        if self.should_apply_transform:
            sp = self.beta_dist.sample(self.mask_size).squeeze(1)
            image[self.mask] = sp if self.scale is None else sp * self.scale
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
    def __init__(
        self,
        concentration: float = 0.5,
        rate: float = 0.5,
        gamma_clip_range: tuple = (0.5, 2.0),
        prob: float = 1.0,
    ):
        """Concentration and rate are parameters of the gamma distribution from
        which the gamma coefficient is sampled (but clipped to gamma_clip_range)."""
        super().__init__(prob)
        self.concentration = concentration
        self.rate = rate
        self.gamma_clip_range = gamma_clip_range

    # def _sample_gamma(self, device):
    #     return torch.exp(self.mean + self.std * torch.randn(1, device=device)).clamp(*self.gamma_clip_range)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not self.should_apply_transform():
            return image

        m = torch.distributions.Gamma(
            torch.tensor([self.concentration], device=image.device),
            torch.tensor([self.rate], device=image.device),
        )
        gamma = m.sample().clamp(*self.gamma_clip_range)
        # apply transform
        ra = image.aminmax()
        r = ra.max - ra.min  # original range
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
            input=bf_small[None, None],  # add batch, channel dims
            size=tuple(self.size),
            **self.interpolate_kwargs,
        )[
            0
        ].exp()  # remove batch dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Add a log-transformed bias field to an image."""
        return image * self.biasfield if self.should_apply_transform() else image


class RandBlendImages(RandomizableTransform):
    def __init__(self, blend_images: torch.Tensor | list[torch.Tensor], prob: float = 1.0):
        super().__init__(prob)

        if isinstance(blend_images, torch.Tensor):
            self.blend_images = [blend_images]
        else:
            self.blend_images = blend_images
        self.n_blend = len(blend_images)

    def forward(self, image: torch.Tensor):
        if not self.should_apply_transform:
            return image

        # choose image
        if self.n_blend > 1:
            i = torch.randint(0, self.n_blend, (1,), device=image.device)
        else:
            i = 0

        # choose ratio
        sr = torch.rand(1, device=image.device)
        return sr * image + (1 - sr) * self.blend_images[i]
