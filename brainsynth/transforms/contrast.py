import torch

from brainsynth.transforms.base import RandomizableTransform


class RandSyntheticImage(RandomizableTransform):
    def __init__(self, mu_offset, mu_scale, sigma_offset, sigma_scale, add_partial_volume, alternative_images, prob: float = 1):
        super().__init__(prob)

        self.mu_offset = mu_offset
        self.mu_scale = mu_scale
        self.sigma_offset = sigma_offset
        self.sigma_scale = sigma_scale

        self.add_partial_volume = add_partial_volume
        self.alternative_images = alternative_images



    def sample_gaussians(self):
        # extracerebral tissue (from k-means clustering) are added as
        # 500 + label so we need to take this into account, hence the value of
        # n
        n = 500 + 10
        mu = self.mu_offset + self.mu_scale * torch.rand(n)
        sigma = self.sigma_offset + self.sigma_scale * torch.rand(n)

        # set the background to zero every once in a while (or always in photo mode)
        if self.do_task["photo_mode"] or torch.rand(1) < 0.5:
            mu[0] = 0

        return mu, sigma

    def add_partial_volume_effect(self, label, image, mu, sigma):
        if not self.add_partial_volume:
            return image

        # That is, 2 < label < 4
        # wm = lut.Left_Cerebral_White_Matter
        # gm = lut.Left_Lateral_Ventricle
        mask = (label > lut.Left_Cerebral_White_Matter) & (label < lut.Left_Lateral_Ventricle)

        if mask.any(): # only if we have partial volume information
            Gv = label[mask]
            isv = torch.zeros(Gv.shape)
            pw = (Gv <= 3) * (3 - Gv)
            isv += pw * mu[2] + pw * sigma[2] * torch.randn(Gv.shape)
            pg = (Gv <= 3) * (Gv - 2) + (Gv > 3) * (4 - Gv)
            isv += pg * mu[3] + pg * sigma[3] * torch.randn(Gv.shape)
            pcsf = (Gv >= 3) * (Gv - 3)
            isv += pcsf * mu[4] + pcsf * sigma[4] * torch.randn(Gv.shape)

            # this will also kill the random gaussian noise in the PV areas
            image[mask] = isv

        return image

    # def synthesize_image(
    #     self,
    #     gen_label: torch.Tensor,
    #     input_resolution: torch.Tensor,
    # ):
    #     if not self.do_task["intensity"]:
    #         return {}

    #     # Generate (log-transformed) bias field
    #     biasfield = self.synthesize_bias_field(gen_label.meta)
    #     # biasfield = monai.transforms.RandBiasField(degree=5, coeff_range=(0,0.1), prob=0.5)


    #     # Synthesize, deform, corrupt
    #     synth = self.synthesize_contrast(gen_label)
    #     synth = self.apply_grid_sample(synth, self.deformed_grid)
    #     synth = self.gamma_transform(synth)
    #     synth = self.add_bias_field(synth, biasfield)
    #     synth = self.simulate_resolution(synth, input_resolution)

    #     return dict(biasfield=biasfield, synth=synth)


    def forward(self, label):
        if not self._do_transform:
            # select alternative synth
            return

        Gr = label.round().long()

        mu, sigma = self.sample_gaussians()

        insert_pv = False



        # skull-strip every now and then by relabeling extracerebral tissues to
        # background
        if self.do_task["skull_strip"]:
            Gr[Gr >= 500] = 0
            # Should we also skull-strip the biasfield..?

        # sample synthetic image from gaussians
        # a = mu[Gr]
        # b = sigma[Gr]
        # c = torch.randn(Gr.shape)
        # SYN = a + b * c
        # d = self._rand_buffer.normal_(a, b, generator=self._generator)
        # torch.normal(a, b, generator=self._generator, out=self._rand_buffer)

        scale_factor = mu / (self.mu_offset + self.mu_scale)
        scaled_sigma = sigma * scale_factor

        SYN = mu[Gr] + scaled_sigma[Gr] * torch.randn(Gr.shape)

        # SYN = mu[Gr] + sigma[Gr] * torch.randn(Gr.shape)

        # SYN = self.rand_gauss_noise(SYN)

        SYN = self.add_partial_volume_effect(label, SYN, mu, sigma)


        SYN[SYN < lut.Unknown] = lut.Unknown

        return SYN

class RandSkullStrip(RandomizableTransform):
    def __init__(self, prob: float = 1):
        super().__init__(prob)

    def forward(self, image):
        if self._do_transform:
            image[self.mask] = 0
        return image


class NormalizeIntensity(torch.nn.Module):
    def forward(self, image):
        # r = image.aminmax()
        # return (image - r.min) / (r.max - r.min)
        ql = image.quantile(0.001)
        qu = image.quantile(0.999)
        return torch.clip((image - ql) / (qu - ql), 0.0, 1.0)


class RandGaussianNoise(RandomizableTransform):
    def __init__(self, prob, mean, std_range):
        super().__init__(prob)
        self.mean = mean
        self.std_range = std_range

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        super().randomize()
        if not self._do_transform:
            return image
        a,b = self.std_range
        std = a + (b - a) * torch.rand(1)
        return image + self.mean + std * torch.randn(image.shape)


class RandGammaTransform(RandomizableTransform):
    def __init__(self, prob: float, mean, std):
        super().__init__(prob)
        self.mean = mean
        self.std = std

    def _sample_gamma(self):
        return torch.exp(self.mean + self.std * torch.randn(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        super().randomize()
        if not self._do_transform:
            return image
        gamma = self._sample_gamma()
        # apply transform
        ra = image.aminmax()
        r = ra.max - ra.min
        return ((image - ra.min) / r) ** gamma * r + ra.min

class RandBiasField(RandomizableTransform):
    def __init__(self, size, scale_min, scale_max, std_min, std_max, photo_mode: bool, interpolate_kwargs=None, prob: float = 1):
        super().__init__(prob)

        self.size = size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.std_min = std_min
        self.std_max = std_max
        self.interpolate_kwargs = interpolate_kwargs or dict(mode="trilinear", align_corners=True)

        self.photo_mode = photo_mode

        self.generate_biasfield()

    def generate_biasfield(self):
        super().randomize()

        if not self._do_transform:
            self.biasfield = None
            return

        """Synthesize a bias field."""

        bf_scale = self.scale_min + torch.rand(1) * (self.scale_max - self.scale_min)
        size_BF_small = torch.round(bf_scale * self.size).to(torch.int)

        if self.self.photo_mode:
            size_BF_small[1] = torch.round(self.size[1] / self.spacing).to(
                torch.int
            )

        # reduced size
        BFsmall = self.std_min + (self.std_max - self.std_min) * torch.rand(1) * torch.randn(
            *size_BF_small
        )

        # Resize
        self.biasfield = torch.nn.functional.interpolate(
            input=BFsmall[None, None], # add batch, channel dims
            size=tuple(self.size),
            **self.interpolate_kwargs,
        )[0].exp() # remove batch dim

    def forward(self, image: torch.Tensor, randomize=False) -> torch.Tensor:
        """Add a log-transformed bias field to an image."""
        if randomize:
            self.generate_biasfield()

        # factor = 300.0
        # # Gamma transform
        # gamma = torch.exp(self.config["intensity"]["gamma_std"] * torch.randn(1))
        # SYN_gamma = factor * (image / factor) ** gamma
        return image if self.biasfield is None else image * self.biasfield



class RandBlendImages(RandomizableTransform):
    def __init__(self, image_name: str, blend_names: tuple | list, prob: float = 1):
        super().__init__(prob)

        self.image_name = image_name
        self.blend_names = blend_names

    def forward(self, images: dict[str, torch.Tensor]):
        if not self._do_transform:
            return images[self.image_name]

        # choose image
        i = torch.randint(0, len(self.config.intensity.blend.images), (1,))
        # choose ratio
        sr = torch.rand(1)
        ir = 1 - sr
        return sr * images[self.image_name] + ir * images[self.blend_names[i]]
