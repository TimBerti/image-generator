import cv2
import einops
import numpy as np
import random
import torch
import pickle

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.ddim_hacked import DDIMSampler


class ImageGenerator:
    def __init__(self, params, input_image: np.ndarray):
        self.params = params
        self.input_image = input_image
        self.set_seed()
        self.load_model()
        self.process_image()
        self.build_conditions()

    def set_seed(self):
        if self.params.seed == -1:
            self.params.seed = random.randint(0, 65535)
        seed_everything(self.params.seed)

    def load_model(self):
        with open('./models/model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        self.ddim_sampler = DDIMSampler(self.model)
        self.model.control_scales = [self.params.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.params.guess_mode else ([self.params.strength] * 13)

    def process_image(self):
        self.img = resize_image(HWC3(self.input_image), self.params.image_resolution)

        detected_map = cv2.Canny(self.img, self.params.low_threshold, self.params.high_threshold)
        self.detected_map = HWC3(detected_map)

        control = torch.from_numpy(self.detected_map.copy()).float().cpu() / 255.0
        control = torch.stack([control for _ in range(self.params.num_samples)], dim=0)
        self.control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    def build_conditions(self):
        self.cond = {"c_concat": [self.control], "c_crossattn": [self.model.get_learned_conditioning([self.params.prompt + ', ' + self.params.a_prompt] * self.params.num_samples)]}
        self.un_cond = {"c_concat": None if self.params.guess_mode else [self.control], "c_crossattn": [self.model.get_learned_conditioning([self.params.n_prompt] * self.params.num_samples)]}

    def generate(self):
        shape = (4, self.img.shape[0] // 8, self.img.shape[1] // 8)
        samples, _ = self.ddim_sampler.sample(self.params.ddim_steps, self.params.num_samples, shape, 
                                              self.cond, verbose=False, eta=self.params.eta, 
                                              unconditional_guidance_scale=self.params.scale, 
                                              unconditional_conditioning=self.un_cond)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        return [x_samples[i] for i in range(self.params.num_samples)]


# import defaults.params as params
# import imageio
# import matplotlib.pyplot as plt

# input_image = imageio.imread('./defaults/mri_brain.jpg')
# image_generator = ImageGenerator(params, input_image)

# result = image_generator.generate()
# plt.imshow(result[0])
# plt.axis(False)
# plt.show()
