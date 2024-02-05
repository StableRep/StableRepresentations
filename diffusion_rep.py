import time
import torch

from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import ipdb
from torchvision import transforms as T
from diffusers.image_processor import VaeImageProcessor
from matplotlib import pyplot as plt
from daam import trace

def tokenize_captions(prompt, tokenizer):
    inputs = tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    return inputs.input_ids

def diffusion_transforms(resize_size=512, center_crop=True):
    # Preprocessing the datasets.
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resize_size) if center_crop else T.Identity(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

def diffusion_transforms_unnormalized(resize_size=512, center_crop=True):
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resize_size) if center_crop else T.Identity(),
        ]
    )

class DiffusionRepresentation(torch.nn.Module):
    def __init__(self, model_name_or_path, dtype="float32", device="cuda"):
        super().__init__()
        dtype = torch.float16 if dtype == "float16" else torch.float32
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.device = device
        self.vae_scale_factor = 8

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name_or_path, subfolder="text_encoder"
        ).to(device, dtype=dtype).eval()
        self.vae = AutoencoderKL.from_pretrained(
            model_name_or_path, subfolder="vae"
        ).to(device, dtype=dtype).eval()
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name_or_path,
            subfolder="unet",
        ).to(device, dtype=dtype).eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name_or_path, subfolder="tokenizer"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")

    def encode_prompt(self, prompt):
        tokens = tokenize_captions(prompt, self.tokenizer).to(self.device)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(tokens)[0]
        return encoder_hidden_states 
    
    def forward(self, latents, prompt=[""], t=[199], get_mid_representations=True):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        bsz = latents.shape[0]
        # confirm timesteps is between 0 and num_train_timesteps
        assert torch.all((torch.tensor(t) >= 0) & (torch.tensor(t) < self.noise_scheduler.config.num_train_timesteps)), \
            f"timesteps must be between 0 and {self.noise_scheduler.config.num_train_timesteps}"

        # create timesteps for each sample in the batch
        timesteps = torch.tensor(t, device=latents.device).repeat(bsz)
        timesteps = timesteps.long()

        encoder_hidden_states = self.encode_prompt(prompt)

        if encoder_hidden_states.shape[0] == 1:
            # If only one caption was passed in, repeat it for each sample in the batch
            encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)                          

        noisy_latents = latents
        # noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        return self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            return_dict=False,
        )[0]

def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
    # get the original timestep using init_timestep
    if denoising_start is None:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
    else:
        t_start = 0
    
    timesteps = self.noise_scheduler.timesteps[t_start * self.noise_scheduler.order :]

    # Strength is irrelevant if we directly request a timestep to start at;
    # that is, strength is determined by the denoising_start instead.
    if denoising_start is not None:
        discrete_timestep_cutoff = int(
            round(
                self.noise_scheduler.config.num_train_timesteps
                - (denoising_start * self.noise_scheduler.config.num_train_timesteps)
            )
        )
        timesteps = list(filter(lambda ts: ts < discrete_timestep_cutoff, timesteps))
        return torch.tensor(timesteps), len(timesteps)

    return timesteps, num_inference_steps - t_start

def denoise(img_path, prompt, search_word, use_actual_image=False, use_daam=False):
    denoising_start = 0.9
    num_inference_steps = 50
    # extract just the image name without the extension from the path
    img_name = img_path.split('/')[-1].split('.')[0]

    layer_id = 0
    resize_size = 256
    wrapper = DiffusionRepresentation('runwayml/stable-diffusion-v1-5')
    transforms = diffusion_transforms(resize_size=resize_size, center_crop=True)
    copy_transforms = diffusion_transforms_unnormalized(resize_size=resize_size, center_crop=True)
    image_processor = VaeImageProcessor(vae_scale_factor=wrapper.vae_scale_factor)

    def dummy(images, **kwargs): return images, [False] * len(images) 
    wrapper.run_safety_checker = dummy

    if not use_actual_image:
        wrapper.noise_scheduler.set_timesteps(num_inference_steps, device='cuda')
        timesteps = wrapper.noise_scheduler.timesteps
    else:
        timesteps, num_inference_steps = get_timesteps(wrapper, num_inference_steps, strength=1, device='cuda', denoising_start=denoising_start)
        timesteps = timesteps[::20]
        num_inference_steps = len(timesteps)
        wrapper.noise_scheduler.set_timesteps(50, device='cuda')

    img = Image.open(img_path)
    img_copy = Image.open(img_path)

    img = transforms(img)
    img_copy = copy_transforms(img_copy)
    x = image_processor.preprocess(img)
    print('max val in img' , x.max())
    height, width = x.shape[-2:]
    print(f"Image shape: {x.shape}")
    
    if use_actual_image:
        latents = wrapper.vae.encode(x.to(wrapper.device, dtype=wrapper.dtype)).latent_dist.sample().detach()  # TODO: Decide between sample and mean
        latents = latents * wrapper.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        latents = wrapper.noise_scheduler.add_noise(latents, noise, timesteps[0])

        # debug
        image = wrapper.vae.decode(latents / wrapper.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = image_processor.postprocess(image.float().detach().cpu(), output_type='pil', do_denormalize=do_denormalize)
        image[0].save(f"vae_{img_name}.png")

    else:
        # 5. Prepare latent variables
        num_channels_latents = wrapper.unet.config.in_channels
        height = 512
        width = 512
        shape = (1, num_channels_latents, height // wrapper.vae_scale_factor, width // wrapper.vae_scale_factor)
        latents = torch.randn(shape, device=wrapper.device, dtype=wrapper.dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * wrapper.noise_scheduler.init_noise_sigma

    init_latents = latents.clone()

    if use_daam:
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(wrapper, image_size=resize_size) as tc:

                for i, t in enumerate(timesteps[:1]):
                    print(t)
                    model_pred = wrapper.forward(latents, prompt, t)
                    latents = wrapper.noise_scheduler.step(model_pred, t, latents)[0]

                heat_map = tc.compute_global_heat_map() # factors=[4]
                # heat_map = heat_map.compute_word_heat_map_raw(search_word)
                heat_map = heat_map.compute_word_heat_map(search_word)
                heat_map.plot_overlay(img_copy)           
                # img_copy.save(f"test{img_name}_{search_word}.png")
                plt.savefig(f'fix_test{img_name}_{search_word}_layer{layer_id}.png')

    else:
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                print(t)
                model_pred = wrapper.forward(latents, prompt, t, get_mid_representations=False)    
                latents = wrapper.noise_scheduler.step(model_pred, t, latents)[0]

        image = wrapper.vae.decode(latents / wrapper.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = image_processor.postprocess(image.detach().cpu(), output_type='pil', do_denormalize=do_denormalize)
        # save image
        image[0].save(f"construct_{img_name}_layer{layer_id}.png")

def denoise_from_scratch(use_actual_image=False, use_daam=False):

    img_prompt_pairs = [
        ]

    for img_path, prompt, search_word in img_prompt_pairs:
        denoise(img_path, prompt, search_word, use_actual_image=use_actual_image, use_daam=use_daam)
    
    # img_path, prompt, search_word = img_prompt_pairs[0]

    # denoise(img_path, prompt, search_word, use_actual_image=use_actual_image, use_daam=use_daam)
    

def load_video_and_save_frames(video_path):
    # load an mp4 file and save a few frames
    import imageio
    import numpy as np
    import os
    ipdb.set_trace()
    vid = imageio.get_reader(video_path,  'ffmpeg')
    num_frames = vid.count_frames()
    print(f"num frames: {num_frames}")
    frames = []
    for i in range(num_frames):
        if i%20 == 0:
            image = vid.get_data(i)
            frames.append(image)

    # save images
    for i, image in enumerate(frames):
        imageio.imwrite(f"frame_{i}.png", image)


def __main__():
    import argparse
    denoise('FK1_Knob1OnRandom_v2d-v4_seed2_left_4.png', '', '', use_daam=False, use_actual_image=True)

if __name__ == "__main__":
    __main__()
