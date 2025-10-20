import math
import torch
import comfy.utils
import comfy.sample
import comfy.samplers
import latent_preview
from comfy.extra_samplers import uni_pc

def predict_noise(self, x, timestep, model_options={}, seed=None):
    if math.isclose(self.cfg, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = self.conds.get("negative", None)

    conds = [self.conds.get("positive", None), uncond_]
    if "sampler_calc_cond_batch_function" in model_options:
        args = {"conds": conds, "input": x, "sigma": timestep, "model": self.inner_model, "model_options": model_options}
        out = model_options["sampler_calc_cond_batch_function"](args)
    else:
        out = comfy.samplers.calc_cond_batch(self.inner_model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds":conds, "conds_out": out, "cond_scale": self.cfg, "timestep": timestep,
                "input": x, "sigma": timestep, "model": self.inner_model, "model_options": model_options}
        out = fn(args)

    cfg_result = x + (out[0] - out[1]) * self.cfg

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": out[0], "uncond": out[1], "cond_scale": self.cfg, "model": self.inner_model, "uncond_denoised": self.conds.get("negative", None), "cond_denoised": self.conds.get("positive", None),
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)
    return cfg_result

def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = comfy.samplers.CFGGuider(model)
    cfg_guider.predict_noise = predict_noise.__get__(cfg_guider, comfy.samplers.CFGGuider)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)

def sampler_object(name):
    if name == "uni_pc":
        sampler = comfy.samplers.KSAMPLER(uni_pc.sample_unipc)
    elif name == "uni_pc_bh2":
        sampler = comfy.samplers.KSAMPLER(uni_pc.sample_unipc_bh2)
    elif name == "ddim":
        sampler = comfy.samplers.ksampler("euler", inpaint_options={"random": True})
    else:
        sampler = comfy.samplers.ksampler(name)
    return sampler

def KSampler_sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    if sigmas is None:
        sigmas = self.sigmas

    if last_step is not None and last_step < (len(sigmas) - 1):
        sigmas = sigmas[:last_step + 1]
        if force_full_denoise:
            sigmas[-1] = 0

    if start_step is not None:
        if start_step < (len(sigmas) - 1):
            sigmas = sigmas[start_step:]
        else:
            if latent_image is not None:
                return latent_image
            else:
                return torch.zeros_like(noise)

    sampler = sampler_object(self.sampler)

    return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

def common_ksampler_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    sampler = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
    sampler.sample = KSampler_sample.__get__(sampler, comfy.samplers.KSampler)

    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = common_ksampler_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

class FE_Edit_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

NODE_CLASS_MAPPINGS = {
    "MyCustomNodeName": FE_Edit_KSampler
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCustomNodeName": "FE Edit KSampler"
}
