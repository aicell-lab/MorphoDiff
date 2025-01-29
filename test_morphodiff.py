import os
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import AutoFeatureExtractor
from morphodiff.train import CustomStableDiffusionPipeline
from morphodiff.perturbation_encoder import PerturbationEncoderInference

def main():
    ckpt_path = "/proj/aicell/users/x_aleho/MorphoDiff/models/bbbc021_14_compounds_morphodiff_ckpt/checkpoint"
    dataset_id = "BBBC021_experiment_01_resized"
    prompt = "aphidicolin"
    out_image = "morphodiff_output.png"
    guidance_scale = 1.0
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unet = UNet2DConditionModel.from_pretrained(
        ckpt_path,
        subfolder="unet_ema"
    )

    vae = AutoencoderKL.from_pretrained(
        ckpt_path,
        subfolder="vae"
    )

    scheduler = DDPMScheduler.from_pretrained(
        ckpt_path,
        subfolder="scheduler"
    )

    feature_extractor_path = os.path.join(ckpt_path, "feature_extractor")
    feature_extractor = None
    if os.path.isdir(feature_extractor_path):
        feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    perturbation_encoder = PerturbationEncoderInference(
        dataset_id=dataset_id,
        model_type="conditional",
        model_name="SD"
    )

    pipeline = CustomStableDiffusionPipeline(
        vae=vae,
        unet=unet,
        text_encoder=perturbation_encoder,
        feature_extractor=feature_extractor,
        scheduler=scheduler,
        safety_checker=None  # MorphoDiff doesn't use a safety checker
    )

    pipeline.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device.type, enabled=(device.type == "cuda")):
        output = pipeline(prompt, generator=generator, guidance_scale=guidance_scale)

    image = output.images[0]
    image.save(out_image)
    print(f"Saved '{out_image}' with prompt='{prompt}', guidance_scale={guidance_scale}.")

if __name__ == "__main__":
    main()