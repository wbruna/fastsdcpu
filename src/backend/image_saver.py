import json
from os import path, mkdir
from typing import Any
from uuid import uuid4
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from utils import get_image_file_extension
from PIL import PngImagePlugin


def get_exclude_keys():
    exclude_keys = {
        "init_image": True,
        "generated_images": True,
        "lora": {
            "models_dir": True,
            "path": True,
        },
        "dirs": True,
        "controlnet": {
            "adapter_path": True,
        },
    }
    return exclude_keys


class ImageSaver:
    @staticmethod
    def save_images(
        output_path: str,
        images: Any,
        folder_name: str = "",
        format: str = "PNG",
        lcm_diffusion_setting: LCMDiffusionSetting = None,
    ) -> None:
        gen_id = uuid4()

        if images:
            image_seeds = []

            for index, image in enumerate(images):

                image_seed = image.info.get('image_seed')
                if image_seed is not None:
                    image_seeds.append(image_seed)

                if not path.exists(output_path):
                    mkdir(output_path)

                if folder_name:
                    out_path = path.join(
                        output_path,
                        folder_name,
                    )
                else:
                    out_path = output_path

                if not path.exists(out_path):
                    mkdir(out_path)
                image_extension = get_image_file_extension(format)

                meta_args = {}

                # Save metadata as Gradio does:
                # - png metadata collecting any string inside info
                # - exif already in serialized formatted

                if format == "PNG":
                    info = PngImagePlugin.PngInfo()
                    have_meta = False
                    for key, value in image.info.items():
                        if isinstance(key, str) and isinstance(value, str):
                            info.add_text(key, value)
                            have_meta = True
                    if have_meta:
                        meta_args["pnginfo"] = info

                if format == 'JPEG':
                    exif = image.info.get("exif")
                    if exif:
                        meta_args["exif"] = exif

                image.save(path.join(out_path, f"{gen_id}-{index+1}{image_extension}"), **meta_args)

            if lcm_diffusion_setting:
                data = lcm_diffusion_setting.model_dump(exclude=get_exclude_keys())
                if image_seeds:
                    data['image_seeds'] = image_seeds
                with open(path.join(out_path, f"{gen_id}.json"), "w") as json_file:
                    json.dump(
                        data,
                        json_file,
                        indent=4,
                    )
