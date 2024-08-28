from PIL import Image, ExifTags
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from os import path

def resize_pil_image(
    pil_image: Image,
    image_width,
    image_height,
):
    return pil_image.convert("RGB").resize(
        (
            image_width,
            image_height,
        ),
        Image.Resampling.LANCZOS,
    )

def format_metadata(
    config: dict,
) -> str:
    lines = []

    lines.append(config['prompt'])
    lines.append('Negative prompt: ' + config['negative_prompt'])

    info = []

    info.append(('Steps', config['inference_steps']))

    info.append(('Sampler', 'LCM'))

    info.append(('CFG Scale', config['guidance_scale']))

    seed = config.get('seed', -1)
    if seed != -1:
        info.append(('Seed', seed))

    info.append(('Size', '{}x{}'.format(config['image_width'], config['image_height'])))

    model = None
    if config['use_openvino']:
        model = config['openvino_lcm_model_id']
    elif config['use_lcm_lora']:
        model = config['lcm_lora']['base_model_id']
    else:
        model = config['lcm_model_id']
    if model:
        if model.endswith('.safetensors'):
            # filename without extension
            model = path.basename(model).rsplit('.', 1)[0]
        elif path.isdir(model):
            # get the last component
            model = path.basename(path.normpath(model))
        # else assume it's a model id
        info.append(('Model', model))

    diffusion_task = config['diffusion_task']

    if diffusion_task == 'image_to_image':
        info.append(('Mode', 'Img2Img'))
    elif diffusion_task == 'text_to_image':
        info.append(('Mode', 'Txt2Img'))

    if diffusion_task == 'image_to_image':
        info.append(('Denoising strength', config["strength"]))

    clip_skip = config.get('clip_skip', 1)
    if clip_skip > 1:
        info.append(('Clip skip', clip_skip))

    token_merging = config.get('token_merging', 0.0)
    if token_merging >= 0.01:
        info.append(('Token merging ratio', token_merging))

    lines.append(', '.join(['{}: {}'.format(k, v) for (k, v) in info]))

    return '\n'.join(lines)

def add_metadata_to_pil_image(
    pil_image: Image,
    config: LCMDiffusionSetting,
) -> None:
    '''add generation parameters to the image info fields, in a Gradio-compatible way'''

    config_dict = config.model_dump()
    # the image seed will be different from the configured one for random
    # seeds or multiple images, so just override it
    config_dict['seed'] = pil_image.info.get('image_seed', -1)
    metadata = format_metadata(config_dict)

    pil_image.info['parameters'] = metadata

    # borrowed from piexif
    usercomment = b'UNICODE\0' + metadata.encode('utf_16_be', errors='replace')
    # The PIL Exif encoder detects both bytes and bytearrays as sequences of
    # integers, so they get encoded with the wrong type, and most tools won't
    # interpret that as text. A list wrapping a bytearray dodges those
    # heuristics, correctly storing the data as a byte sequence.
    usercomment = [bytearray(usercomment)]
    exif = pil_image.getexif()
    exif.setdefault(ExifTags.IFD.Exif, {})[ExifTags.Base.UserComment] = usercomment
    pil_image.info['exif'] = exif.tobytes()

