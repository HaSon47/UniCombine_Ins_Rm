import os,sys
from tqdm import tqdm
import ipdb
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import torch
from src.condition import Condition
from PIL import Image, ImageDraw
from src.UniCombineTransformer2DModel import UniCombineTransformer2DModel
from src.UniCombinePipeline import UniCombinePipeline
from accelerate.utils import set_seed
import json
import argparse
import cv2
import numpy as np
from datetime import datetime
weight_dtype = torch.bfloat16
device = torch.device("cuda:0")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,default="ckpt/FLUX.1-schnell",)
    parser.add_argument("--transformer",type=str,default="ckpt/FLUX.1-schnell/transformer",)
    parser.add_argument("--condition_types", type=str, nargs='+', default=["fill","subject"],)
    parser.add_argument("--condition_lora_dir",type=str,default="ckpt/Condition_LoRA",)
    parser.add_argument("--denoising_lora_dir",type=str,default="ckpt/Denoising_LoRA",)
    parser.add_argument("--denoising_lora_name",type=str,default="subject_fill_union",)
    parser.add_argument("--denoising_lora_weight",type=float,default=1.0,)
    parser.add_argument("--work_dir",type=str,default="/mnt/disk1/aiotlab/hachi/Output/Location_squared/Turn_1/Unicombine",)
    parser.add_argument("--test_dir",type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--exam_size",type=int,default=224,)
    parser.add_argument("--version",type=str,default="training-based",choices=["training-based","training-free"])
    args = parser.parse_args()
    args.revision = None
    args.variant = None
    
    return args

def convert_image(image):
    return image if isinstance(image, Image.Image) else Image.open(image)

def get_inputs(test_dir, root_path="/mnt/disk1/aiotlab/hachi/data/OBJ_INS_FSC"):
    print("get input")
    inputs = []
    img_folder_path = os.path.join(test_dir, "Image")
    anno_bbox_folder_path = os.path.join(test_dir, "Anno")
    for img in tqdm(os.listdir(img_folder_path)):
        img_name = img.split(".")[0]
        # get fill
        img_path = os.path.join(img_folder_path, img)
        anno_bbox_path = os.path.join(anno_bbox_folder_path, img.replace("png", "json"))
        with open(anno_bbox_path, "r") as f:
            anno_bbox = json.load(f)
        prompt = f"a {anno_bbox["class"]}"
        loc_bbox = anno_bbox["pred_box"]

        #get subject
        img_root_path = os.path.join(root_path, "test", img_name, "ground_truth.jpg")
        anno_root_path =os.path.join(root_path, "test", img_name, "fixed_annotation.json")
        if not os.path.exists(img_root_path):
            img_root_path = os.path.join(root_path, "val", img_name, "ground_truth.jpg")
            anno_root_path =os.path.join(root_path, "val", img_name, "fixed_annotation.json")
        with open(anno_root_path, "r") as f:
            anno_root = json.load(f)
        exam_bbox = anno_root["inpainted_bboxes"][0]
        exam_path = img_root_path
        inputs.append([
            img_name,
            prompt,
            img_path,
            loc_bbox,
            exam_path,
            exam_bbox
        ])
    return inputs

def get_conditions(img_path, loc_bbox, exam_path, exam_bbox, exam_size):
    conditions = []
    # subject
    subject = Image.open(exam_path).convert("RGB").crop(exam_bbox)
    subject = subject.resize((exam_size,exam_size), Image.Resampling.LANCZOS)
    conditions.append(Condition("subject", raw_img=convert_image(subject)))
    # fill
    fill  = convert_image(img_path)
    fill.paste((0,0,0), loc_bbox)
    #fill  = convert_image(img_path).convert("RGB")
    #import pdb; pdb.set_trace()
    conditions.append(Condition("fill", raw_img=convert_image(fill)))

    # mask (for caculate metric)
    mask = create_mask_from_bbox(fill.size, loc_bbox)
    return conditions, mask

def create_mask_from_bbox(image_size, bbox):
    """
    Create mask image from bbox (xyxy)
    
    Args:
        image_size: (width, height)
        bbox: (x1, y1, x2, y2)
    
    Returns:
        PIL Image mask (mode 'L')
    """
    width, height = image_size
    x1, y1, x2, y2 = bbox

    # Create black mask
    mask = Image.new('L', (width, height), 0)

    # Draw white rectangle for bbox
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask

def inference(args):
    if args.seed is not None:
        set_seed(args.seed)
    
    # model
    transformer = UniCombineTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=args.transformer,
    ).to(device = device, dtype=weight_dtype)

    for condition_type in args.condition_types:
        transformer.load_lora_adapter(f"{args.condition_lora_dir}/{condition_type}.safetensors", adapter_name=condition_type)

    # import pdb; pdb.set_trace()
    pipe = UniCombinePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype = weight_dtype,
        transformer = None
    )
    pipe.transformer = transformer

    if args.version == "training-based":
        pipe.transformer.load_lora_adapter(os.path.join(args.denoising_lora_dir,args.denoising_lora_name),adapter_name=args.denoising_lora_name, use_safetensors=True)
        pipe.transformer.set_adapters([i for i in args.condition_types] + [args.denoising_lora_name],[1.0,1.0,args.denoising_lora_weight])
    elif args.version == "training-free":
        pipe.transformer.set_adapters([i for i in args.condition_types])

    pipe = pipe.to(device)

    output_dir = args.work_dir
    os.makedirs(os.path.join(output_dir, "grid"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    # infer
    inputs = get_inputs(args.test_dir)
    for img_name, prompt, img_path, loc_bbox, exam_path, exam_bbox in tqdm(inputs):
        conditions, mask = get_conditions(img_path, loc_bbox, exam_path, exam_bbox, args.exam_size)
        result_img = pipe(
            prompt=prompt,
            conditions=conditions,
            height=512,
            width=512,
            num_inference_steps=8,
            max_sequence_length=512,
            model_config = {},
        ).images[0]
    
        concat_image = Image.new("RGB", (512 + len(args.condition_types)*512, 512))
        for j, cond_type in enumerate(args.condition_types):
            cond_image = conditions[j].condition
            if cond_type == "fill":
                cond_image = cv2.rectangle(np.array(cond_image), loc_bbox[:2], loc_bbox[2:], color=(128, 128, 128),thickness=-1)
                cond_image = Image.fromarray(cv2.rectangle(cond_image, loc_bbox[:2], loc_bbox[2:], color=(255, 215, 0), thickness=2))
            concat_image.paste(cond_image, (j * 512, 0))
        concat_image.paste(result_img, (j * 512 + 512, 0))
        concat_image.save(os.path.join(output_dir, "grid", f"{img_name}.jpg"))
        result_img.save(os.path.join(output_dir, "img",f"{img_name}.jpg"))
        mask.save(os.path.join(output_dir, "mask",f"{img_name}.jpg"))

if __name__ == "__main__":
    args = parse_args()
    inference(args)
