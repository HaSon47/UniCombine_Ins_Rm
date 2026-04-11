import os
import json 
from accelerate.logging import get_logger
import torch
logger = get_logger(__name__)
from PIL import Image
from .condition import Condition
from diffusers.image_processor import VaeImageProcessor
from datasets import load_dataset, concatenate_datasets, Dataset

def get_dataset(args):
    root_dir = args.dataset_name[0] if isinstance(args.dataset_name, list) else args.dataset_name
    
    image_dir = os.path.join(root_dir, 'images')
    inpaint_dir = os.path.join(root_dir, 'inpaint')
    box_dir = os.path.join(root_dir, 'box')

    filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    data_list = []
    for name in filenames:
        image_path = os.path.join(image_dir, f"{name}.png")
        inpaint_path = os.path.join(inpaint_dir, f"{name}.png")
        box_path = os.path.join(box_dir, f"{name}.json") # <-- SỬA ĐUÔI FILE Ở ĐÂY
        
        if os.path.exists(inpaint_path) and os.path.exists(box_path):
            data_list.append({
                "image_path": image_path,
                "inpaint_path": inpaint_path,
                "box_path": box_path
                # Ta không gán description rỗng ở đây nữa, sẽ xử lý nó lúc đọc JSON
            })
        else:
            logger.warning(f"Missing inpaint or box file for {name}. Skipping.")

    dataset = Dataset.from_list(data_list)
    logger.info(f"Loaded {len(dataset)} valid samples from local dataset.")
    return dataset

def prepare_dataset(dataset, vae_scale_factor, accelerator, args):
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_resize=True, do_convert_rgb=True)

    def preprocess_conditions(conditions):
        conditioning_tensors = []
        conditions_types = []
        for cond in conditions:
            conditioning_tensors.append(image_processor.preprocess(cond.condition, width=args.resolution, height=args.resolution).squeeze(0))
            conditions_types.append(cond.condition_type)
        return torch.stack(conditioning_tensors, dim=0), conditions_types

    def preprocess(examples):
        pixel_values = []
        condition_latents = []
        condition_types = []
        bboxes = []
        descriptions = [] 

        for img_p, inpaint_p, box_p in zip(examples["image_path"], examples["inpaint_path"], examples["box_path"]):
            # 1. Load Images
            input_image = Image.open(img_p).convert("RGB")     
            gt_image = Image.open(inpaint_p).convert("RGB")    
            width, height = gt_image.size

            # 2. Đọc dữ liệu từ file JSON
            with open(box_p, 'r', encoding='utf-8') as f:
                box_data = json.load(f)
                x1, y1, x2, y2 = box_data["loc_box"]
                class_name = box_data["class_name"]

            # 3. Scale Bbox về resolution của mô hình
            scaled_x1 = int(x1 * args.resolution / width)
            scaled_y1 = int(y1 * args.resolution / height)
            scaled_x2 = int(x2 * args.resolution / width)
            scaled_y2 = int(y2 * args.resolution / height)
            bboxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])

            # 4. Trích xuất Subject
            subject_image = gt_image.crop((x1, y1, x2, y2))

            # 5. Xử lý Ground Truth thành tensor
            pixel_values.append(image_processor.preprocess(gt_image, width=args.resolution, height=args.resolution).squeeze(0))

            # 6. Chuẩn bị các Conditions
            conditions = []
            for condition_type in args.condition_types:
                if condition_type == "subject":
                    conditions.append(Condition("subject", condition=subject_image))
                elif condition_type == "fill":
                    conditions.append(Condition("fill", condition=input_image))
                else:
                    raise ValueError(f"Only support subject and fill for local folders currently. Got {condition_type}")

            cond_tensors, cond_types = preprocess_conditions(conditions)
            condition_latents.append(cond_tensors)
            condition_types.append(cond_types)
            
            # 7. Tận dụng class_name làm Text Prompt
            # Bạn có thể format chuỗi này tùy theo cách bạn muốn model "hiểu" lệnh
            prompt_text = f"insert a {class_name}" 
            descriptions.append({"description_0": prompt_text, "item": class_name})

        examples["pixel_values"] = pixel_values
        examples["condition_latents"] = condition_latents
        examples["condition_types"] = condition_types
        examples["bbox"] = bboxes
        examples["description"] = descriptions 
        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess)

    return dataset