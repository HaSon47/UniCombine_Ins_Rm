import os
import json 
import random
from accelerate.logging import get_logger
import torch
import torchvision.transforms as transforms
logger = get_logger(__name__)
from PIL import Image
from .condition import Condition
from diffusers.image_processor import VaeImageProcessor
from datasets import load_dataset, concatenate_datasets, Dataset

from typing import List

def get_dataset(args):
    root_dir = args.dataset_name[0] if isinstance(args.dataset_name, list) else args.dataset_name
    
    image_dir = os.path.join(root_dir, 'images')
    inpaint_dir = os.path.join(root_dir, 'inpaint')
    box_dir = os.path.join(root_dir, 'box_json')

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

    # --- KHỞI TẠO AUGMENTATIONS ---
    # Bạn có thể tuỳ chỉnh các tham số degrees, distortion_scale, hoặc values của ColorJitter theo ý muốn
    geom_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=15, fill=127)
        ], p=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=127)
    ])
    
    color_transforms = transforms.ColorJitter(
        brightness=0.15, 
        contrast=0.15, 
        saturation=0.15, 
        hue=0.05
    )
    # ------------------------------

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
            orig_width, orig_height = gt_image.size

            # 2. Đọc dữ liệu từ file JSON
            with open(box_p, 'r', encoding='utf-8') as f:
                box_data = json.load(f)
                x1, y1, x2, y2 = box_data["loc_box"]
                class_name = box_data["class_name"]

            # ==========================================
            # 3. LOGIC SMART SQUARE CROP THEO BBOX
            # ==========================================
            # Xác định kích thước khung cắt vuông
            crop_size = int(min(args.resolution, orig_width, orig_height))
            
            # Tính toán tâm của bbox hiện tại
            box_cx = (x1 + x2) / 2
            box_cy = (y1 + y2) / 2
            
            # Tính điểm bắt đầu (trên-trái) của khung cắt, cố gắng giữ bbox ở giữa
            crop_x1 = int(box_cx - crop_size / 2)
            crop_y1 = int(box_cy - crop_size / 2)
            
            # Đảm bảo khung cắt không bị trượt ra ngoài viền ảnh
            if crop_x1 < 0:
                crop_x1 = 0
            elif crop_x1 + crop_size > orig_width:
                crop_x1 = orig_width - crop_size
                
            if crop_y1 < 0:
                crop_y1 = 0
            elif crop_y1 + crop_size > orig_height:
                crop_y1 = orig_height - crop_size
                
            crop_x2 = crop_x1 + crop_size
            crop_y2 = crop_y1 + crop_size
            
            # Thực hiện crop trên CẢ 2 ẢNH với cùng một toạ độ
            input_image = input_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            gt_image = gt_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            # Dịch chuyển toạ độ bbox về không gian của ảnh mới (đã crop)
            new_x1 = x1 - crop_x1
            new_y1 = y1 - crop_y1
            new_x2 = x2 - crop_x1
            new_y2 = y2 - crop_y1
            
            # Đề phòng trường hợp bbox ban đầu to hơn cả crop_size, ta "kẹp" (clip) toạ độ lại
            new_x1 = max(0, min(crop_size, new_x1))
            new_y1 = max(0, min(crop_size, new_y1))
            new_x2 = max(0, min(crop_size, new_x2))
            new_y2 = max(0, min(crop_size, new_y2))
            # ==========================================

            # 5. Scale Bbox về Resolution chuẩn bị đưa vào model
            # Vì ảnh hiện tại là vuông (crop_size x crop_size) nên tỉ lệ scale cho cả 2 trục là giống nhau
            scaled_x1 = int(new_x1 * args.resolution / crop_size)
            scaled_y1 = int(new_y1 * args.resolution / crop_size)
            scaled_x2 = int(new_x2 * args.resolution / crop_size)
            scaled_y2 = int(new_y2 * args.resolution / crop_size)
            bboxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])

            # 6. Xử lý Ground Truth thành tensor (VaeImageProcessor sẽ tự động scale từ crop_size lên args.resolution)
            pixel_values.append(image_processor.preprocess(gt_image, width=args.resolution, height=args.resolution).squeeze(0))

            # 4. Trích xuất Subject từ ảnh Ground Truth ĐÃ CROP
            subject_image = gt_image.crop((new_x1, new_y1, new_x2, new_y2))
            # Tạo padding vuông nền xám để giữ nguyên aspect ratio
            sub_w, sub_h = subject_image.size
            max_dim = max(sub_w, sub_h)
            square_subject = Image.new("RGB", (max_dim, max_dim), (127, 127, 127))
            paste_x = (max_dim - sub_w) // 2
            paste_y = (max_dim - sub_h) // 2
            square_subject.paste(subject_image, (paste_x, paste_y))
            subject_image = square_subject

            # --- 3. ÁP DỤNG AUGMENTATIONS ---
            # Giả định bạn truyền pg và pa qua args (ví dụ args.pg = 0.3, args.pa = 0.5)
            # Nếu chưa có trong args, bạn có thể hardcode tạm ở đây (VD: pg = 0.3, pa = 0.3)
            pg = getattr(args, 'pg', 0.5) 
            pa = getattr(args, 'pa', 0.5)

            # Geometric Augmentation
            if random.random() < pg:
                subject_image = geom_transforms(subject_image)
            
            # Appearance Augmentation
            if random.random() < pa:
                subject_image = color_transforms(subject_image)

            # --------------------------------

            # 7. Chuẩn bị các Conditions
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
            
            # 8. Tận dụng class_name làm Text Prompt
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