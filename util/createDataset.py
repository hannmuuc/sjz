

import os
import cv2
import json
import glob
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple  # 添加类型注解

def create_balanced_dataset(
    train_size=10000,
    val_size=2000,
    hit_ratio=0.5,
    min_scale=0.8,
    max_scale=1.2,
    crop_ratio=(0.8, 1.0),  # 随机裁剪比例范围
    flip_prob=0.5,          # 水平反转概率
    noise_prob=0.3,         # 加噪声概率
    noise_level=(0, 20),    # 噪声强度范围
    root_path="./dataset",
    hit_annotation_json="./dataset/target/hit/result.json"  # 新增：hit类标注文件路径
):
    """
    创建平衡的检测数据集，hit类样本通过JSON文件获取目标框，no_hit类保持原有逻辑
    """
    # 计算各类别样本数（确保1:1平衡）
    train_hit = int(train_size * hit_ratio)
    train_nohit = train_size - train_hit
    val_hit = int(val_size * hit_ratio)
    val_nohit = val_size - val_hit

    print(f"数据集配置：")
    print(f"训练集：{train_size}张（hit: {train_hit}, no_hit: {train_nohit}）")
    print(f"验证集：{val_size}张（hit: {val_hit}, no_hit: {val_nohit}）")
    print(f"随机缩放范围：{min_scale}-{max_scale}x")
    print(f"随机裁剪比例：{crop_ratio[0]}-{crop_ratio[1]}")
    print(f"水平反转概率：{flip_prob}")
    print(f"加噪声概率：{noise_prob}，强度：{noise_level[0]}-{noise_level[1]}")
    print(f"hit类标注文件路径：{hit_annotation_json}")

    # 输出目录设置
    annotations_path = os.path.join(root_path, "annotations")
    images_path = os.path.join(root_path, "images")
    os.makedirs(annotations_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    # --------------------------
    # 新增：读取hit类标注JSON文件
    # --------------------------
    if not os.path.exists(hit_annotation_json):
        raise FileNotFoundError(f"未找到hit类标注文件！路径：{hit_annotation_json}")
    
    with open(hit_annotation_json, "r", encoding="utf-8") as f:
        hit_annotations = json.load(f)
    
    # 验证JSON格式有效性
    if "images" not in hit_annotations:
        raise ValueError("hit类标注JSON格式错误，缺少'images'字段")
    
    hit_source_annotations = hit_annotations["images"]
    if not hit_source_annotations:
        raise ValueError("hit类标注JSON中'images'列表为空")
    
    # 检查标注文件中的图像是否存在
    for ann in hit_source_annotations:
        if not os.path.exists(ann["original_path"]):
            raise FileNotFoundError(f"标注文件中图像不存在：{ann['original_path']}")
    
    print(f"已加载hit类标注：{len(hit_source_annotations)} 条记录")

    # 读取no_hit类原始图像（保持原有逻辑）
    nohit_source = glob.glob(os.path.join(root_path, "source", "nohit", "image", "*.jpg")) + \
                   glob.glob(os.path.join(root_path, "source", "nohit", "image", "*.png"))

    if not nohit_source:
        raise ValueError(f"未找到no_hit类原始图像！请确认路径：{os.path.join(root_path, 'source', 'nohit', 'image')}")
    print(f"已加载no_hit类原始图像：{len(nohit_source)} 张")

    # COCO格式基础结构
    def init_coco_dataset():
        return {
            "info": {
                "description": "Balanced Detection Dataset (hit from JSON, no_hit auto-generated)",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "categories": [
                {"supercategory": "none", "id": 1, "name": "hit"},
                {"supercategory": "none", "id": 2, "name": "no_hit"}
            ],
            "images": [],
            "annotations": []
        }

    # 初始化训练集和验证集
    train_dataset = init_coco_dataset()
    val_dataset = init_coco_dataset()

    # 图像增强函数（保持原有逻辑）
    def random_resize(image):
        scale = random.uniform(min_scale, max_scale)
        h, w = image.shape[:2]
        new_h = int(h * scale)
        new_w = int(w * scale)
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        return cv2.resize(image, (new_w, new_h)), scale

    def random_crop_standard(image):
        h, w = image.shape[:2]
        crop_scale = random.uniform(crop_ratio[0], crop_ratio[1])
        new_h = int(h * crop_scale)
        new_w = int(w * crop_scale)
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        x = random.randint(0, w - new_w)
        y = random.randint(0, h - new_h)
        cropped = image[y:y+new_h, x:x+new_w]
        return cropped, (x, y, new_w, new_h)

    def random_crop_with_bbox_guarantee(image, bbox):
        """确保目标框完全在裁剪范围内（适配从JSON读取的box）"""
        h, w = image.shape[:2]
        bbox_x, bbox_y, bbox_w, bbox_h = bbox  # bbox格式：(x1, y1, x2, y2) -> 转换为(x, y, w, h)
        bbox_x1, bbox_y1 = bbox_x, bbox_y
        bbox_x2, bbox_y2 = bbox_w, bbox_h
        bbox_width = bbox_x2 - bbox_x1
        bbox_height = bbox_y2 - bbox_y1

        # 1. 确保裁剪尺寸至少能容纳目标框
        crop_scale = random.uniform(crop_ratio[0], crop_ratio[1])
        min_crop_w = max(int(w * crop_scale), bbox_width)
        min_crop_h = max(int(h * crop_scale), bbox_height)

        # 2. 计算裁剪区域有效范围
        max_x1 = min(bbox_x1, w - min_crop_w)
        min_x1 = max(0, bbox_x2 - min_crop_w)
        max_y1 = min(bbox_y1, h - min_crop_h)
        min_y1 = max(0, bbox_y2 - min_crop_h)

        # 3. 随机选择裁剪起点
        x1 = random.randint(min_x1, max_x1) if min_x1 <= max_x1 else 0
        y1 = random.randint(min_y1, max_y1) if min_y1 <= max_y1 else 0

        # 4. 执行裁剪
        x2 = x1 + min_crop_w
        y2 = y1 + min_crop_h
        x2 = min(x2, w)
        y2 = min(y2, h)

        cropped = image[y1:y2, x1:x2]
        return cropped, (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)

    def random_flip(image):
        if random.random() < flip_prob:
            return cv2.flip(image, 1), True
        return image, False

    def add_random_noise(image):
        if random.random() < noise_prob:
            row, col, ch = image.shape
            mean = 0
            var = random.randint(noise_level[0], noise_level[1])
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image + gauss
            return np.clip(noisy, 0, 255).astype(np.uint8), True
        return image, False

    # --------------------------
    # 生成样本函数（核心修改）
    # --------------------------
    def generate_hit_samples(
        num_samples: int,
        source_annotations: List[Dict],
        dataset: Dict,
        start_image_id: int,
        start_annotation_id: int
    ) -> Tuple[Dict, int, int]:
        """生成hit类样本（从JSON标注读取目标框）"""
        image_id = start_image_id
        annotation_id = start_annotation_id
        valid_samples = 0
        num_annotations = len(source_annotations)

        while valid_samples < num_samples:
            # 循环使用标注数据（支持重复采样）
            ann_idx = valid_samples % num_annotations
            ann = source_annotations[ann_idx]
            img_path = ann["original_path"]
            box = ann["box"]  # [x1, y1, x2, y2]

            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取hit类图像 {img_path}，已跳过")
                valid_samples += 1
                continue

            # 验证原始目标框有效性
            h, w = img.shape[:2]
            x1, y1, x2, y2 = box
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or (x2 - x1) <= 0 or (y2 - y1) <= 0:
                print(f"警告：无效目标框 {box} 在图像 {img_path}，已跳过")
                valid_samples += 1
                continue

            # 1. 随机缩放
            img_scaled, scale = random_resize(img)
            scaled_h, scaled_w = img_scaled.shape[:2]

            # 2. 缩放原始目标框（适配缩放后的图像）
            scaled_x1 = int(x1 * scale)
            scaled_y1 = int(y1 * scale)
            scaled_x2 = int(x2 * scale)
            scaled_y2 = int(y2 * scale)
            scaled_bbox = (scaled_x1, scaled_y1, scaled_x2, scaled_y2)  # (x1, y1, x2, y2)

            # 3. 智能裁剪（确保目标框完整）
            img_cropped, crop_info = random_crop_with_bbox_guarantee(img_scaled, scaled_bbox)
            crop_x, crop_y, crop_w, crop_h = crop_info  # (x, y, w, h)
            cropped_h, cropped_w = img_cropped.shape[:2]

            # 4. 随机翻转
            img_flipped, flipped = random_flip(img_cropped)

            # 5. 随机加噪声
            img_final, noised = add_random_noise(img_flipped)

            # 生成唯一文件名
            base_name = ann["filename"]
            new_filename = f"hit_{image_id}_{base_name}.jpg"
            dest_path = os.path.join(images_path, new_filename)
            cv2.imwrite(dest_path, img_final)

            # 添加图像信息到COCO标注
            dataset["images"].append({
                "file_name": new_filename,
                "height": cropped_h,
                "width": cropped_w,
                "id": image_id
            })

            # 计算裁剪+翻转后的目标框坐标
            # 裁剪后坐标：原始缩放框 - 裁剪起点
            cropped_x1 = scaled_x1 - crop_x
            cropped_y1 = scaled_y1 - crop_y
            cropped_x2 = scaled_x2 - crop_x
            cropped_y2 = scaled_y2 - crop_y

            # 应用翻转变换
            if flipped:
                final_x1 = cropped_w - cropped_x2  # 水平翻转后x坐标转换
                final_y1 = cropped_y1
                final_x2 = cropped_w - cropped_x1
                final_y2 = cropped_y2
            else:
                final_x1 = cropped_x1
                final_y1 = cropped_y1
                final_x2 = cropped_x2
                final_y2 = cropped_y2

            # 转换为COCO格式的bbox [x, y, w, h]
            coco_bbox = [
                final_x1,
                final_y1,
                final_x2 - final_x1,
                final_y2 - final_y1
            ]

            # 添加标注
            dataset["annotations"].append({
                "area": (final_x2 - final_x1) * (final_y2 - final_y1),
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": coco_bbox,
                "category_id": 1,  # hit类固定ID
                "id": annotation_id,
                "ignore": 0,
                "segmentation": []
            })

            # 更新计数和ID
            valid_samples += 1
            image_id += 1
            annotation_id += 1
            if valid_samples % 100 == 0:
                print(f"已生成 {valid_samples}/{num_samples} 个hit类样本")

        return dataset, image_id, annotation_id

    def generate_nohit_samples(
        num_samples: int,
        source_images: List[str],
        dataset: Dict,
        start_image_id: int,
        start_annotation_id: int
    ) -> Tuple[Dict, int, int]:
        """生成no_hit类样本（保持原有逻辑）"""
        image_id = start_image_id
        annotation_id = start_annotation_id
        valid_samples = 0
        source_len = len(source_images)

        while valid_samples < num_samples:
            img_idx = valid_samples % source_len
            img_path = source_images[img_idx]
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取no_hit类图像 {img_path}，已跳过")
                valid_samples += 1
                continue

            # 1. 随机缩放
            img_scaled, _ = random_resize(img)
            
            # 2. 标准裁剪
            img_cropped, _ = random_crop_standard(img_scaled)
            
            # 3. 随机翻转
            img_flipped, _ = random_flip(img_cropped)
            
            # 4. 随机加噪声
            img_final, _ = add_random_noise(img_flipped)

            # 生成唯一文件名
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            new_filename = f"nohit_{image_id}_{base_name}.jpg"
            dest_path = os.path.join(images_path, new_filename)
            cv2.imwrite(dest_path, img_final)

            # 添加图像信息（no_hit类无标注）
            cropped_h, cropped_w = img_final.shape[:2]
            dataset["images"].append({
                "file_name": new_filename,
                "height": cropped_h,
                "width": cropped_w,
                "id": image_id
            })

            # 更新计数和ID
            valid_samples += 1
            image_id += 1
            if valid_samples % 100 == 0:
                print(f"已生成 {valid_samples}/{num_samples} 个no_hit类样本")

        return dataset, image_id, annotation_id

    # --------------------------
    # 生成数据集（主流程）
    # --------------------------
    print("\n===== 生成训练集 =====")
    # 生成训练集hit类（从JSON读取）
    train_dataset, img_id, ann_id = generate_hit_samples(
        num_samples=train_hit,
        source_annotations=hit_source_annotations,
        dataset=train_dataset,
        start_image_id=1,
        start_annotation_id=1
    )
    # 生成训练集no_hit类（原有逻辑）
    train_dataset, img_id, ann_id = generate_nohit_samples(
        num_samples=train_nohit,
        source_images=nohit_source,
        dataset=train_dataset,
        start_image_id=img_id,
        start_annotation_id=ann_id
    )

    print("\n===== 生成验证集 =====")
    # 生成验证集hit类（从JSON读取）
    val_dataset, img_id, ann_id = generate_hit_samples(
        num_samples=val_hit,
        source_annotations=hit_source_annotations,
        dataset=val_dataset,
        start_image_id=img_id,
        start_annotation_id=ann_id
    )
    #生成验证集 no_hit 类（原有逻辑）
    val_dataset, img_id, ann_id = generate_nohit_samples(
    num_samples=val_nohit,
    source_images=nohit_source,
    dataset=val_dataset,
    start_image_id=img_id,
    start_annotation_id=ann_id
    )
    #保存 COCO 格式标注文件
    train_json = os.path.join(annotations_path, "instances_train.json")
    val_json = os.path.join(annotations_path, "instances_val.json")
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=2)
    # 输出结果统计
    print ("\n===== 数据集生成完成 =====")
    print (f"训练集图像数：{len (train_dataset ['images'])}")
    print (f"训练集标注数（hit 类）：{len (train_dataset ['annotations'])}")
    print (f"验证集图像数：{len (val_dataset ['images'])}")
    print (f"验证集标注数（hit 类）：{len (val_dataset ['annotations'])}")
    print (f"标注文件路径：{annotations_path}")
    print (f"图像存放路径：{images_path}")
    return train_dataset, val_dataset

def draw_bboxes_from_coco(
    annotation_file, 
    images_dir, 
    output_dir, 
    show_label=True, 
    color=(0, 255, 0),  # 目标框颜色，BGR格式
    thickness=2         # 目标框线宽
):
    """从COCO格式标注文件绘制目标框"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    image_map = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    for img_id, img_info in image_map.items():
        img_filename = img_info['file_name']
        img_path = os.path.join(images_dir, img_filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图片 {img_path}，已跳过")
            continue
        
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                bbox = ann['bbox']
                x, y, w, h = map(int, bbox)
                category_id = ann['category_id']
                category_name = category_map.get(category_id, f"类别{category_id}")
                
                # 绘制矩形框
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
                
                # 显示标签
                if show_label:
                    label = f"{category_name}"
                    # 计算文本位置（在框的上方）
                    text_x = x
                    text_y = y - 10 if y > 10 else y + 10
                    
                    # 绘制文本背景（增加可读性）
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        image, 
                        (text_x, text_y - text_height - 5), 
                        (text_x + text_width + 5, text_y + 5), 
                        color, 
                        -1  # 填充矩形
                    )
                    
                    # 绘制文本
                    cv2.putText(
                        image, 
                        label, 
                        (text_x + 3, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255),  # 白色文字
                        1
                    )
        
        # 保存绘制后的图片
        output_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(output_path, image)
    
    print(f"所有图片已处理完成，结果保存在：{output_dir}")

# 使用示例
if __name__ == "__main__":
    # 生成数据集
    create_balanced_dataset(
        train_size=10000,  # 训练集10000张
        val_size=2000,     # 验证集2000张
        hit_ratio=0.5,     # 1:1比例
        min_scale=0.8,     # 最小缩放0.8x
        max_scale=1.2,     # 最大缩放1.2x
        crop_ratio=(0.8, 1.0),  # 随机裁剪比例
        flip_prob=0.5,     # 50%概率水平反转
        noise_prob=0.3,    # 30%概率添加噪声
        noise_level=(0, 20)# 噪声强度范围
    )

    # 可视化验证集目标框
    annotation_file = "./dataset/annotations/instances_val.json"  # 标注文件路径
    images_dir = "./dataset/images"                              # 原始图片目录
    output_dir = "./dataset/annotations_visualization"           # 结果保存目录
    
    draw_bboxes_from_coco(
        annotation_file=annotation_file,
        images_dir=images_dir,
        output_dir=output_dir,
        show_label=True,
        color=(0, 255, 0),  # 绿色框
        thickness=2
    )
