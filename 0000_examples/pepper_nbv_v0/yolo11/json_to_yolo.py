import json
import os
import shutil
from pathlib import Path


def json_to_yolo(json_path, output_dir, class_mapping, image_output_dir):
    """
    将JSON标注文件转换为YOLO格式的TXT文件，并将对应图片复制到指定目录
    参数:
        json_path: JSON文件路径
        output_dir: YOLO标签输出目录
        class_mapping: 类别名称到ID的映射字典
        image_output_dir: 图片输出目录
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']
    image_path = data.get('imagePath')

    yolo_lines = []

    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']

        if label not in class_mapping:
            print(f"警告: 跳过未定义类别 '{label}'")
            continue

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_lines.append(f"{class_mapping[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    if yolo_lines:
        output_path = Path(output_dir) / (Path(json_path).stem + '.txt')
        with open(output_path, 'w') as f:
            f.writelines(yolo_lines)

    if image_path:
        source_image_path = Path(json_path).parent / image_path
        target_image_path = Path(image_output_dir) / Path(image_path).name
        try:
            shutil.copy(source_image_path, target_image_path)
            print(f"图片已复制到: {target_image_path}")
        except Exception as e:
            print(f"复制图片失败: {source_image_path} -> {target_image_path} 错误: {e}")


def convert_dataset(json_dir, output_dir, image_output_dir, class_mapping):
    """
    批量转换JSON文件，并复制对应图片到指定目录
    参数:
        json_dir: JSON文件目录
        output_dir: YOLO标签输出目录
        image_output_dir: 图片输出目录
        class_mapping: 类别映射字典
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    if not json_files:
        print(f"在 {json_dir} 中没有找到JSON文件")
        return

    print(f"开始转换 {len(json_files)} 个文件...")

    for i, json_file in enumerate(json_files, 1):
        json_path = os.path.join(json_dir, json_file)
        json_to_yolo(json_path, output_dir, class_mapping, image_output_dir)
        print(f"转换进度: {i}/{len(json_files)} - {json_file}")

    print(f"转换完成! 标签保存在: {output_dir}，图片保存在: {image_output_dir}")


if __name__ == "__main__":
    # 配置路径
    JSON_DIR = r"F:\WRS\wrs-virtual\0000_examples\pepper\yolo11\JPG_DATA"  # JSON文件目录
    OUTPUT_DIR = r"F:\WRS\wrs-virtual\0000_examples\pepper\yolo11\labels"  # 标签输出目录
    IMAGE_OUTPUT_DIR = r"F:\WRS\wrs-virtual\0000_examples\pepper\yolo11\images"  # 图片输出目录

    # 类别映射
    CLASS_MAPPING = {
        "pepper": 0,
        "stem": 1
    }

    # 执行转换
    convert_dataset(JSON_DIR, OUTPUT_DIR, IMAGE_OUTPUT_DIR, CLASS_MAPPING)