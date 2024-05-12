# """
#划分数据集并保存到本地
import os
import shutil
from sklearn.model_selection import train_test_split

current_directory = os.path.dirname(__file__)
dataset_root = current_directory + "/malimg_dataset"
# 数据集根目录路径
data_dir = dataset_root + "/sort_dataset/"

# 划分后的训练集和验证集保存路径
train_dir = dataset_root + "/Imb_dataset/train_dataset/"
test_dir = dataset_root + "/Imb_dataset/val_dataset/"

# 获取所有类别文件夹路径
class_folders = [os.path.join(data_dir, cls) for cls in sorted(os.listdir(data_dir))]

# 创建训练集和验证集文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 遍历每个类别文件夹
for class_folder in class_folders:
    class_name = os.path.basename(class_folder)
    # 获取类别下的所有图像文件路径
    image_files = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
    # 进行分层划分
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42, stratify=None)

    # 将训练集样本复制到训练集文件夹
    for train_file in train_files:
        train_dst = os.path.join(train_dir, class_name, os.path.basename(train_file))
        os.makedirs(os.path.dirname(train_dst), exist_ok=True)
        shutil.copy(train_file, train_dst)

    # 将验证集样本复制到验证集文件夹
    for test_file in test_files:
        test_dst = os.path.join(test_dir, class_name, os.path.basename(test_file))
        os.makedirs(os.path.dirname(test_dst), exist_ok=True)
        shutil.copy(test_file, test_dst)
# """

"""
#按样本顺序排列数据集
import os
import shutil
from torchvision.datasets import ImageFolder
from collections import Counter

current_directory = os.path.dirname(__file__)
dataset_root = current_directory + "/malimg_dataset"
# 数据集根目录路径
data_dir = dataset_root + "/dataset/"

# 新的保存文件夹路径
new_data_dir = dataset_root + "/sort_dataset/"

# 使用ImageFolder加载数据集
dataset = ImageFolder(data_dir)

# 统计每个类的样本数量
class_sample_counts = Counter(dataset.targets)

# 按样本数量降序排序
sorted_classes = sorted(class_sample_counts.items(), key=lambda x: x[1], reverse=True)

# 创建新的保存文件夹
os.makedirs(new_data_dir, exist_ok=True)

# 计数器变量
count = 1

# 遍历排序后的类别
for class_index, _ in sorted_classes:
    class_name = dataset.classes[class_index]
    original_class_folder = os.path.join(data_dir, class_name)
    new_class_folder_name = f"{count:02d}_{class_name}"  # 保持文件夹顺序和样本数量顺序一致
    new_class_folder_path = os.path.join(new_data_dir, new_class_folder_name)
    os.makedirs(new_class_folder_path, exist_ok=True)

    # 复制原始子文件夹下的图像数据集到新的子文件夹
    for img_file in os.listdir(original_class_folder):
        img_src = os.path.join(original_class_folder, img_file)
        img_dst = os.path.join(new_class_folder_path, img_file)
        shutil.copy(img_src, img_dst)

    # 更新计数器变量
    count += 1
"""

"""
#从每个类别中随机选择25个样本保存到一个文件夹中，剩下的样本保存到另一个文件夹中
import os
import random
import shutil
from torchvision.datasets import ImageFolder

current_directory = os.path.dirname(__file__)
dataset_root = current_directory + "/malimg_dataset"
# 数据集根目录路径
data_dir = dataset_root + "/sort_dataset/"

# 保存样本的文件夹路径
selected_data_dir = dataset_root + "/bal_dataset/val/"
remaining_data_dir = dataset_root + "/bal_dataset/train/"

# 随机选择的样本数
num_selected_samples = 25

# 使用ImageFolder加载数据集
dataset = ImageFolder(data_dir)

# 创建保存样本的文件夹
os.makedirs(selected_data_dir, exist_ok=True)
os.makedirs(remaining_data_dir, exist_ok=True)

# 遍历每个类别
for class_index in range(len(dataset.classes)):
    class_name = dataset.classes[class_index]
    class_folder = os.path.join(data_dir, class_name)

    # 获取该类别下的所有图像文件路径
    image_paths = [os.path.join(class_folder, img_file) for img_file in os.listdir(class_folder)]

    # 随机选择指定数量的样本
    selected_samples = random.sample(image_paths, num_selected_samples)
    remaining_samples = [img_path for img_path in image_paths if img_path not in selected_samples]

    # 将选中的样本复制到保存样本的文件夹
    for img_path in selected_samples:
        img_file = os.path.basename(img_path)
        img_dst = os.path.join(selected_data_dir, class_name, img_file)  # 保持样本属于原本的类别子文件夹
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        shutil.copy(img_path, img_dst)

    # 将剩余的样本复制到剩余样本的文件夹
    for img_path in remaining_samples:
        img_file = os.path.basename(img_path)
        img_dst = os.path.join(remaining_data_dir, class_name, img_file)  # 保持样本属于原本的类别子文件夹
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        shutil.copy(img_path, img_dst)
"""