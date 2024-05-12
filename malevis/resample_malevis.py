#resample malevis dataset的训练集
#考虑到：test集中other类的数量偏大，所以resample后的train集中，other类别的样本最多-350
import os
import sys
import random
import shutil

#输出保存到本地
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

current_directory = os.path.dirname(__file__)
root_1 = current_directory + "/log/resample_malevis.log"
root_2 = current_directory + "/log/resample_malevis.log_file"

sys.stdout = Logger(root_1, sys.stdout)
sys.stderr = Logger(root_2, sys.stderr)

# 数据集根目录路径
current_directory = os.path.dirname(__file__)
dataset_root = current_directory + "/malevis_train_val_300x300/train"

# 子文件夹列表，每个子文件夹表示一个类别
class_folders = sorted(os.listdir(dataset_root))

# 每个类别需要读取的数据数量列表
# data_counts = [350, 299, 255, 218, 187, 160, 136, 117, 100, 85, 73, 62, 53,
#                45, 39, 33, 28, 24, 20, 17, 15, 13, 11, 9, 8, 7] #imb_factor = 0.02
# data_counts = [350, 291, 242, 201, 167, 139, 115, 96, 80, 66, 55, 46, 38, 31, 26, 22, 18, 15, 12, 10, 8, 7, 6, 5, 4, 3] #imb_factor = 0.01
data_counts = [350, 319, 291, 265, 242, 220, 201, 183, 167, 152, 139, 127, 115, 105, 96, 87, 80, 73, 66, 60, 55, 50, 46, 42, 38, 35] #imb_factor = 0.1 sum = 3604

# 保存数据集的文件夹路径
output_dir = current_directory + "/malevis_train_val_300x300/train_10"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个子文件夹
for i, class_folder in enumerate(class_folders):
    class_name = os.path.basename(class_folder)
    class_path = os.path.join(dataset_root, class_folder)

    # 获取该类别下的所有图像文件路径
    image_files = [os.path.join(class_path, img) for img in os.listdir(class_path)]

    # 获取当前类别需要读取的数据数量
    data_count = data_counts[i]

    # 如果数据数量大于实际可用的数据数量，则取实际可用的数据数量
    if data_count > len(image_files):
        data_count = len(image_files)

    # 从图像文件列表中随机选择相应数量的数据
    selected_files = random.sample(image_files, data_count)

    # 将选中的数据复制到输出文件夹中
    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for file in selected_files:
        shutil.copy(file, output_class_dir)

    # print(f"Class {class_name}: Selected {data_count} samples and saved to {output_class_dir}")