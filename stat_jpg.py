from PIL import Image
import os

# 指定目录路径
directory = "/ML-A100/sshare-app/saiwanming/workdir/data/laion-high/000"

# 创建一个字典来存储不同尺寸的文件数量
size_count = {}

# 遍历目录中的文件
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        # 获取文件的完整路径
        file_path = os.path.join(directory, filename)
        
        # 打开图像并获取其尺寸
        with Image.open(file_path) as img:
            width, height = img.size
            size = f"{width}x{height}"
            
            # 统计不同尺寸的文件数量
            if size in size_count:
                size_count[size] += 1
            else:
                size_count[size] = 1

# 打印不同尺寸的文件数量
for size, count in size_count.items():
    print(f"Size: {size} - Count: {count}")
