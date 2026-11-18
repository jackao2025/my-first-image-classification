import os

# 你的目标文件夹绝对路径 (这个可以自行修改)
folder_path = "/home/raid/ayx/self_learning_CCNU/image_classification/my-first-image-classification/dataset/train/dog"

# 支持的图片后缀（全部转为小写比较）
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

def batch_rename():
    if not os.path.exists(folder_path):
        print(f"错误: 找不到文件夹 {folder_path}")
        return

    # 1. 获取目录下所有的图片文件
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not files:
        print("文件夹中没有找到图片文件！")
        return

    # 排序，保证重命名的顺序是稳定的（按原文件名）
    files.sort()
    
    print(f"找到 {len(files)} 张图片，开始重命名...")

    # 2. 第一阶段：先重命名为临时文件名（防止原文件名和目标新文件名冲突覆盖）
    temp_files = []
    for i, filename in enumerate(files):
        old_path = os.path.join(folder_path, filename)
        # 获取原后缀（如 .jpg 或 .png），保持原样
        ext = os.path.splitext(filename)[1] 
        
        # 临时命名：__tmp_0.jpg, __tmp_1.png ...
        temp_name = f"__tmp_{i}{ext}"
        temp_path = os.path.join(folder_path, temp_name)
        
        os.rename(old_path, temp_path)
        temp_files.append((temp_name, ext))

    # 3. 第二阶段：将临时文件重命名为最终的 cat001, cat002...（或者别的名字，可以自行修改）
    for i, (temp_name, ext) in enumerate(temp_files):
        temp_path = os.path.join(folder_path, temp_name)
        
        # 格式化新名字，03d 表示不足3位补零（如 001）
        # 如果图片超过 999 张，会自动变成 1000, 1001
        new_name = f"dog{i+1:03d}{ext}"
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(temp_path, new_path)

    print(f"重命名完成！共处理 {len(files)} 张图片。")

if __name__ == "__main__":
    batch_rename()