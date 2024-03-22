import os
import shutil
import numpy as np

def partition():
   
    # 数据集的路径
    data_dir = 'dataset'

    # 创建训练、验证和测试目录
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 定义数据划分比例
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # 对每个类别进行处理
    for category in ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']:
        category_dir = os.path.join(data_dir, category)
        all_images = os.listdir(category_dir)
        np.random.shuffle(all_images)

        # 划分数据集
        train_split = int(len(all_images) * train_ratio)
        val_split = train_split + int(len(all_images) * val_ratio)

        # 创建子目录
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # 分配数据
        for i, img in enumerate(all_images):
            src = os.path.join(category_dir, img)
            if i < train_split:
                dst = os.path.join(train_dir, category, img)
            elif i < val_split:
                dst = os.path.join(val_dir, category, img)
            else:
                dst = os.path.join(test_dir, category, img)
            shutil.copyfile(src, dst)

partition()