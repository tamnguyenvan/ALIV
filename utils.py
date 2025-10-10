import os
import os.path as osp

import numpy as np


def get_experiment_id(debug=False):
    if debug:
        return 999999999999
    import time
    time.sleep(0.5)
    return int(time.time()*100)

def get_path_schedule(type, **kwargs):
    if type == 'repeat':
        max_path = kwargs['max_path']
        schedule_each = kwargs['schedule_each']
        return [schedule_each] * max_path
    elif type == 'list':
        schedule = kwargs['schedule']
        return schedule
    elif type == 'exp':
        import math
        base = kwargs['base']
        max_path = kwargs['max_path']
        max_path_per_iter = kwargs['max_path_per_iter']
        schedule = []
        cnt = 0
        while sum(schedule) < max_path:
            proposed_step = min(
                max_path - sum(schedule), 
                base**cnt, 
                max_path_per_iter)
            cnt += 1
            schedule += [proposed_step]
        return schedule
    else:
        raise ValueError

def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append( edict_2_dict(x[i]) )
        return xnew
    else:
        return x

def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)
def write_image_detail_txt(para, content, save_path, pic_name):
    # 要写入的文件名
    file = save_path + pic_name + "_detail.txt"

    # 使用with语句确保文件正常关闭
    with open(file, para, encoding='utf-8') as file:
        file.write(content)  # 写入内容
    return 0

def find_opaque_color(image_array):
    # 获取图像的 alpha 通道
    alpha_channel = image_array[:, :, 3]

    # 寻找 alpha 通道中不为 0 的位置
    opaque_indices = np.where(alpha_channel != 0)

    # 如果存在不透明像素
    if len(opaque_indices[0]) > 0:
        # 获取第一个不透明像素的位置
        opaque_pixel_index = (opaque_indices[0][0], opaque_indices[1][0])

        # 获取该像素的 RGB 值
        opaque_color = image_array[opaque_pixel_index[0], opaque_pixel_index[1], :3]

        return opaque_color
    else:
        return None

def get_image_name_without_extension(image_path):
    name_with_extension = os.path.basename(image_path)
    return os.path.splitext(name_with_extension)[0]
def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)
