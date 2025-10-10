# start from 24.01.16 end on 02
import time

# 得到单色图层
import array
import datetime
import itertools
import math
import torch
import numpy as np
import cv2

from PIL import Image
import colorsys

# 计算rgb值 hex
from utils import get_image_name_without_extension, write_image_detail_txt, check_and_create_dir
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



def rgb_2_hex(r, g, b):
    hex_color = '{:02x}{:02x}{:02x}'.format(r.item(), g.item(), b.item())
    return hex_color


# 还原rgb ox
def hex_2_rgb(hex_value):
    # 去除前面的0x号（如果有）
    if hex_value.startswith("0x"):
        hex_value = hex_value[2:]
    # 去除前面的空格（如果有）
    if hex_value.startswith(" "):
        hex_value = hex_value[1:]

    # 分割成红、绿、蓝三部分并转换为十进制数字
    red = int(hex_value[:2], 16)
    green = int(hex_value[2:4], 16)
    blue = int(hex_value[4:], 16)

    return torch.tensor(red), torch.tensor(green), torch.tensor(blue)


# 统计图片有多少种颜色，每个颜色数量是多少,并记录该颜色像素坐标
# 图片路径 保存路径
def pic_color_count(pic_obj, pic_path, save_path, save):
    if pic_obj is None:
        pic = Image.open(pic_path)

    else:
        pic = pic_obj
    # 图片
    pic_name = get_image_name_without_extension(pic_path)

    # 建立新字典
    color_dict = dict()
    # 图片转tensor
    pic_array = torch.tensor(np.asarray(pic))

    # 遍历图片像素
    for y in range(pic.shape[1]):
        for x in range(pic.shape[0]):
            # 当前像素点
            r, g, b = pic_array[x][y][0:3]
            # 转hex
            hex_color = rgb_2_hex(r, g, b)
            # 如果字典有该颜色
            if hex_color in color_dict:
                # 该颜色像素数量加一,数量放在第一个元素
                color_dict[hex_color][0] += 1
            else:
                # 如果没有该颜色 ，初始化该颜色的字典值：列表
                color_dict[hex_color] = []
                # 将该色值插入列表第一个元素，元素值为1
                color_dict[hex_color].append(1)
            # 记录该像素的坐标到字典对应元素的第一个元素后面
            color_dict[hex_color].append([x, y])
    content_ = "图片名称：" + pic_name + "  图片大小：" + str(pic.shape[0]) + "," + str(pic.shape[1]) + "\n"
    content = "图片共" + str(len(color_dict.keys())) + "种像素\n" + content_
    print(content, end="")
    if save:
        write_image_detail_txt('w', content, save_path, pic_name)
    return color_dict


# 用给定rgb差值将颜色进行合并 将颜色从rgb值域映射到自定义的域
# 输入粒度，图片路径 返回更新后的dict和pic
# rgb差值为4肉眼还不太分辨
def pic_color_mix(color_distance, pic_obj, pic_path, save_path, save):
    if pic_obj is None:
        pic = Image.open(pic_path).convert("RGBA")
    else:
        pic = pic_obj

    pic_name = get_image_name_without_extension(pic_path)
    # 图片
    pic = torch.tensor(np.asarray(pic))
    count = 0
    # 将图片进行颜色统计然后按照颜色值进行降序得到list
    # 为什么不在该循环直接就mix，是因为要先比较颜色的相似度，这里用rgb值做排序就是在度量颜色的相似度
    old_list = sorted(pic_color_count(pic, pic_name, save_path, save).items(), key=lambda x2: -int(x2[0], 16))
    # 新建一个dict用来存放颜色融合后的dict
    new_dict = dict()
    # 临时计数变量 i指向当前新的颜色类别，j指向当前遍历颜色
    i = 0
    j = 0
    # 原图数组
    pic_array = torch.tensor(np.asarray(pic).copy())
    # 遍历list 用给定阈值进行融合
    while j < len(old_list):
        # 当前颜色类别
        now_color = old_list[i][0]
        # 当前颜色值
        now_color_value = int(now_color, 16)
        # 字典新类初始化
        if now_color not in new_dict:
            # 当前字典新颜色
            new_dict[now_color] = []
            # 新类数量置零
            new_dict[now_color].append(0)
        # 待比较的颜色
        tmp_color = old_list[j][0]
        # 待比较的颜色值
        tmp_color_value = int(tmp_color, 16)
        # 小于阈值就合并，i == j是新类初始 差值为0
        if now_color_value - tmp_color_value <= color_distance:
            # 像素数量累加
            new_dict[now_color][0] += old_list[j][1][0]
            # 记录像素位置
            new_dict[now_color] += old_list[j][1][1:]
            # 并把原图该类像素颜色也置为新类
            # 遍历该类坐标
            for item in old_list[j][1][1:]:
                # 当前坐标
                x = item[0]
                y = item[1]
                # 当前大类颜色rgb
                r, g, b = hex_2_rgb(now_color)
                m, n, p = pic_array[x][y][0:3]
                if now_color != rgb_2_hex(m, n, p):
                    count += 1
                # 将当前像素颜色改为大类颜色
                pic_array[x][y] = torch.tensor([r, g, b, 255])

            # 指针移动
            j += 1
        # 判定为新类开始
        else:
            # 指针移动
            i = j
    if save:
        Image.fromarray(pic_array.cpu().numpy()).save(
            save_path + pic_name + "_mix_color" + str(color_distance) + ".png")
    content = "合并了" + str(count / (pic.shape[0] * pic.shape[1])) + "比率数量的像素，共" + str(
        count) + "个像素，使用的颜色差值阈值为" + str(color_distance) + "\n"

    print(content, end="")
    if save:
        write_image_detail_txt('a', content, save_path, pic_name)
    return new_dict, pic_array


# 将新的颜色分类进行过滤，根据阈值删去数量极少的颜色类别
# 数量阈值，颜色差值，图片名
# 根据图片的特点进行过滤阈值选取
def pic_color_filter(num_threshold, color_distance, pic_obj, pic_path, save_path, save):
    pic_name = get_image_name_without_extension(pic_path)
    if pic_obj is None:
        pic = Image.open(pic_path).convert('RGBA')
        pic = np.asarray(pic)
        pic = torch.tensor(pic)
    else:
        pic = pic_obj
    # 总像素数量
    total = pic.shape[0] * pic.shape[1]
    # 像素过滤数量阈值
    rule = 0
    # 计数值
    count = 0
    # 如果threshold 是 float
    if type(num_threshold) == float:
        # 要替换的像素数量标准就要大于总数*threshold
        rule = total * num_threshold
    if type(num_threshold) == int:
        # 如果输入是整形就直接用
        rule = num_threshold
    # 要返还的字典和图片rgba数组
    res_ = pic_color_mix(color_distance, pic, pic_path, save_path, save)
    color_dict = res_[0]
    pic_arr = res_[1]

    # 非主成分类
    color_dict["none"] = []
    color_dict["none"].append(0)
    color_dict["none"].append([])
    # 扫描列表
    for key in color_dict.copy():
        # 如果不是主成分
        if key != "none" and color_dict[key][0] < rule:
            # 将其记录到非主成分
            color_dict["none"][0] += color_dict[key][0]
            color_dict["none"][1].extend(color_dict[key][1:])
            # 在原图数组上将该像素设置为透明
            for item in color_dict[key][1:]:
                x = item[0]
                y = item[1]
                pic_arr[x][y][3] = torch.tensor(0)
                count += 1
            # 删除该元素
            del color_dict[key]
    colors = []
    for c in color_dict.keys():
        # print(c)
        if c != 'none':
            colors.append(c)
    if save:
        Image.fromarray(pic_arr.cpu().numpy()).save(
            save_path + pic_name + "_filter_" + str(num_threshold) + "_mix_" + str(color_distance) + ".png")
    content_ = "分别是：" + str(colors) + "\n"
    content = pic_name + "的主成分共" + str(len(color_dict.keys()) - 1) + "种\n" + content_ + "数量低于" + str(
        num_threshold) + "个像素的非主成分共占比" + str(count / (pic.shape[0] * pic.shape[1])) + "\n" + str(
        datetime.datetime.now())

    print(content, end="")
    if save:
        write_image_detail_txt('a', content, save_path, pic_name)
    res2 = color_dict, pic_arr
    return res2


# 将非主成分用物理最近的主成分进行替换
# 数量阈值， 颜色混合差值，图片路径 保存路径 是否保存 return 处理后的图片数组，更新后的字典
def pic_color_recovery(num_threshold, color_distance, pic_obj, pic_path, save_path, save):
    pic_name = get_image_name_without_extension(pic_path)
    if pic_obj is None:
        pic = Image.open(pic_path).convert('RGBA')
        pic = np.asarray(pic)
        pic = torch.tensor(pic)
    else:
        pic = pic_obj
    # 得到成分分析字典 和 处理后的图片数组
    res_ = pic_color_filter(num_threshold, color_distance, pic, pic_path, save_path, save)
    color_dict = res_[0]
    pic_array = res_[1]
    # 主成分列表
    main = sorted(color_dict.keys())[:-1]
    # 遍历图片数组，找出非主成分像素最相邻的颜色差值最小的主成分颜色，并用其填充，但是暂时还要设置为透明以标记非主成分
    for j in range(pic.shape[1]):
        for i in range(pic.shape[0]):
            # 当前的像素
            now_px = pic_array[i][j]
            # 当前像素颜色
            b1, b2, b3 = now_px[:3]
            now_px_color = rgb_2_hex(b1, b2, b3)
            # 当前像素颜色值
            now_px_color_value = int(now_px_color, 16)
            # 如果像素透明说明是非主成分 但是如果本
            if now_px[3] == 0 and now_px_color not in main:
                # 寻找物理上最近的最相似的颜色 直接就取8个周围像素
                # 1 2 3
                # 4 * 5
                # 6 7 8
                # 第一个元素是周围像素个数
                px = [0]
                px_color = [0]
                px_color_value = [0]
                if i - 1 >= 0 and j - 1 >= 0:
                    px.append(pic_array[i - 1][j - 1])
                    px[0] += 1
                if i - 1 >= 0 and j - 1 >= 0:
                    px.append(pic_array[i - 1][j])
                    px[0] += 1
                if i - 1 >= 0 and j + 1 < len(pic_array[0]):
                    px.append(pic_array[i - 1][j + 1])
                    px[0] += 1
                if j - 1 >= 0:
                    px.append(pic_array[i][j - 1])
                    px[0] += 1
                if j + 1 < len(pic_array[0]):
                    px.append(pic_array[i][j + 1])
                    px[0] += 1
                if i + 1 < len(pic_array) and j - 1 >= 0:
                    px.append(pic_array[i + 1][j - 1])
                    px[0] += 1
                if i + 1 < len(pic_array):
                    px.append(pic_array[i + 1][j])
                    px[0] += 1
                if i + 1 < len(pic_array) and j + 1 < len(pic_array[0]):
                    px.append(pic_array[i + 1][j + 1])
                    px[0] += 1
                # 记录变量
                min1 = int("FFFFFF", 16)
                # 这8个像素的颜色和颜色值
                for it1 in range(len(px)):
                    # 除了第一个用来计总数
                    if it1 != 0:
                        # 拿到rgba
                        a1, a2, a3, a4 = px[it1]
                        # 颜色
                        px_color.append(rgb_2_hex(a1, a2, a3))
                        # 颜色值
                        px_color_value.append(int(px_color[it1], 16))
                        # 如果是主成分，就是透明的或者在主成分列表里，并且他还差值小
                        if (px[it1][3] == 255 or (px_color in main)) and abs(
                                now_px_color_value - px_color_value[it1]) <= min1:
                            # 记录最小差值
                            min1 = abs(now_px_color_value - px_color_value[it1])
                            # 记录最小差值颜色号到px_color第一个元素 最小差值颜色rgb = px[px_color[0]]
                            px_color[0] = it1
                            # 记录周围主成分数量到px_color_value
                            px_color_value[0] += 1
                # 最后看看周围有几个主成分
                if px_color_value[0] != 0:
                    # 如果有就取差值最小的上色
                    pic_array[i][j][:3] = px[px_color[0]][:3]
                    c1, c2, c3 = pic_array[i][j][:3]
                    main_color = rgb_2_hex(c1, c2, c3)
                else:
                    # 如果没有就先取这一行最近的主成分(rgb差值可能不正常表示视觉效果差异）
                    # 找最近的主成分
                    # 指针
                    k = 0
                    m = 0
                    # 这行后面
                    while j + k < pic.shape[1]:
                        q1, q2, q3 = pic_array[i][j + k][:3]
                        color = rgb_2_hex(q1, q2, q3)
                        if color in main:
                            # 就他了
                            pic_array[i][j][:3] = pic_array[i][j + k][:3]
                            break
                        else:
                            k += 1
                    # 选择的主成分颜色
                    c1, c2, c3 = pic_array[i][j][:3]
                    main_color_r = rgb_2_hex(c1, c2, c3)
                    while j - m > 0:
                        q1, q2, q3 = pic_array[i][j - m][:3]
                        color = rgb_2_hex(q1, q2, q3)
                        if color in main:
                            # 就他了
                            pic_array[i][j][:3] = pic_array[i][j - m][:3]
                            break
                        else:
                            m += 1
                        # 选择的主成分颜色
                    c1, c2, c3 = pic_array[i][j][:3]
                    main_color_l = rgb_2_hex(c1, c2, c3)
                    # 选择的主成分颜色
                    if main_color_l in main and main_color_r in main:
                        if abs(int(main_color_l, 16) - int(now_px_color, 16)) <= abs(
                                int(main_color_r, 16) - int(now_px_color, 16)):
                            main_color = main_color_l
                        else:
                            main_color = main_color_r
                    elif main_color_l in main and main_color_r not in main:
                        main_color = main_color_l
                    elif main_color_r in main and main_color_l not in main:
                        main_color = main_color_r
                # 更新字典
                color_dict[main_color][0] += 1
                color_dict[main_color].append([i, j])
    # 删除非主成分类
    del color_dict["none"]
    if save:
        Image.fromarray(pic_array.cpu().numpy()).convert("RGB").save(
            save_path + pic_name + "_recover_filter_" + str(num_threshold) + "_mix_" + str(color_distance) + ".png")

    return pic_array, color_dict


# 将得到的图片数组进行颜色分层，对每张图片进行去噪
# 图片数组， 相应的分布字典， 图片名， 保存文件夹名 返回：图片数组
def pic_simple_color_filter(pic_array, pic_dict, pic_path, save_path, save):
    # 创建一组空白透明原图
    one_color_pics = []
    pic_array_null = pic_array
    for x in range(pic_array_null.shape[0]):
        for y in range((pic_array_null.shape[1])):
            pic_array_null[x][y] = torch.tensor([255, 255, 255, 0])
    # 主成分列表
    main_list = sorted(pic_dict.keys())
    images = []
    # 遍历主成分
    for i in range(len(main_list)):
        # 建立空白图层
        one_color_pics.append([])
        one_color_pics[i] = pic_array_null.clone()
        # 当前主成分颜色
        now_main_color = main_list[i]
        # 当前主成分颜色rgb
        r, g, b = hex_2_rgb(now_main_color)
        # 当前主成分坐标list
        main_color_coordinate = pic_dict[now_main_color][1:]
        # 遍历主成分坐标
        for item in main_color_coordinate:

            # 染色
            one_color_pics[i][item[0]][item[1]] = torch.tensor([r, g, b, 255])

        now_pic = Image.fromarray(one_color_pics[i].cpu().numpy())
        pic_name = get_image_name_without_extension(pic_path)
        new_pic_path = save_path + pic_name + "_" + str(i) + ".png"
        if save:
            now_pic.save(new_pic_path)
        now_pic = cv2.cvtColor(np.asarray(now_pic), cv2.COLOR_RGBA2BGRA)
        images.append([now_pic, new_pic_path])

    return images


def cq_by_kmeans(img, k, pic_path, save_path, save):
    if img is None:
        img = cv2.imread(pic_path)
    if img.shape[2]==4:
        img = img[:,:,:3]
        data = np.float32(img).reshape((-1, 3))

    # 标准化数据
    scaler = StandardScaler()
    img_std = scaler.fit_transform(data)

    # 执行PCA降维
    pca = PCA(n_components=2)
    img_pca = pca.fit_transform(img_std)

    # 在降维后的数据上执行K-means++初始化
    kmeans_pca = KMeans(n_clusters=k, init='k-means++')
    kmeans_pca.fit(img_pca)

    # 获取初始化的聚类中心
    init_centers_pca = kmeans_pca.cluster_centers_

    # 将初始化的聚类中心映射回高维空间
    init_centers_high_dim = pca.inverse_transform(init_centers_pca)

    # 使用PCA初始化的聚类中心进行K-means聚类
    kmeans = KMeans(n_clusters=k, init=init_centers_high_dim, n_init=1)
    kmeans.fit(img_std)

    # 聚类结果
    labels = kmeans.labels_
    palette = kmeans.cluster_centers_

    # 将数据转换回8位元整数
    palette = np.uint8(scaler.inverse_transform(palette))

    # 根据聚类标签重新构造图像
    segmented_image = palette[labels]

    # 将结果转换回原始图像尺寸
    segmented_image = segmented_image.reshape(img.shape)
    segmented_image_ = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    if save:
        cv2.imwrite(f"{save_path}/{k}_k-means.png", segmented_image_)

    return segmented_image

if __name__ == '__main__':
    e = time.time()
    pic_name = 'f-5'
    img_path = './data/test/'+pic_name+'.png'
    save_path = './data/test_res/'+pic_name+'/'
    check_and_create_dir(save_path)

    save = True

    pic = Image.open(img_path).convert('RGBA')
    pic = np.asarray(pic)
    pic = torch.tensor(pic)


    pic[pic[:,:,3] == 0] = torch.tensor([255,255,255,255],dtype=torch.uint8)

    Image.fromarray(pic.cpu().numpy()).convert("RGB").save(
        save_path + pic_name+'.png')
    # res_ = pic_color_mix(10000, None, img_path, save_path, save)
    # 把图片量化
    res_cq = pic_color_recovery(500, 100, pic, img_path, save_path, save)
    img_cq = res_cq[0]
    img_one_color = pic_simple_color_filter(img_cq, res_cq[1], img_path, save_path, save)
    s = time.time()
    t = str(s-e)
    print(t)