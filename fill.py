# start from 24.03.26 
import copy
import filecmp
import os
import random
import shutil
import time

import PIL
import cv2
import math
import numpy as np
import torch
from PIL import Image

# 将透明转为黑色，不透明颜色转为白色
# 返回二值图像和原图层颜色hex
from color_seg import rgb_2_hex, hex_2_rgb, pic_color_recovery, pic_simple_color_filter
from utils import get_image_name_without_extension, write_image_detail_txt,check_and_create_dir,find_opaque_color
import multiprocessing
import matplotlib.pyplot as plt

def pic2opaque(img):
    # 读取并处理图像，
    img_bi = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # 图层颜色
    hex_color = '!'
    # 将图像转化为二值图像，透明色变为黑色，不透明色变为白色
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # 当前像素
            now_p = img[y][x]

            # 如果像素透明
            if now_p[3] == 0:
                # 把像素设为黑色
                img_bi[y][x] = 0
            else:
                img_bi[y][x] = 255
                # 如果还没找到图层颜色
                if hex_color == '!':
                    hex_color = rgb_2_hex(now_p[0], now_p[1], now_p[2])
    # 将RGBA图像转换为灰度图像
    gray_image = cv2.cvtColor(img_bi, cv2.COLOR_RGB2GRAY)

    # 阈值分割，将灰度图像转换为二值图像
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image, hex_color


# 用形态学方法进行图像孔洞填补
# 输入图像路径，输出填补后的图像
def fill_blank_with_shape(binary_image):
    # 对原图取反，‘255’可以根据图像中类别值进行修改。（例如，图像中二值为0和1，那么255则修改为1）
    # 此时mask用来约束膨胀结果。原图白色为边界，黑色为孔洞和背景，取反后黑色为边界，白色为孔洞和背景。
    mask = 255 - binary_image

    # 以带有白色边框的黑色图像为初始Marker，用来SE来连续膨胀，该图通过迭代生成填充图像。
    marker = np.zeros_like(binary_image)

    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255

    # 形态学重建
    SE = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3))
    count = 0
    while True:
        count += 1
        marker_pre = marker
        # 膨胀marker
        dilation = cv2.dilate(marker, kernel=SE)
        # 和mask进行比对，用来约束膨胀。由于mask中黑色为边界，白色为孔洞和背景。
        # 当遇到黑色边界后，就无法继续继续前进膨胀。当遇到白色后，会继续向里面膨胀。孔洞闭合的，遇到全部的黑色边界后，内部自然就已经被填充。
        marker = np.min((dilation, mask), axis=0)
        # 判断经过膨胀后的结果是否和上次迭代一致，如果一致则完成孔洞填充。
        if (marker_pre == marker).all():
            break

    # 将结果取反，还原为原来的图像情况。即白色为边界，黑色为孔洞和背景，
    dst = 255 - marker
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    return dst


# 用轮廓填充进行图像孔洞填补
# 输入图像路径，输出填补后的图像
def fill_blank_with_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out_img = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    for i in range(len(contours)):
        cnt = contours[i]
        # 通过轮廓填充。
        cv2.fillPoly(out_img, [cnt], color=255)
    # cv2.imshow("cnt", out_img)
    # cv2.waitKey(0)

    return out_img


# 图形填色
def binary_img_to_color(img, res_layers, r, g, b):
    # 遍历二值图像将白色部分填为原色
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # 当前的二值图像像素
            now_bi_p = res_layers[y][x]
            # 如果是白色
            if now_bi_p == 255:
                # 原图改为原图层颜色，不透明
                img[y][x] = r, g, b, 255
            else:
                if rgb_2_hex(r, g, b) == 'ffffff':
                    # 如果本身是白色就背景为黑色
                    img[y][x] = 0, 0, 0, 0
                else:
                    # 否则将像素设置为白色，透明
                    img[y][x] = 255, 255, 255, 0
    return img


# 用于调试的可视化函数
def show_image(title, img):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def show_corners_polyfit(image,px_dis,epsilon_value = 0.001):
    if image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 寻找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 选择合适的epsilon值，它决定了轮廓近似的程度
    epsilon = epsilon_value * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    # 移除相邻距离过近的顶点
    filtered_approx = remove_close_points(approx, min_dist=px_dis)
    length = cv2.arcLength(approx, True)
    return filtered_approx,length

# 计算每个连通域的角点
def count_corners(image, max_corners=100, quality_level=0.01, min_distance=5):
    if image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_ = cv2.goodFeaturesToTrack(image, max_corners, quality_level, min_distance)
    # 如果出现角点检测数量为0
    # 1.图片铺满屏幕
    if corners_ is None:
        corners_num = -1
    else:
        corners = [corner[0] for corner in corners_]
        corners_num = len(corners)
        # 要四个才能成型
        if corners_num < 4:
            corners_num = 4

    # 使用列表推导式来简化循环

    return corners_num


def remove_close_points(points, min_dist):
    """ 移除相邻距离过近的点 """
    if len(points) < 2:
        return points
    filtered_points = [points[0]]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i][0] - points[i - 1][0]) >= min_dist:
            filtered_points.append(points[i])
    return filtered_points


# 将单色图层的各个连通域分割
# 输入图片，在图片最外层加上

# 图像填充 返回结果各个图层及其各连通域、质心、角点
def pic_fill(pic_obj, pic_path, save_path, save, fill):
    if pic_obj is None:
        # 读取原图层
        img = cv2.imread(pic_path, cv2.IMREAD_UNCHANGED)
    else:
        img = pic_obj

    # 得到原图二值图像
    b_img, color_org = pic2opaque(img)
    r, g, b = hex_2_rgb(color_org)
    pic_name = get_image_name_without_extension(pic_path)
    # 对二值图像分割连通域，对每个联通域分别处理
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b_img, connectivity=8, ltype=cv2.CV_32S)

    # 根据标签数建立联通域图层数组
    layers = np.zeros((num_labels, b_img.shape[0], b_img.shape[1]), np.uint8)
    # 结果图层
    res_layers = []
    # 图层各个连通域
    res_connections_belong = []

    # 建立存储每个连通域的文件夹
    folder_path = save_path + "con_" + pic_name
    if save:
        # 检查文件夹是否存在
        if os.path.exists(folder_path):
            # 清空文件夹
            shutil.rmtree(folder_path)
            # 重新创建空文件夹
            os.makedirs(folder_path)
            print(f"\n文件夹 '{folder_path}' 已清空并重新创建。\n")
        else:
            os.mkdir(folder_path)
    # 遍历所有的连通域，将每个联通域放一个数组，每个数组白色为原色
    simple_connection = []
    for i in range(1, num_labels):
        mask = labels == i
        layers[i][mask] = 255
        if fill:
            # 填充处理后的二值图像
            binary_pic = fill_blank_with_contours(layers[i])

        # 可视化当前连通域
        # show_image(f'Binary Pic {i}', binary_pic)

        # 放数组
        simple_connection.append(binary_pic)

        if i == 1:
            res_layers.append(binary_pic.copy())  # 使用.copy()来创建binary_pic的副本
            res_connections_belong.append([0])
        else:
            placed = False
            for k in range(len(res_layers)):
                # 判断与当前层是否重叠
                overlap_flag = np.any(np.logical_and(res_layers[k] == 255, binary_pic == 255))

                if overlap_flag:
                    # 可视化重叠检测
                    # show_image(f'Overlap Detected Layer {k} with Binary Pic {i}',
                    #            np.logical_and(res_layers[k] == 255, binary_pic == 255))
                    # 如果最后一层了还是重叠 就新建
                    if k == len(res_layers) - 1:
                        res_layers.append(binary_pic.copy())  # 使用.copy()来创建binary_pic的副本
                        res_connections_belong.append([i - 1])
                    # 否则继续往后
                    else:
                        k += 1
                # 如果没重叠
                else:
                    # 相加
                    res_layers[k][binary_pic == 255] = 255
                    res_connections_belong[k].append(i - 1)
                    placed = True
                    break

            if not placed:
                res_layers.append(binary_pic.copy())  # 使用.copy()来创建binary_pic的副本
                res_connections_belong.append([i - 1])

    # 最后可视化所有层
    # for idx, layer in enumerate(res_layers):
    #     show_image(f'Result Layer {idx}', layer)

    res = []
    res_simple_connections = []
    res_simple_connections_centers = []
    res_simple_connections_corners_num = []

    # 遍历这个图层的连通域序号
    for j in range(len(res_layers)):

        folder_path_next = folder_path + '/' + str(j)
        if save:
            # 检查文件夹是否存在
            if os.path.exists(folder_path_next):
                # 清空文件夹
                shutil.rmtree(folder_path_next)
                # 重新创建空文件夹
                os.makedirs(folder_path_next)
                print(f"\n文件夹 '{folder_path_next}' 已清空并重新创建。\n")
            else:
                os.mkdir(folder_path_next)

        # 把属于他的放进去
        for index in res_connections_belong[j]:

            # 角点数量
            corner_num = count_corners(simple_connection[index])
            res_simple_connections_corners_num.append(corner_num)

            # 连通域颜色还原
            simple_connection[index] = binary_img_to_color(img.copy(), simple_connection[index], b, g, r)
            # 连通域
            res_simple_connections.append(simple_connection[index])
            # 质心
            res_simple_connections_centers.append(centroids[index + 1])

            if save:
                # 存储图片
                pil_image_b_array = cv2.cvtColor(simple_connection[index], cv2.COLOR_RGB2RGBA)
                pil_image_b = Image.fromarray(pil_image_b_array)
                # name = folder_path_next + "/" + pic_name + "_" + str(index) + "fill_connection.png"
                name = folder_path_next + "/connection_" + str(index) + ".png"

                pil_image_b.save(name)
                # 存储信息
                content = ("\n" + name + "的中心是：" + "\n" + str(centroids[index + 1]) + "\n 角点共" + str(
                    corner_num) + "个 \n")
                write_image_detail_txt('a', content, folder_path_next + "/", pic_name)
        bi_img = res_layers[j]
        # 颜色还原
        img = binary_img_to_color(img, res_layers[j], b, g, r)
        res.append(
            [res_simple_connections, bi_img, img, res_simple_connections_centers, res_simple_connections_corners_num])
        if save:
            # OpenCV默认为BGR模式，转换为RGB模式
            rgb_image = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(folder_path_next + "/layer_" + str(j) + ".png")

    return res


# 对二值图像进行排序，使图像不被遮挡
def count_layers_order(imgs):
    # 返回的结果数组


    # 存储面积和对应索引的列表
    areas_with_indices = []

    # 计算每张图像的白色区域面积并收集索引
    for index, binary_image in enumerate(imgs):
        area = cv2.countNonZero(binary_image)
        areas_with_indices.append((area, index))

    # 根据面积对图像进行排序（升序）
    areas_with_indices.sort(key=lambda x: x[0])

    # 提取排序后的索引
    sorted_indices = [index for _, index in areas_with_indices]

    return sorted_indices


def show_connection_mask(simple_connection):
    # 将二维数组转换为 8 位单通道图像
    mask = simple_connection.astype('uint8')
    # 应用阈值以确保掩码中的值为 0 或 255
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # 显示掩码图像
    cv2.imshow('Connection Mask', mask)
    cv2.waitKey(0)  # 等待用户按键

def generate_distinct_colors(total, min_diff=50):
    def color_diff(c1, c2):
        return sum(abs(x - y) for x, y in zip(c1, c2))

    unique_colors = set()
    while len(unique_colors) < total:
        new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if all(color_diff(new_color, existing_color) >= min_diff for existing_color in unique_colors):
            unique_colors.add(new_color)

    return list(unique_colors)
# 对每个图层进行合并并进行该色
def layer_set_color(cq_res, con_color_dis, save_path, save, con_size_dis, fill,px_dist):
    # cq_res 里面[now_pic, new_pic_path]
    # 染色的颜色差值 需要所有联通域的数量不大于 255 * 255 * 255 /dis 默认最大255 * 255 * 255 /50个
    color = 0
    # 计算随机颜色列表
    if con_color_dis is None:
        print(f"\n开始计算随机颜色列表")
        # 计算总连通域数
        total = 0
        for i in range(len(cq_res)):
            if cq_res[i][0] is None:
                now_pic = cv2.imread(cq_res[i][1], cv2.IMREAD_UNCHANGED)
            else:
                # 量化后的图层
                now_pic = copy.deepcopy(cq_res[i][0])

            now_color = find_opaque_color(copy.deepcopy(now_pic))
            now_color = rgb_2_hex(now_color[0],now_color[1],now_color[2])

            if now_color == "000000":

                now_pic[now_pic[:,:,3] == 255] = [255,255,255,255]
                now_pic[now_pic[:,:,3] == 0] = [0,0,0,0]

            # 将RGBA图像转换为灰度图，因为连通域检测需要单通道图像
            gray_image = cv2.cvtColor(now_pic, cv2.COLOR_RGBA2GRAY)
            # 黑色前景白色背景
            # 将灰度图转换为float32以便进行计算
            gray_image = gray_image.astype(np.float32)

            # 使用Alpha通道的反掩模将背景设置为黑色
            gray_image = (gray_image * (now_pic[:, :, 3] / 255.0)).astype(np.uint8)

            # 查找每个连通域，并进行改色
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_image, connectivity=8,
                                                                      ltype=cv2.CV_32S)

            total += num_labels - 1
            print(f'\n连通域查找中 已有{total}个 第{i}层 {num_labels-1}个 共{len(cq_res)-1}层')
        # 随机颜色
        if total>200:
            print("颜色数量过多 递归生成过慢 自动设为固定分布")
            con_color_dis = 50
        else:
            print(f'颜色计算中...')
            colors = generate_distinct_colors(total)
            print(f"\n颜色列表计算完成")
    # 创建一个与原图像大小相同的空白图像，初始化为黑色
    res = np.zeros((cq_res[0][0].shape[0], cq_res[0][0].shape[1], 3), dtype=np.uint8)
    print(f"总连通域数量：{total}")
    # 图结果
    connect_result = []
    binary_res = []
    # 连通域序号
    j = 0

    print(f'\n开始改色')
    # 遍历每个图层
    for ii in range(len(cq_res)):
        print(f'\n第{ii}个图层')
        if cq_res[ii][0] is None:
            now_pic_ = cv2.imread(cq_res[i][1], cv2.IMREAD_UNCHANGED)
        else:
            # 量化后的图层
            now_pic_ = copy.deepcopy(cq_res[ii][0])
        # 真实的颜色
        real_c = find_opaque_color(copy.deepcopy(now_pic_))
        now_color_ = rgb_2_hex(real_c[0],real_c[1],real_c[2])
        # 其所在位置
        now_pic_path = cq_res[ii][1]
        # 图层图片名称
        now_pic_name = get_image_name_without_extension(now_pic_path)
        if save:
            # 创建每层的目录
            layer_content = save_path + 'colorful_' + str(ii) + '/'
            check_and_create_dir(layer_content)
        # 透明区域变成黑色背景 不透明变白色 图像变为二值图像
        if now_color_ == "000000":
            now_pic_[now_pic_[:, :, 3] == 255] = [255, 255, 255, 255]
            now_pic_[now_pic_[:, :, 3] == 0] = [0, 0, 0, 0]
            if save:
                cv2.imwrite(layer_content + 'black_' + str(j) + '.png', now_pic_)

        # 将RGBA图像转换为灰度图，因为连通域检测需要单通道图像
        gray_image = cv2.cvtColor(now_pic_, cv2.COLOR_RGBA2GRAY)

        # 黑色前景白色背景
        # 将灰度图转换为float32以便进行计算
        gray_image = gray_image.astype(np.float32)
        # 使用Alpha通道的反掩模将背景设置为黑色
        gray_image = (gray_image * (now_pic_[:, :, 3] / 255.0)).astype(np.uint8)

        # 查找每个连通域，并进行改色
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_image, connectivity=8,
                                                                                ltype=cv2.CV_32S)


        # 为每个连通域分配一个颜色
        num_labels = num_labels - 1  # 0是背景标签
        # 创建一个与原图像大小相同的空白图像，初始化为黑色
        color_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

        # 遍历每个连通域并染色
        for label in range(1, num_labels + 1):  # 从1开始，因为0是背景
            _, _, w, h = stats[label][:4]
            area = w * h
            if con_size_dis is not None:
                if area > con_size_dis:
                    flag = True
                else:
                    print(f"\n{j}-面积{area}小于等于阈值{con_size_dis}")
                    flag = False
            else:
                flag = True
            # 创建颜色掩模
            mask = labels == label
            if con_color_dis is not None:
                # 颜色差值
                color += con_color_dis
                # 应用颜色
                rgb_color = hex_2_rgb(f'{color:06x}')
            else:
                rgb_color = colors[j]
            color_image[mask] = rgb_color
            if flag:
                print(f'\n第{label}个连通域')
                # 创建一个与原图像大小相同的空白图像，初始化为黑色背景白色前景
                conn = np.zeros((cq_res[0][0].shape[0], cq_res[0][0].shape[1]), dtype=np.uint8)
                conn[mask] = 255
                # 创建一个与原图像大小相同的空白图像
                conn_colorful = np.zeros((cq_res[0][0].shape[0], cq_res[0][0].shape[1],3), dtype=np.uint8)
                conn_colorful[mask] = rgb_color
                # 填充处理后的二值图像 黑底白字
                if fill:
                    binary_pic = fill_blank_with_contours(conn)
                else:
                    binary_pic = conn
                # 角点数识别
                corners,length = show_corners_polyfit(binary_pic,px_dist)

                # 可视化当前连通域
                # show_image(f'Binary Pic {i}', binary_pic)
                # 存一下每个连通域的染色颜色,真实颜色，序号，角数，目标长度
                if save:
                    if fill:
                        cv2.imwrite(layer_content+'fill_'+str(j)+'.png', binary_pic)
                    cv2.imwrite(layer_content + 'binary_' + str(j) + '.png', conn)
                    cv2.imwrite(layer_content + 'colorful_' + str(j) + '.png', conn_colorful)
                connect_result.append([rgb_color,real_c, j, corners, length])
                binary_res.append(binary_pic)
                print(f"\n{j}连通域改色完成")
                # 序号加1
                j += 1
        print(f'\n{ii}图层改色完成')
        # cv2.imshow('layer', color_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if save:
            cv2.imwrite(layer_content + str(ii)+'.png', color_image)
        # 这层的非透明区域设置为mask，结果图片mask至为0
        # 使用布尔掩模将 res 中 Alpha 通道非零的位置设置为 0
        alpha_mask = now_pic_[:, :, 3] > 0  # Alpha 通道大于 0 的位置
        res[alpha_mask] = color_image[alpha_mask]
        # 叠加
        # cv2.imshow('res', res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print(f'\n改色完成')
    if save:
        cv2.imwrite(save_path + 'colorful_target.png', res)

    return res,connect_result,binary_res


# 输入一系列乱序的圆，输出其正确顺序
def count_circle_order(circles):
    # 使用enumerate来保留原始索引
    indexed_circles = list(enumerate(circles))
    # 根据半径排序，保持相同半径的圆的原始顺序
    indexed_circles.sort(key=lambda x: x[1][2], reverse=True)
    order = []
    for i in indexed_circles:
        order.append(i[0])
    return order


def circles_intersect(circle1,circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    # 计算圆心之间的距离
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # 检查两个圆是否相交
    return dist < (r1 + r2)

def is_enough_circle_bag(circle,bag):
    # 先看看是否与bag的圆相交
    for index,c_bag in enumerate(bag):
        # 如果两个圆相交就不够
        if circles_intersect(c_bag,circle):
            return False
    return True

def count_layer_circle_order(circles):
    bags = []
    bags_order = []
    bags.append([circles[0]])
    bags_order.append([0])
    now = 0
    for index, circle in enumerate(circles[1:]):
        if is_enough_circle_bag(circle,bags[now]):
            bags[now].append(circle)
            bags_order[now].append(index+1)
        else:
            now += 1
            bags.append([circle])
            bags_order.append([index+1])



    return bags,bags_order


# if __name__ == '__main__':
#     res = []
#     for i in range(2):
#         pic_path = f'./curve_fit4/fill_trouble_{i}.png'
#         img = cv2.imread(pic_path, cv2.IMREAD_UNCHANGED)
#         res.append([img,pic_path])
#     s_time = time.time()
#     res, _ = layer_set_color(res,None,)
#     f_time = time.time()
#     print(str(f_time-s_time))
#     cv2.imwrite('./curve_fit4/res.png',res)