from operator import itemgetter

import math
import numpy as np
import torch
import cv2
import diffbl



def curve_loss(points, target_points,w,h):

    # points: (N, 2) tensor, N个点，每个点包含x, y坐标
    # target_points: (M, 2) tensor, M个目标点，每个点包含x, y坐标

    # 初始化损失
    loss = 0

    # 遍历每个点
    for i in range(0, points.size(0), 3):  # 从0开始，每次增加3
        # 计算点points[i]到所有target_points的距离
        distances = ((points[i] - target_points) ** 2).sum(1)

        # 找到最小距离，并累加到损失中
        min_distance = distances.min()
        loss += min_distance

    # 计算平均最小距离
    average_min_distance = loss / points.size(0)

    return torch.relu(average_min_distance/(w + h))

def edge_loss(points, target_length):
    # 输入的两个图像应该是黑白描边图像
    target_length = torch.tensor(target_length)
    total_length = 0.

    num = len(points)//3
    for i in range(num):
        if i != num -1:
            p = points[i*3:i*3+4]
        else:

            p = torch.stack([points[i*3], points[i*3 + 1], points[i*3 + 2], points[0]])
        length = diffbl.compute_bezier_curve_length(p,num_samples=1000)
        total_length = total_length + length
    loss = torch.abs(target_length - total_length) / target_length



    return loss

def edges_loss(points_list, target_lengths):
    total_loss = 0.
    for i in range(len(points_list)):
        part_loss = edge_loss(points_list[i], target_lengths[i])
        total_loss = total_loss + part_loss
    total_loss = total_loss / len(points_list)

    return total_loss




if __name__ == '__main__':
    control_points = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0], [3.0, 1.0], [3.0, 2.0], [5.0, 1.0], [4.0, 0.0]])
    init_control_points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [1.0, 2.0], [2.0, 1.0], [4.0, 0.0]], requires_grad=True)
    # t_l,_ = diffbl.compute_bezier_curve_length(control_points,100)

    # print(loss)
    # 优化器
    optimizer = torch.optim.Adam([init_control_points], lr=1.0)

    for iteration in range(500):


        # 清零梯度
        optimizer.zero_grad()

        # 计算损失
        loss = edge_loss(init_control_points, torch.tensor([11.0]))

        # 反向传播计算梯度
        loss.backward()

        # 在这里访问 point_var.grad 之前，不要执行任何会清空梯度的操作
        print("Gradients with respect to control points:", init_control_points.grad)

        # 更新参数
        optimizer.step()
    print(init_control_points)