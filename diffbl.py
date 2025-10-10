import torch


def bezier_curve(t, control_points):
    """
    三次贝塞尔曲线的参数化表示。

    :param t: 参数 t，形状为 [num_samples, 1]，取值在 [0, 1] 之间。
    :param control_points: 控制点，形状为 [4, 2]，代表四个控制点的 (x, y) 坐标。
    :return: 曲线上的点，形状为 [num_samples, 2]。
    """
    # 计算贝塞尔基函数
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t ** 2
    b3 = t ** 3

    # 计算曲线上的点
    points = b0.unsqueeze(1) * control_points[0] + \
             b1.unsqueeze(1) * control_points[1] + \
             b2.unsqueeze(1) * control_points[2] + \
             b3.unsqueeze(1) * control_points[3]
    return points


def bezier_curve_derivative(t, control_points):
    # 计算导数系数
    d0 = -3 * (1 - t) ** 2
    d1 =  6 * (1 - t) * t
    d2 = -3 * t ** 2

    # 控制点差分
    cp_diff1 = control_points[1] - control_points[0]
    cp_diff2 = control_points[2] - control_points[1]
    cp_diff3 = control_points[3] - control_points[2]

    # 计算导数向量
    derivatives = d0.unsqueeze(1) * cp_diff1 + \
                  d1.unsqueeze(1) * cp_diff2 + \
                  d2.unsqueeze(1) * cp_diff3
    return derivatives
def point_distance(p1, p2):
    """
    计算两个点之间的欧几里得距离。
    """
    return torch.norm(p1 - p2, p=2, dim=2)

def total_length_of_points(points):
    """
    计算一系列点连线的长度。
    """
    distances = point_distance(points[1:], points[:-1])
    total_length = torch.sum(distances)
    return total_length
def compute_bezier_curve_length(control_points, num_samples=100):

    control_points.requires_grad_(True)
    t_samples = torch.linspace(0, 1, num_samples).unsqueeze(1)
    points = bezier_curve(t_samples, control_points.unsqueeze(1))
    l = total_length_of_points(points)


    return l
if __name__ == '__main__':
    # 示例控制点
    control_points = torch.tensor([[281.5488, 136.9664],
        [282.3675, 142.1356],
        [282.6414, 147.3620],[282.9153, 152.5885]], requires_grad=True)

    # 计算曲线长度和梯度
    length = compute_bezier_curve_length(control_points,1000)

    # 打印曲线长度和梯度
    print(f"The length of the cubic Bezier curve is: {length.item()}")

    # print("Gradients with respect to control points:")
    # print(gradients)