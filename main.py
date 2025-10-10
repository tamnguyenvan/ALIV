# 算损失用叠加
import time

import diffvg
import pydiffvg
import torch
import cv2
import matplotlib.pyplot as plt
import random
import argparse
import math
import errno
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.nn.functional import adaptive_avg_pool2d
import color_seg, fill
import warnings

warnings.filterwarnings("ignore")

import PIL
import PIL.Image
import os
import os.path as osp
import numpy as np
import numpy.random as npr
import shutil
import copy
# import skfmm

from shape_loss import smoothness_loss, edge_loss, curve_loss, edges_loss

import yaml
from easydict import EasyDict as edict

gamma = 1.0

##########
# helper #
##########

from utils import \
    get_experiment_id, \
    edict_2_dict, \
    check_and_create_dir, write_image_detail_txt, find_opaque_color


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--config", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--color_num", type=int, default=0)
    parser.add_argument("--color_distance", type=int, default=0)
    parser.add_argument("--target", type=str, help="target image path")
    parser.add_argument('--log_dir', metavar='DIR', default="log/debug")
    parser.add_argument('--signature', nargs='+', type=str)
    parser.add_argument("--px_dis", type=float, default=0)
    parser.add_argument("--edge_loss_weight", type=float, default=9)
    parser.add_argument("--structure_loss_weight", type=float, default=9)
    parser.add_argument("--loss_weight", type=float, default=9)


    cfg = edict()
    args = parser.parse_args()
    cfg.debug = args.debug
    cfg.config = args.config

    cfg.color_num = args.color_num
    cfg.color_distance = args.color_distance
    cfg.px_dis = args.px_dis
    cfg.edge_loss_weight = args.edge_loss_weight
    cfg.structure_loss_weight = args.structure_loss_weight
    cfg.loss_weight = args.loss_weight
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.target = args.target
    cfg.log_dir = args.log_dir
    cfg.signature = args.signature
    return cfg


def get_bezier_ploy(corners):
    points = []
    for c in corners:
        points.append(c[0])
        points.append(c[0])
        points.append(c[0])
    points = torch.tensor(points)
    points = points.type(torch.FloatTensor)
    return points


def init_all_shapes(conns_):

    total_num = 0
    shapes = []
    shape_groups = []
    changed_color = []
    real_color = []
    for index, conn in enumerate(conns_):
        # 染色颜色,真实颜色，序号，角数，目标长度
        # 由于改变颜色的部分是用cv所以要bgr变rgb
        color_b, color_g, color_r = conn[0]
        # 真实颜色
        real_color.append(torch.tensor([conn[1][2] / 255.0, conn[1][1] / 255.0, conn[1][0] / 255., 1.0]))

        num_path = index
        cors = conn[3]
        length = conn[4]

        # 曲线数量
        num_segments = len(cors)
        # 一段曲线是2个控制点
        num_control_points = [2] * num_segments
        num_control_points = torch.LongTensor(num_control_points)
        total_num += num_segments * 3
        # 用多边形拟合
        points = get_bezier_ploy(cors)
        # 构造路径，这里先不考虑描边
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=torch.tensor(0.0),
                             is_closed=True)
        # 将路径加入到形状
        shapes.append(path)
        fill_color_init = torch.tensor([color_r / 255.0, color_g / 255.0, color_b / 255., 1])
        changed_color.append(fill_color_init)
        # 编组，不描边
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([num_path]),
            fill_color=fill_color_init,
            stroke_color=torch.tensor([0, 0, 0, 0]),
        )
        # 加入形状
        shape_groups.append(path_group)
        # 所有点的坐标
        if cfg.save.output:
            filename = os.path.join(
                cfg.experiment_dir, "init-svg/", f"init{num_path}.svg")
            check_and_create_dir(filename)
            pydiffvg.save_svg(filename, w, h, shapes, shape_groups)
    point_var = []
    # 赋予点集梯度
    for index, path in enumerate(shapes):
        # 方形背景和小方块都不学
        if length < 2 or (length == (w + h) * 2 and num_segments == 4):
            print("存在背景或者极小方块将不会学习")
        else:
            path.points.requires_grad = True
            point_var.append(path.points)
    for group in shape_groups:
        group.fill_color.requires_grad = False
    # 返回形状，编组，点集
    print(f"总参数{total_num}\n")

    return shapes, shape_groups, point_var, real_color, changed_color


class linear_decay_lrlambda_f(object):
    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n // self.decay_every
        decay_step = n % self.decay_every
        lr_s = self.decay_ratio ** decay_time
        lr_e = self.decay_ratio ** (decay_time + 1)
        r = decay_step / self.decay_every
        lr = lr_s * (1 - r) + lr_e * r
        return lr


if __name__ == "__main__":
    #############
    #    配置    #
    #############

    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_device(torch.device('cuda:1'))
    device = pydiffvg.get_device()

    # 读取配置
    cfg_arg = parse_args()
    with open(cfg_arg.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_default = edict(cfg['default'])
    cfg = edict(cfg[cfg_arg.experiment])
    cfg.update(cfg_default)
    cfg.update(cfg_arg)
    cfg.exid = get_experiment_id(cfg.debug)
    if cfg.color_num != 0 and cfg.color_distance != 0:
        cfg.cq_para.use_cq = True
        cfg.cq_para.color_num = cfg.color_num
        cfg.cq_para.color_distance = cfg.color_distance
    if cfg.structure_loss_weight != 9:
        cfg.loss.structure_loss_weight = cfg.structure_loss_weight
    if cfg.edge_loss_weight != 9:
        cfg.loss.edge_loss_weight = cfg.edge_loss_weight
    if cfg.loss.loss_weight != 9:
        cfg.loss.loss_weight = cfg.loss_weight
    if cfg.px_dis != 0:
        cfg.cq_para.px_dis = cfg.px_dis

    # 创建文件夹
    cfg.experiment_dir = \
        osp.join(cfg.log_dir, '{}_{}'.format(cfg.exid, '_'.join(cfg.signature)))
    configfile = osp.join(cfg.experiment_dir, 'config.yaml')
    check_and_create_dir(configfile)
    with open(osp.join(configfile), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)

    # ！！读入图像
    gt = np.array(PIL.Image.open(cfg.target))



    # 原图备份
    gt_color = np.array(PIL.Image.open(cfg.target).convert('RGBA'))

    # 打印图像信息与提示
    print(f"输入图像的尺寸是 {gt.shape}")
    if len(gt.shape) == 2:
        print("将二值图像转为rgb")
        gt = np.array(PIL.Image.open(cfg.target).convert('RGBA'))


        # 使用Alpha通道的反掩模将背景
        gt[gt[:, :, 3] == 0] = [255, 255, 255, 255]
        gt = gt[:, :, :3]

    elif gt.shape[2] == 4:
        print("将透明通道脱去")
        gt = gt[:, :, :3]

    # 原图宽高
    h, w = gt.shape[:2]
    gt = (gt / 255).astype(np.float32)
    gt = torch.FloatTensor(gt).permute(2, 0, 1)[None].to(device)

    # !!随机数配置
    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    render = pydiffvg.RenderFunction.apply

    ###########
    #  预处理  #
    ###########

    # 1.将原图按照算法进行CQ得到量化后的单色图层
    # CQ 参数
    cq_s_time = time.time()
    save = cfg.save.cq
    cq_para = cfg.cq_para
    cq_res = cfg.experiment_dir + "/cq/"
    # 函数写的要输入图片
    cq_pic = gt_color
    if save:
        check_and_create_dir(cq_res)
    # 如果需要量化
    if cq_para.use_cq:

        # 把图片量化
        res_cq = color_seg.pic_color_recovery(cq_para.color_num, cq_para.color_distance, cq_pic, cfg.target, cq_res,
                                              save)
        img_cq = res_cq[0]
        color_dict = res_cq[1]
        img_one_color = color_seg.pic_simple_color_filter(img_cq, color_dict, cfg.target, cq_res, save)
    # 使用kmeans量化
    elif cq_para.use_cq_kmeans is not None and cq_para.use_cq_kmeans > 0:
        res_cq = color_seg.cq_by_kmeans(cq_pic, cq_para.use_cq_kmeans, cfg.target, cq_res, save)
        color_dict = color_seg.pic_color_count(res_cq, cfg.target, cq_res, save)
        img_cq = res_cq

        img_one_color = color_seg.pic_simple_color_filter(torch.tensor(gt_color), color_dict, cfg.target, cq_res, save)
    else:
        color_dict = color_seg.pic_color_count(gt_color, cfg.target, cq_res, save)
        # 如果不需要量化，还是要返回一样的
        img_one_color = color_seg.pic_simple_color_filter(torch.tensor(gt_color), color_dict, cfg.target, cq_res, save)
        # 不进行量化的图片会带有锯齿，需要设置抗锯齿
    cq_e_time = time.time()
    cq_time = str(cq_e_time - cq_s_time)

    print(f"图片量化完成 耗时{cq_time}\n开始进行目标量化图片的改色...\n")

    # 2.将图层二值化识别连通域，并赋予不同的颜色，同时记录原来的颜色，估计的角点, 轮廓长度

    fill_s_time = time.time()

    res, conns, order_binary = fill.layer_set_color(img_one_color, cfg.cq_para.fill_color_dis, cq_res, save,
                                                    cfg.cq_para.fill_size_dis, cfg.cq_para.use_fill, cfg.cq_para.px_dis)
    fill_e_time = time.time()
    fill_time = str(fill_e_time - fill_s_time)
    if cfg.cq_para.use_fill:
        print(f"图片改色填充完成 耗时{fill_time} 开始进行排序\n")
    else:
        print(f"图片未进行填充 耗时{fill_time}\n")


    # 3.计算所有联通域的外接圆层级顺序
    order_s_time = time.time()
    ordered_conns = []
    target_lengths = []

    order = fill.count_layers_order(order_binary)
    order = reversed(order)
    for o in order:
        ordered_conns.append(conns[o])
        target_lengths.append(conns[o][-1])

    order_e_time = time.time()
    order_time = str(order_e_time - order_s_time)

    print(f"图层排序完成 耗时{order_time}")

    #############
    #   矢量化   #
    #############

    ml_s_time = time.time()
    final_shapes = []
    final_shapes_group = []
    print(f"\n矢量初始化开始...\n")
    # 将所有图层按序进行初始化
    shapes, shapes_group, point_var, real_c, changed_c = init_all_shapes(ordered_conns)
    print(f"矢量初始化结束 共{len(shapes)}个路径\n")

    # 开始学习
    if cfg.save.output:
        filename = os.path.join(
            cfg.experiment_dir, "init-svg/", "init.svg")
        check_and_create_dir(filename)
        pydiffvg.save_svg(filename, w, h, shapes, shapes_group)
    pg = [{'params': point_var, 'lr': cfg.lr_base.point}]
    optim = torch.optim.Adam(pg)
    lrlambda_f = linear_decay_lrlambda_f(cfg.num_iter, cfg.lr_base.decay)
    scheduler = LambdaLR(
        optim, lr_lambda=lrlambda_f, last_epoch=-1)
    res = (res / 255).astype(np.float32)
    res = torch.FloatTensor(res).permute(2, 0, 1)[None].to(device)
    t_range = tqdm(range(cfg.num_iter))
    black_bg = torch.tensor([0., 0., 0.], requires_grad=False, device=device)
    white_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)

    loss = 0.
    for t in t_range:
        optim.zero_grad()


        # 将图像颜色还原后的 最后还要改回来
        for index, s_group in enumerate(shapes_group):
            s_group.fill_color = real_c[index]
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shapes_group)
        img_r = render(w, h, 2, 2, t, None, *scene_args)
        if cfg.bg == "white":
            # 用来存储图片和算损失
            img_r = img_r[:, :, 3:4] * img_r[:, :, :3] + \
                    white_bg * (1 - img_r[:, :, 3:4])
        elif cfg.bg == "black":

            # 用来存储图片和算损失
            img_r = img_r[:, :, 3:4] * img_r[:, :, :3] + \
                    black_bg * (1 - img_r[:, :, 3:4])

        img_r = img_r[:,:,:3]
        if cfg.save.video:
            filename = os.path.join(
                cfg.experiment_dir, "video-png/",
                "iter{}.png".format(t))
            check_and_create_dir(filename)

            imshow = img_r.detach().cpu()
            pydiffvg.imwrite(imshow, filename, gamma=gamma)
        x_r = img_r.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        loss = (x_r - gt).pow(2).mean()
        mse = loss
        loss = loss * cfg.loss.loss_weight
        for index, s_group in enumerate(shapes_group):
            s_group.fill_color = changed_c[index]

        if (cfg.loss.structure_loss_weight is not None) \
                and (cfg.loss.structure_loss_weight > 0):
            scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shapes_group)
            img = render(w, h, 2, 2, t, None, *scene_args)

            if cfg.bg == "white":
                # 用来存储图片和算损失
                img = img[:, :, 3:4] * img[:, :, :3] + \
                      white_bg * (1 - img[:, :, 3:4])
            elif cfg.bg == "black":
                # 用来存储图片和算损失
                img = img[:, :, 3:4] * img[:, :, :3] + \
                      black_bg * (1 - img[:, :, 3:4])

            img = img[:, :, :3]
            x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW

            # 图像基础重建损失 改变颜色的部分

            loss_original = (x - res).pow(2).mean()
            loss = loss_original * cfg.loss.structure_loss_weight + loss

        if (cfg.loss.edge_loss_weight is not None) \
                and (cfg.loss.edge_loss_weight > 0):
            # 轮廓周长损失
            loss_edge = edges_loss(point_var, target_lengths)
            loss = loss + loss_edge * cfg.loss.edge_loss_weight

        # if (cfg.loss.xing_loss_weight is not None) \
        #         and (cfg.loss.xing_loss_weight > 0):
        #     x_loss = xing_loss(point_var)
        #     loss = loss + x_loss * cfg.loss.xing_loss_weight

        t_range.set_postfix({'a': loss.item(),'mse': mse.item()})
        loss.backward()
        optim.step()
        scheduler.step()
        for group in shapes_group:
            group.fill_color.data.clamp_(0.0, 1.0)

    ml_e_time = time.time()
    ml_time = str(ml_e_time - ml_s_time)
    if cfg.save.video:
        print("saving iteration video...")
        img_array = []
        for tt in range(0, cfg.num_iter):
            filename = os.path.join(
                cfg.experiment_dir, "video-png/",
                "iter{}.png".format(tt))
            img = cv2.imread(filename)
            # cv2.putText(
            #     img, "Path:{} \nIteration:{}".format(pathn_record_str, ii),
            #     (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            img_array.append(img)

        videoname = os.path.join(
            cfg.experiment_dir, "video-avi/video.avi")
        check_and_create_dir(videoname)
        out = cv2.VideoWriter(
            videoname,
            # cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'FFV1'),
            20.0, (w, h))
        for iii in range(len(img_array)):
            out.write(img_array[iii])
        out.release()
        # shutil.rmtree(os.path.join(cfg.experiment_dir, "video-png"))
    if cfg.save.output:
        for index, s_group in enumerate(shapes_group):
            s_group.fill_color = changed_c[index]
        filename = os.path.join(
            cfg.experiment_dir, "output-svg/", "res_colorful.svg")
        check_and_create_dir(filename)
        pydiffvg.save_svg(filename, w, h, shapes, shapes_group)

    if cfg.save.output:
        for index, s_group in enumerate(shapes_group):
            s_group.fill_color = real_c[index]
        filename = os.path.join(
            cfg.experiment_dir, "output-svg/", "res_original.svg")
        filename_ = os.path.join(
            cfg.experiment_dir, "output-svg/", "res_original.png")
        check_and_create_dir(filename)
        check_and_create_dir(filename_)
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shapes_group)
        img_r = render(w, h, 2, 2, t, None, *scene_args)
        imshow = img_r.detach().cpu()
        pydiffvg.imwrite(imshow, filename_, gamma=gamma)
        pydiffvg.save_svg(filename, w, h, shapes, shapes_group)



    if cfg.save.output:
        for index, s_group in enumerate(shapes_group):
            s_group.fill_color = torch.tensor([1., 1., 1., 1])
            s_group.stroke_color = torch.tensor([0., 0., 0., 1])
        for index, p in enumerate(shapes):
            p.stroke_width = torch.tensor(1.0)
        filename = os.path.join(
            cfg.experiment_dir, "output-svg/", "res_line.svg")
        check_and_create_dir(filename)
        pydiffvg.save_svg(filename, w, h, shapes, shapes_group)
    if cfg.save.loss:
        loss_file = osp.join(cfg.experiment_dir, 'detail.txt')
        points_num = 0
        for index, p in enumerate(shapes):
            points_num += len(p.points)
        check_and_create_dir(loss_file)
        content = f'\nloss:{loss.item()} mse:{mse.item()} cq_time:{cq_time} fill_time:{fill_time} ml_time:{ml_time}\npath_num:{len(shapes)} points_num:{points_num}'
        with open(loss_file, 'w', encoding='utf-8') as file:
            file.write(content)  # 写入内容