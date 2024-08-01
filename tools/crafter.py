# -*- coding: utf-8 -*-

import wandb
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn

#from .visualization import epoch_visual
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from utils.save_visualization_file import save_ply
#import shutil


def mean(lis): return sum(lis) / len(lis)


class Crafter(nn.Module):
    """ Helper class to train/valid a deep network.
        Overload this class `forward_backward` for your actual needs.

    Usage: 
        train/valid = Trainer/Valider(net, loader, loss, optimizer)
        for epoch in range(n_epochs):
            train()/valid()
    """

    def __init__(self, net):
        nn.Module.__init__(self)
        self.net = net

    def iscuda(self):
        # 取出网络中的第一个参数 # type(self.net.parameters()): <class 'generator'>
        # 因为 self.net.parameters() 是一个生成器(生成器是返回迭代器的函数),
        # 所以 next(self.net.parameters()) 就是他返回的迭代器的第一个东西, 就是网路的第一个参数
        return next(self.net.parameters()).device != torch.device('cpu')

    def todevice(self, x):
        if isinstance(x, dict):
            return {k: self.todevice(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [self.todevice(v) for v in x]

        if self.iscuda():
            # 如果网络在 cuda 里面, 则将数据 contiguous 化,
            # non_blocking 是从 cpu 拷贝到 gpu 可能可以异步进行
            if isinstance(x, str):
                return x
            else:
                return x.contiguous().cuda(non_blocking=True)
        else:
            return x.cpu()

    def __call__(self, epoch):
        raise NotImplementedError()


def sample_visual_index(loader):
    # vidx = list(np.random.choice(np.arange(1, len(loader) - 1), 4, replace=False))
    # vidx.append(0)
    # vidx.append(len(loader) - 1)
    vidx = list(np.random.choice(np.arange(len(loader)), 9, replace=False))
    vbatch = []
    return vidx, vbatch


# noinspection PyTypeChecker
def StepLR(optimizer, step, schedule):
    """
    use the Adam optimizer with a constant leaning rate of 10^{−4}
        for the first 200k/100k/50k iterations,
        followed by an exponential decay of 0.999998/0.999992/0.999992
        until iteration 900k.
    Args:
        optimizer: 
        step:
        schedule: dict: {steps=array(50k, 100k, 200k, 900k),
                         lr=array(0.999992, 0.999992, 0.999998, 1.0)}

    Returns:

    """
    base_lr = 1e-5
    if schedule is None:
        schedule = dict(steps=np.array([50 * 1000, 100 * 1000, 200 * 1000, 9000 * 1000]),
                        lrs=np.array([0.999992, 0.999992, 0.999998, 1.0]))
    steps = schedule.get('steps')  # [50k, 100k, 200k]
    lrs = schedule.get('lrs')
    decay = lrs[len(steps) - sum(step < steps)]

    lr = optimizer.param_groups[0]['lr']
    lr = lr * decay
    lr = max(lr, base_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def update_flow(flow, persp):
    assert flow.shape[-1] == 2 and persp.shape[1:] == (3, 3)

    b, h, w, two = flow.shape
    flow = flow.reshape(b, -1, two)
    flow = torch.matmul(flow, persp[:, :, :2].permute(
        0, 2, 1)) + persp[:, :, 2].unsqueeze(1)
    flow /= flow[:, :, -1].unsqueeze(-1)
    flow = flow[:, :, :2].reshape(b, h, w, two).permute(0, 3, 1, 2)
    return flow


class Trainer(Crafter):
    def __init__(self, net, loader, loss, optimizer, wandbid=None):
        Crafter.__init__(self, net)
        self.loader = loader
        self.loss_func = loss
        self.optimizer = optimizer
        self.wandbid = wandbid

    def __call__(self, epoch: int, istrain=True, velid=False, visual=False):
        if istrain:
            # torch.set_grad_enabled(True)
            self.net.train()
            key = 'Train'
        else:
            if velid:
                self.net.eval()
                key = 'Velid'
            else:
                self.net.eval()
                key = 'Valid'

        # assert torch.is_grad_enabled() == istrain, '"grad_enabled" is differ "istrain"'
        stats = defaultdict(list)  # 简单来看, 当作一个 dictionary 即可

        loader = self.loader[key]
        loader.dataset.epoch += 1
        name = loader.dataset.dataset

        if velid:
            # ------------------------------
            # visualization model prediction
            # save results
            # ------------------------------
            if visual:
                bar = tqdm(
                    loader, bar_format="{l_bar}{bar:3}{r_bar}", ncols=110)

                visual_dir = os.path.join('visual', self.wandbid)
                eval_dir = os.path.join('eval', self.wandbid)

                os.makedirs(visual_dir, exist_ok=True)
                os.makedirs(eval_dir, exist_ok=True)
                eval_file = os.path.join(eval_dir, '{}.npy'.format(name))
                pred_dir = os.path.join(visual_dir, name)
                os.makedirs(pred_dir, exist_ok=True)

                cur = 1
                vertices = []
                filenames = []
                from modules.smpl import SMPL, get_smpl_vertices
                smpl = SMPL().cuda()
                rotmats = np.zeros((0, 16, 24, 3, 3))
                for iter, inputs in enumerate(bar):
                    inputs = self.todevice(inputs)
                    output = self.forward_net(inputs)
                    # pred_vertices_smpl = output['pred_vertices_smpl']
                    pred_rotmats = output['pred_rotmats']
                    B, T = pred_rotmats.shape[:2]
                    pred_vertices = get_smpl_vertices(output['trans'].reshape(
                        B * T, 3), pred_rotmats.reshape(B * T, 24, 3, 3), output['betas'].reshape(B * T, 10), smpl)
                    rotmats = np.concatenate(
                        (rotmats, pred_rotmats.cpu().detach().numpy()), axis=0)
                    for index in range(pred_vertices.shape[0]):
                        # save_ply(pred_vertices_smpl[index], os.path.join(
                        #     pred_vertices_smpl_dir, '{}.ply'.format(cur)))
                        vertices.append(
                            pred_vertices[index].squeeze().cpu().detach().numpy())
                        filenames.append(os.path.join(
                            pred_dir, '{}.ply'.format(cur)))
                        cur += 1

                # predict_mesh_file = os.path.join(
                #     out_file_path, 'predict_mesh.ply')
                # crop_image_file = os.path.join(
                #     out_file_path, 'correspond_images.jpg')
                # save_ply(pred_vertices_smpl, predict_mesh_file)
                # # crop_image(image_path[0], box, crop_image_file)
                # command_pc = 'cp {} {}'.format(
                #     points_path[0], out_file_path + '/points.ply')
                # command_mesh = 'cp {} {}'.format(
                #     mesh_path[0], out_file_path + '/mesh.ply')
                # os.system(command_pc)
                # os.system(command_mesh)
                np.save(eval_file, rotmats)
                return vertices, filenames
            # ------------------------------
            # evaluating
            # calculate metrics result
            # ------------------------------
            else:
                bar = tqdm(
                    loader, bar_format="{l_bar}{bar:3}{r_bar}", ncols=110)
                for iter, inputs in enumerate(bar):
                    inputs = self.todevice(inputs)
                    details = self.forward_val(inputs)
                    for k, v in details.items():
                        if type(v) is not dict:
                            if isinstance(v, torch.Tensor):
                                stats[k].append(v.detach().cpu().numpy())
                            else:
                                stats[k].append(v)
                    del inputs, details
                final_loss = {k: mean(v) for k, v in stats.items()}
        # ------------------------------
        #
        # training
        # ------------------------------
        else:
            vidx, vbatch = sample_visual_index(loader)
            bar = tqdm(loader, bar_format="{l_bar}{bar:3}{r_bar}", ncols=110)
            bar.set_description(f'{key} {epoch:02d}')
            for iter, inputs in enumerate(bar):
                inputs = self.todevice(inputs)
                # compute gradient and do model update
                if istrain:
                    self.optimizer.zero_grad()
                    details = self.forward_backward(inputs)
                    self.optimizer.step()
                else:
                    details = self.forward_val(inputs)

                for k, v in details.items():
                    if type(v) is not dict:
                        if isinstance(v, torch.Tensor):
                            stats[k].append(v.detach().cpu().numpy())
                        else:
                            stats[k].append(v)
                # if is training, print median stats on terminal
                if istrain:
                    N = len(stats['loss']) // 10 + 1
                    loss = stats['loss']
                    bar.set_postfix(loss=f'{mean(loss[:N]):06.06f} -> '
                                         f'{mean(loss[-N:]):06.06f} '
                                         f'({mean(loss):06.06f})')

                if not istrain and (iter + 1) == len(loader):
                    bar.set_postfix(loss=f'{mean(stats["loss"]):06.06f}')

                first_step = epoch == 1 and iter == 0 and istrain
                #log_step = (iter + 1) % wandb.config.log_interval == 0 and iter != 0 and istrain
                log_step = (iter + 1) % 1 == 0 and iter != 0 and istrain
                if first_step or log_step:
                    # Use trained image numbers as step
                    step = (epoch - 1) * len(loader.dataset) \
                        + (iter + 1) * loader.batch_size
                    logs = {k: mean(v) for k, v in stats.items()}
                    logs['lr'] = self.optimizer.param_groups[0]['lr']
                    logs['global_step'] = step
                    wandb.log(logs)
                    del logs
                del inputs, details
            # Visual selected images
            wandbid = [x for x in wandb.run.dir.split(
                '/') if wandb.run.id in x][-1]
            optim_dir = os.path.join('output', wandbid, 'optimizer')
            # epoch_visual(vbatch, f'{optim_dir}/{key}-E{epoch}.png')
            del vbatch

            # Summary of valid losses during this epoch
            final_loss = {k: mean(v) for k, v in stats.items()}

        return final_loss

    def forward_backward(self, inputs):
        raise NotImplementedError()

    def forward_net(self, inputs):
        raise NotImplementedError()

    def forward_val(self, inputs):
        raise NotImplementedError()
