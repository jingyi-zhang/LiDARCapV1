import glob
import os
import shutil
import sys
import wandb
import argparse
import torch
import torch.nn as nn

#import torch.multiprocessing as mp
#import torch.distributed as dist

sys.path.append(os.path.join('..', os.path.dirname(__file__)))
#from dataloader import Lidarcapv2_Dataset
from dataloader import Lidarcapv2_Dataset as Lidarcapv2_Dataset
from loss import mqh_Loss as Loss
from crafter import Trainer
from regressor import Regressor, LabelRegressor, SegmentRegressor


#torch.set_num_threads(1)       # 避免cpu利用率出现不合理的暴增

#torch.backends.`cudnn.benchmark = True

parser = argparse.ArgumentParser()
# bs
parser.add_argument('--bs', type=int, default=8,
                    help='input batch size for training (default: 24)')
parser.add_argument('--eval_bs', type=int, default=1,
                    help='input batch size for evaluation')
# threads
parser.add_argument('--threads', type=int, default=1,
                    help='Number of threads (default: 4)')

# epochs
parser.add_argument('--epochs', type=int, default=10000,
                    help='Traning epochs (default: 100)')

parser.add_argument('--ckpt_path', type=str, default=None,
                    help=f'the saved ckpt needed to be evaluated or visualized')

parser.add_argument('--visual', action='store_true')

parser.add_argument('--debug', action='store_true', default=False)

parser.add_argument('-g', '--gpus', default=1, type=int)

parser.add_argument('--group_name', default='test', type=str)
parser.add_argument('--wandbid', default=None, type=str)

parser.add_argument('--config', type=str, default='config')

def main():
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    if args.debug:
        os.environ["WANDB_MODE"] = 'disabled'

    if args.threads == 0:
        torch.set_num_threads(1)  # 此行可以规避这个bug：threads为0时cpu占用出现不合理的暴增
    else:
        torch.set_num_threads(1 + args.threads)

    wandb.init(project='lidarcapv2_final',entity='lidar_human',resume='allow',group=args.group_name, id=args.wandbid)
    wandb.config.update(args, allow_val_change=True)
    config = wandb.config

    model_dir = os.path.join(os.path.dirname(__file__), 'output', str(wandb.run.id))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=False)

    for f in glob.glob(os.path.join(os.path.dirname(__file__), "*.py")) + glob.glob(
         os.path.join(os.path.dirname(__file__), f"{args.config}.yaml")):
        shutil.copy(f, model_dir)

    from yacs.config import CfgNode
    cfg = CfgNode.load_cfg(open(os.path.join(os.path.dirname(__file__), f'{args.config}.yaml')))
    wandb.config.update(cfg, allow_val_change=False)

    testset = Lidarcapv2_Dataset(cfg.TestDataset)
    testset.__getitem__(0)

    valid_loader = torch.utils.data.DataLoader(
        testset,
        num_workers=config.threads,
        batch_size=config.eval_bs,
        shuffle=False,
        pin_memory=True,
    )

    if not args.visual:
        trainset = Lidarcapv2_Dataset(cfg.TrainDataset)  # 在训练时将use_aug设为False会有更快的收敛速度，而且没有任何性能下降

        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=config.threads,
            batch_size=config.bs,
            shuffle=True,
            pin_memory=True,
        )

        loader = dict(Train=train_loader, Valid=valid_loader)

    else:
        loader = dict(Valid=valid_loader)

    if cfg.TRAIN.with_body_label:
        net, loss = LabelRegressor()
    if cfg.TRAIN.segment_parallel:
        net = SegmentRegressor(cfg.MODEL)
        loss = Loss(cfg.LOSS)
    else:
        net = Regressor()
        loss = Loss(cfg.LOSS)

    net.cuda()
    loss.cuda()

    trainer = Trainer(net, loader, loss, cfg, 0, args.visual)
    #trainer.evaluate()
    if wandb.run.resumed:
        print(f'Resumed from ckpt_path:{args.ckpt_path}')
        start_epoch, best_performance = trainer.resume_pretrained(args.ckpt_path)
    elif args.ckpt_path is not None:
        start_epoch, best_performance = trainer.resume_pretrained(args.ckpt_path)
    else:
        start_epoch = 0

    trainer.fit(start_epoch, config.epochs, model_dir)


if __name__ == "__main__":
    main()