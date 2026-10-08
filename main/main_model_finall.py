import os
import time
import sys
sys.path.append(".")
import hashlib
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from loguru import logger
import scipy.io as sio
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt

from model.model_finall import HQSLearnableLRModel
from util.setup import setup_seed
from util.tools import get_formated_args
from data.calgary_campinas import CCDataset
from data.fast_mri import FastMriDataset
from data.dummy_dataset import Dummyataset
from model.loss import mae_loss

def train(args):
    setup_seed(args.seed)
    device = args.device
    batch_size = args.batch_size

    dir_name = "lr={:e}_seed={}_{}_{}_{}".format(
        args.lr, 
        args.seed,
        args.optimizer,
        args.criterion,
        args.dataset)
    

    # except_keys = ["dataset_root", "mask_path", "project_root_path"]
    except_keys = []
    args_hash_code = hashlib.md5(get_formated_args(args, except_keys=except_keys).encode()).hexdigest()
    logger.info(args_hash_code)
    checkpoint = os.path.join(args.project_root_path, "checkpoints")
    with open("{}/check_info.txt".format(checkpoint), "a+") as f:
        f.write("{}\n".format(args_hash_code))
    # args.checkpoint = os.path.join(checkpoint, args.model_name, dir_name, str(args_hash_code))
    args.checkpoint = os.path.join(checkpoint, args.model_name, str(args_hash_code) + "-" + args.tag)
    logger.info("checkpoint path root: {}".format(args.checkpoint))
    args.args_hash_code = args_hash_code

    # shutil.rmtree()

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    # logger.remove()
    # logger.add(sys.stdout, format="|  {level:<8} | {name}:{function}:{line} - {message}", level="INFO")

    if args.log_to_file:
        path = os.path.join(args.checkpoint, "log.txt")
        logger.add(path)
    logger.info(get_formated_args(args))

    path = os.path.join(args.checkpoint, "args_info.txt")
    args.time_stamp = "{0:%Y-%m-%d %H-%M-%S/}".format(datetime.now())
    with open(path, "w") as f:
        f.write(get_formated_args(args))
    time.sleep(5) # for checking wherther the args is right!
    

    # ################################ setup dataset ##########################
    if args.dataset == "cc":
        trainset_path = os.path.join(args.dataset_root, "train")
        valset_path = os.path.join(args.dataset_root, "val")
        trainset = CCDataset(trainset_path, args.mask_path, args.fft_norm, args.data_range)
        valset = CCDataset(valset_path, args.mask_path, args.fft_norm, args.data_range)
    elif args.dataset == "fastmri":
        trainset_path = os.path.join(args.dataset_root, "train")
        valset_path = os.path.join(args.dataset_root, "val")
        trainset = FastMriDataset(trainset_path, args.mask_path, args.fft_norm, args.data_range)
        valset = FastMriDataset(valset_path, args.mask_path, args.fft_norm, args.data_range)
    elif args.dataset == "dummy":
        trainset = Dummyataset()
        valset = Dummyataset()
    else:
        raise Exception("Unrecognise the model name: {}".format(args.dataset))
    
    logger.info("Trainset Size: {}".format(len(trainset)) )
    logger.info("Valset Size: {}".format(len(valset)) )
    
    trainset_dataloader = DataLoader(trainset, batch_size, shuffle=True)
    valset_dataloader = DataLoader(valset, batch_size, shuffle=True)


    # ############################## 1.setup model ##############################
    if args.model_name == "model_finall":
        model = HQSLearnableLRModel(args.niter, args, True)
        model.to(device)
        module_name = model.__module__.split(".")[-1]
        if args.model_name != module_name:
            raise Exception("The arg.model_name({}) is not equal to module_name({})".format(args.model_name, module_name))
    else:
        raise Exception("Unrecognise the model name: {}".format(args.model_name))
    
    
    # ############################## 2.setup optimizer ##########################
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), args.lr)
        model.optimizer = optimizer
    else:
        raise Exception("Unrecognise the optimizer: {}".format(args.optimizer))

    # ############################## 3.setup criterion ##########################
    if args.criterion == "l1_loss":
        # model.criterion = mae_loss
        model.criterion = nn.L1Loss()
    elif args.criterion == "mseloss":
        model.criterion = nn.MSELoss()
    else:
        raise Exception("Unrecognise the criterion: {}".format(args.criterion))
    # ############################## 4.setup metrics ##########################
    # logger.info(model)
    psnr = torchmetrics.PeakSignalNoiseRatio(data_range=args.data_range, reduction="none", dim=[-3, -2, -1])
    model.add_metric(psnr)
    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=args.data_range, reduction="none", dim=[-3, -2, -1])
    model.add_metric(ssim)

    # ############################## 5.initialize ##########################
    # path = os.path.join(args.checkpoint, "final_loss*.pkl")
    if args.param_mode == "best":
        path = os.path.join(args.checkpoint, "best_*.pkl")
    elif args.param_mode == "final":
        path = os.path.join(args.checkpoint, "final_*.pkl")
    else:
        raise Exception("Unrecognise the loading mode of param: {}".format(args.param_mode))
    files = glob.glob(path)
    if files:
        model.restore_state_dict(files[0], True)
    else:
        # model.initialize()
        pass
    # ############################## 7.training ##########################
    model.fit(trainset_dataloader, valset_dataloader, args.epochs, args.print_by_step)
    model.close()

  



def eval(args):
    setup_seed(args.seed)

    # read the command parameter about training
    path = os.path.join(args.checkpoint, "args_info.txt")
    with open(path, "r") as f:
        lines = f.readlines()
    
    logger.info("==========================================================")
    for line in lines:
        if ": " in line:
            key, val = line.split(": ")
            key = key.strip()
            val = val.strip()
            if hasattr(args, key):
                if str(getattr(args, key)) != str(val):
                    logger.warning("The value of {}, train={}, eval={}".format(key, val, getattr(args, key)))
            else:
                setattr(args, key, val)
    input("Enter any key to continue...")
    logger.info("==========================================================")
    logger.info(get_formated_args(args))

    time.sleep(5) # for checking wherther the args is right!
    

    # ############################## 1.setup model ##############################
    if args.model_name == "iteration_unrolling_hqs":
        model = HQSIterationUnrollingModel(args.niter, args, False)
    else:
        raise Exception("Unrecognise the model name: {}".format(args.model_name))

    # ############################## 2. Load model ##########################
    path = os.path.join(args.checkpoint, "{}_loss*.pkl".format(args.param_mode))
    files = glob.glob(path)
    model.restore_state_dict(files[0], training=False)
    model.to(args.device)


    # ############################## 3. Config metric ##########################
    psnr = torchmetrics.PeakSignalNoiseRatio(data_range=255)
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=255)

    # ############################## 4, config input ##########################
    mask = sio.loadmat(args.mask_path)["mask"]

    metrics = [psnr, ssim]
    index = 80
    targets = sio.loadmat("./data/cc/val/e14155s3_P69120.7.mat")["image"] * 255
    mask = sio.loadmat(args.mask_path)["mask"]
    print(targets.shape)
    print(mask.shape)
    targets = torch.from_numpy(targets)
    mask = torch.from_numpy(mask)[None]


    kspaces = torch.fft.ifftshift(torch.fft.fft2(targets), dim=[-2, -1])

    zf_kspaces = kspaces * mask
    kspaces = torch.fft.ifftshift(torch.fft.fft2(targets), dim=[-2, -1])

    plt.subplot(3, 2, 1)
    plt.imshow(targets[index].abs().numpy(), cmap="gray")

    plt.subplot(3, 2, 2)
    plt.imshow(torch.log(1 + kspaces[index].abs()).numpy(), cmap="gray")

    plt.subplot(3, 2, 3)
    plt.imshow(zf_imgs[index].abs().numpy(), cmap="gray")

    plt.subplot(3, 2, 4)
    plt.imshow(torch.log(1 + zf_kspaces[index].abs()).numpy(), cmap="gray")
    # plt.show()

    def create_input_by_format(x, index):
        output = x[index]
        output = output[None, None, :, :]
        return output

    zf_img = create_input_by_format(zf_imgs, index)
    zf_kspace = create_input_by_format(zf_kspaces, index)
    target = create_input_by_format(targets, index)
    mask = mask.expand_as(target)

    inputs = [zf_img, zf_kspace, targets, mask]
    pred = model.predict(inputs)

    from thop import profile
    flops, params = profile(model.predict, inputs)
    
    x = 1
    plt.subplot(3, 2, 5)
    plt.imshow(pred[0].detach().cpu().abs().numpy(), cmap="gray")
    plt.show()

    target = torch.squeeze(target)
    pred = torch.squeeze(pred).to("cpu")
    zf_img = torch.squeeze(zf_img)
    print(type(target), type(pred), type(zf_img))
    print(target.is_cuda, pred.is_cuda, zf_img.is_cuda)
    
    target = target[None, None]
    pred = pred[None, None]
    
    zf_img = zf_img[None, None]
    for metric in metrics:
        pred_val = metric(pred.abs(), target.abs())
        zf_val = metric(zf_img.abs(), target.abs())
        logger.info("{}: zf={}, rec={}".format(metric.__class__.__name__, zf_val, pred_val))
        


    



#########################################################################################################################
def set_train_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name', type=str, default='base_model', help='which model to train or eval')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--yaml", type=str, default=None, help="yaml configuration to update args")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="which device to run model")
    parser.add_argument("--log_to_file", action="store_true", help="print log to file")
    parser.add_argument("--project_root_path", type=str, required=True, help="the root path of the all output which to save info")
    parser.add_argument('--param_mode', type=str, default="best", choices=["best", "finall"], help='which model parameter to initialize the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--sparsity_loss_weight', type=float, default=1e-3, help='the weight of the encoder-decoder of sparsity.')
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument('--criterion', type=str, default="l1_loss", help='loss function')
    parser.add_argument('--dataset', type=str, default="cc", choices=["cc", "fastmri", "dummy"], help='dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='the path root of dataset')
    parser.add_argument('--mask_path', type=str, required=True, help='the path root of mask')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    # parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_by_step', type=int, default=10, help="The step to display train information.")
    parser.add_argument('--fft_norm', action="store_true", help="tge type of normalization in FFT")
    parser.add_argument('--vis_server', type=str, default="192.168.1.246", help='the ip of the visdom server.')
    parser.add_argument("--niter", type=int,  required=True, help="The number of iteration block.")
    parser.add_argument("--tag", type=str,  default="default", help="Distinguish the different version of same parameters")
    parser.add_argument("--print_exception", action="store_true",  help="Wether to print the exception from visdom")
    parser.add_argument('--gradient_clip', action="store_true", help="tge type of normalization in FFT")
    parser.add_argument('--topk', type=int, default=5, help="how many patch are selected for exemplar in every distance function")
    parser.add_argument('--num_basis', type=int, default=16, help="how many rank-one tensor")
    parser.add_argument("--patch_size", type=int, nargs="+", required=True, help="The size of the block in non-local")
    parser.add_argument('--data_range', type=int, default=255, help="The data range of dataset")
    


def set_eval_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name', type=str, default='base_model', help='which model to  eval')
    parser.add_argument('--mask_path', type=str, required=True, help='the path root of mask')
    parser.add_argument('--param_mode', type=str, required=True, choices=["best", "final"], help='which model param to initialite model.')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--niter", type=int,  required=True, help="The number of iteration block.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="which device to run model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(required=True)

    # ######################################## train subparser #######################################
    train_parser = subparser.add_parser(name='train', help='train model')
    train_parser.set_defaults(func=train)
    set_train_parser(train_parser)

    # ######################################## eval subparser ########################################
    eval_parser = subparser.add_parser(name='eval', help='eval model')
    eval_parser.set_defaults(func=eval)
    set_eval_parser(eval_parser)

    #################################################################################################
    args = parser.parse_args()
    args.func(args)
    print("*****************************")