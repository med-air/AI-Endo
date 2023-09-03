import os.path
import warnings
import logging
warnings.filterwarnings("ignore")
import pickle
import random
import argparse
import numpy as np
from setproctitle import setproctitle
from pytorch_metric_learning import losses

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.esd import ESDDataset
from utils.parser import ParserUse
from utils.util import bcolors, get_lr, plot_loss

from model.resnet import ResNet


def train_resnet(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info("|| "*10 + "Begin training resnet50")

    setproctitle("1Resnet")
    model = ResNet(has_fc=True)
    if os.path.isfile(args.start_iter):
        paras = torch.load(args.start_iter)["model"]
        model.load_state_dict(paras)
    model.cuda()

    optimizer = optim.SGD(params=model.parameters(),
                          lr=args.resnet_lr,
                          momentum=args.resnet_momentum,
                          weight_decay=args.resnet_weight_decay)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.resnet_decay_steps, gamma=0.1)

    con_loss = losses.SupConLoss(temperature=0.08)
    ce_loss = torch.nn.CrossEntropyLoss()

    with open(args.data_file, "rb") as f:
        data_dict = pickle.load(f)
    train_dataset = ESDDataset(data_dict=data_dict, data_idxs=args.train_names, is_train=True, get_name=True, class_weights=args.sample_weights)
    val_dataset = ESDDataset(data_dict=data_dict, data_idxs=args.val_names, is_train=False, get_name=True)
    print(f"Length of validation {len(val_dataset)}, data idxs {args.val_names}")
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.resnet_train_bs, num_workers=args.num_worker, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.resnet_train_bs, num_workers=args.num_worker, shuffle=True)

    iterations = 1
    print("Totally {} iterations for one epoch".format(len(train_loader)))
    best_loss = 10000
    train_losses = []
    val_Losses = []
    while iterations < args.resnet_iterations:
        for data in train_loader:
            torch.cuda.empty_cache()
            model.train()
            imgs, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True).squeeze()
            cls, embeds = model(imgs)
            embeds = F.normalize(embeds, p=2.0, dim=-1)
            loss_con = con_loss(embeds, labels)
            loss_ce = ce_loss(cls, labels)
            loss = loss_ce + 0.25 * loss_con

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations % 100 == 0:
                logging.info("Iterations {:>10d} / {}, Con_Loss {:>10.5f}".format(iterations, args.resnet_iterations, loss.item()))
                train_losses.append([iterations, loss.item()])

            if iterations % 400 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = []
                    for data in val_loader:
                        imgs, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True).squeeze()
                        cls, embeds = model(imgs)
                        embeds = F.normalize(embeds, p=2.0, dim=-1)
                        loss_con = con_loss(embeds, labels)
                        loss_ce = ce_loss(cls, labels)
                        loss = loss_ce + 0.25 * loss_con
                        val_loss.append(loss.cpu().item())
                mean_loss = sum(val_loss) / len(val_loss)
                val_Losses.append([iterations, mean_loss])
                logging.info(">> " * 10 + "Evaluation at iterations {:>10d} is {:>10.5f}".format(iterations, mean_loss))
                logging.info("Learning rate {}".format(get_lr(optimizer)))
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    save_file = os.path.join(args.save_model, "resnet50_{}_best.pth".format(args.log_time))
                    args.resnet_model = save_file
                    torch.save({"model": model.state_dict(),
                                "optim": optimizer.state_dict()}, save_file)
                    logging.info("Saving model at itreation {}".format(iterations))

                plot_loss(train_losses, val_Losses, "./tem/{}_resnet50_loss.pdf".format(args.log_time))

            lr_scheduler.step()
            iterations += 1
            if iterations > args.resnet_iterations:
                break

    if not os.path.isdir(args.save_model):
        os.makedirs(args.save_model)

    save_file = os.path.join(args.save_model, "resnet50_{}_last.pth".format(args.log_time))
    args.resnet_model = save_file
    torch.save({"model": model.state_dict(),
                "optim": optimizer.state_dict()}, save_file)

    logging.info("Trained model saved to {}".format(save_file))

    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='train', required=True, type=str,
                        help='Your detailed configuration of the network')
    parser.add_argument("-n", default="", type=str, help="Notes for paras")
    args = parser.parse_args()
    args = ParserUse(args.cfg, log="resnet").add_args(args)
    ckpts = args.makedir()

    logging.info(args)
    logging.info("====" * 10)
    logging.info(f"{bcolors.OKCYAN} Make sure weights of phases are updated\n {args.class_weights}. {bcolors.ENDC}")
    train_resnet(args)