# ..\venv_base\Scripts\activate
# run: torchrun --nproc_per_node=4 --master_port 39985 main_enc_dec.py
import os
from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

import torch
from torch import nn
from model import Model
from models.encoder_decoder import Encoder_Decoder
from loss import Loss, Loss2
from train_and_test import train_and_test, test
from net_params import *
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from loader import create_dataloader_encoder_decoder, count_offset
import argparse
from collections import OrderedDict
from utils import *
from plotter import plot_train_loss


# fix random seed
def fix_random(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    fix_random(2024)

    LR = cfg.LR
    parser = argparse.ArgumentParser()
    start_year = cfg.start_year
    end_year, offset = count_offset(start_year)
    #
    # # plot loss
    # loss_arr = np.load(cfg.root_path + f'Losses/loss_{cfg.model_name}_{cfg.start_year}.npy')
    # plot_train_loss(cfg.root_path, loss_arr, start_year, end_year, cfg.model_name)
    # raise ValueError

    # parallel group
    torch.distributed.init_process_group(backend="gloo")

    # model
    model = Encoder_Decoder(cfg.in_len * cfg.features_amount, cfg.out_len * cfg.features_amount, (cfg.batch, cfg.height, cfg.width), 3, 1, 0)
    # print(cfg.batch)

    # optimizer
    if cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    elif cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        optimizer = None


    model_save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH + f'/models/features_{cfg.features_amount}_epoch_{cfg.epoch}.pth'
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    print(f'Trying to read {model_save_path},\n exists = {os.path.exists(model_save_path)}\n')

    if cfg.LOAD_MODEL and os.path.exists(model_save_path):
        print('Loading model')
        # original saved file with DataParallel
        state_dict = torch.load(model_save_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            if not ('total' in k):
                new_state_dict[k] = v
        # load params
        model.load_state_dict(new_state_dict)

    threads = cfg.dataloader_thread
    train_data = create_dataloader_encoder_decoder(cfg.root_path, start_year, end_year, cfg)

    # loss
    criterion = Loss2().cuda()

    if cfg.enc_dec_mode == 'train':
        train_sampler = DistributedSampler(train_data, shuffle=True)
        train_loader = DataLoader(train_data, num_workers=threads, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                                  sampler=train_sampler)
        train_loss = np.zeros(cfg.epoch, dtype=float)
        eta = 1.0
        delta = 1 / (cfg.epoch * len(train_loader))
        for epoch in range(1, cfg.epoch + 1):
            epoch_loss = 0.0
            print(f'Epoch {epoch}/{cfg.epoch}', flush=True)
            for idx, train_batch in enumerate(train_loader):
                train_batch = train_batch.cuda()
                train_pred = model(train_batch[:, :cfg.in_len], 'train')
                loss = criterion(train_batch[:, cfg.in_len:, :3], train_pred[:, :, :3])
                loss.backward()
                optimizer.step()
                loss = reduce_tensor(loss)  # all reduce
                epoch_loss += loss.item()
                eta -= delta
                eta = max(eta, 0)

            train_loss[epoch] = epoch_loss
            if is_master_proc():
                print(f'Loss: {epoch_loss}', flush=True)

        if is_master_proc():
            np.save(cfg.root_path + f'Losses/loss_{cfg.model_name}_{cfg.start_year}.npy', train_loss)
            torch.distributed.destroy_process_group()
            save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH
            if cfg.DELETE_OLD_MODEL and os.path.exists(save_path):
                shutil.rmtree(save_path)
            model_save_path = os.path.join(save_path, 'models')
            log_save_path = os.path.join(save_path, 'logs')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                os.makedirs(model_save_path)
                os.makedirs(log_save_path)

    elif cfg.enc_dec_mode == 'encode':
        train_sampler = DistributedSampler(train_data, shuffle=False)
        train_loader = DataLoader(train_data, num_workers=threads, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                                  sampler=train_sampler)
        if not os.path.exists(cfg.root_path + 'Encoded'):
            os.mkdir(cfg.root_path + 'Encoded')
        for start_year in [1979, 1989, 1999, 2009, 2019]:
            print(f'Encoding {start_year}')
            for idx, train_batch in enumerate(train_loader):
                train_pred = model(train_batch[:, :cfg.in_len].cuda(), 'encode')
                torch.save(train_pred, cfg.root_path + f'Encoded/{start_year}_{idx}_encoding.pt')

    elif cfg.enc_dec_mode == 'decode':
        train_sampler = DistributedSampler(train_data, shuffle=False)
        train_loader = DataLoader(train_data, num_workers=threads, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                                  sampler=train_sampler)
        if not os.path.exists(cfg.root_path + 'Decoded'):
            os.mkdir(cfg.root_path + 'Decoded')
        for start_year in [1979, 1989, 1999, 2009, 2019]:
            print(f'Decoding {start_year}')
            for idx, train_batch in enumerate(train_loader):
                train_pred = model(train_batch[:, :cfg.in_len].cuda(), 'decode')
                torch.save(train_pred, cfg.root_path + f'Decoded/{start_year}_{idx}_decoding.pt')
