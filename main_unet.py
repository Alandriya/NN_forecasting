# ..\Venv\Scripts\activate
# run: torchrun --nproc_per_node=1 --master_port 39985 main_unet.py
import datetime
import os

from sympy.physics.units import amount
from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

import torch
from torch import nn
from model import Model
# from models.encoder_decoder import Encoder_Decoder
from models.attetion_unet import AttU_Net
from models.dense_unet import DenseU_Net
from loss import *
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loader import count_offset, load_mask, create_dataloaders
import argparse
from collections import OrderedDict
from utils import *
from plotter import plot_train_loss, plot_predictions
from utils import normalize_data_cuda
from loader import scale_to_bins, load_np_data, Data2


if __name__ == '__main__':
    fix_random(2024)
    mask = load_mask(cfg.root_path)

    LR = cfg.LR
    parser = argparse.ArgumentParser()
    start_year = 1979
    end_year, offset = count_offset(start_year)

    # # plot train loss
    # loss_list = list()
    # for year in [1979, 1989, 1999, 2009]:
    #     if os.path.exists(cfg.root_path + f'Losses/loss_{cfg.model_name}_{year}.npy'):
    #         loss_arr = np.load(cfg.root_path + f'Losses/loss_{cfg.model_name}_{year}.npy')
    #         loss_list += list(loss_arr)
    # plot_train_loss(cfg.root_path, np.array(loss_list), 1979, 2009, cfg.model_name)
    # raise ValueError

    # parallel group
    torch.distributed.init_process_group(backend="gloo")
    threads = cfg.dataloader_thread

    model = AttU_Net(cfg.in_len*3, cfg.out_len*3, (cfg.batch, cfg.height, cfg.width), 3, 1, 0)
    # model =DenseU_Net(cfg.in_len * 3, cfg.out_len * 3,
    #                  (cfg.batch, cfg.height, cfg.width), 3, 1, 0)

    # optimizer
    if cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    elif cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        optimizer = None

    # loss
    criterion_MSE = Loss_MSE().cuda()
    # criterion_simple = True
    criterion = Loss_MSE_informed().cuda()
    criterion_simple = False

    alpha = 0.64
    beta = 0.32
    model_load_path =  cfg.GLOBAL.MODEL_LOG_SAVE_PATH + f'/models/days_{cfg.out_len}_simple.pth'
    # model_save_path = model_load_path
    model_save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH + f'/models/days_{cfg.out_len}_epoch_{cfg.epoch}_alpha_{int(alpha*10)}_beta_{int(beta*10)}.pth'
    # model_load_path = model_save_path
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    # reading model weights if save exists
    print(f'Trying to read {model_load_path},\n exists = {os.path.exists(model_load_path)}\n')
    if cfg.LOAD_MODEL and os.path.exists(model_load_path):
        print('Loading model')
        # original saved file with DataParallel
        state_dict = torch.load(model_load_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            if not ('total' in k):
                new_state_dict[k] = v
        # load params
        model.load_state_dict(new_state_dict)

    # model_save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH + f'/models/features_{cfg.features_amount}_days_{cfg.out_len}_epoch_{cfg.epoch}.pth'
    # create train and test dataloaders, train from 01.01.1979 to 01.01.2024, test from 01.01.2024 to 28.11.2024
    days_delta1 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    train_data = Data2(cfg, 0, days_delta1)
    test_data = Data2(cfg, days_delta1, days_delta1 - cfg.in_len - cfg.out_len + days_delta2)


    if cfg.nn_mode == 'train':
        train_sampler = DistributedSampler(train_data, shuffle=True)
        train_loader = DataLoader(train_data, num_workers=threads, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                                  sampler=train_sampler)
        train_loss = np.zeros(cfg.epoch, dtype=float)
        for epoch in range(1, cfg.epoch + 1):
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_loss_simple = 0.0
            print(f'Epoch {epoch}/{cfg.epoch}', flush=True)
            for idx, train_batch in enumerate(train_loader):
                # print(train_batch.shape) # torch.Size([16, 12, 12, 81, 91])
                optimizer.zero_grad()
                train_batch = normalize_data_cuda(train_batch, cfg.min_vals, cfg.max_vals)
                input = train_batch[:, :cfg.in_len, :3].clone()
                train_batch = train_batch.cuda()
                input = input.cuda()
                train_pred = model(input)
                if criterion_simple:
                    loss = criterion(train_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3], train_pred[:, :, :3])
                else:
                    loss = criterion(train_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3],
                              train_pred[:, :, :3], train_batch[:, :cfg.out_len, 3:], alpha, beta)
                    loss_simple = criterion_MSE(train_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3],
                              train_pred[:, :, :3])
                loss.backward()
                optimizer.step()
                loss = reduce_tensor(loss)  # all reduce
                epoch_loss += loss.item()
                if not criterion_simple:
                    epoch_loss_simple += loss_simple.item()

            train_loss[epoch-1] = epoch_loss
            if is_master_proc():
                if criterion_simple:
                    print(f'Loss: {epoch_loss}', flush=True)
                else:
                    print(f'Loss, alpha = {alpha}: {epoch_loss}', flush=True)
                    print(f'Loss MSE: {epoch_loss_simple}\n', flush=True)
            alpha /= 2
            beta /= 2

        if is_master_proc():
            np.save(cfg.root_path + f'Losses/loss_{cfg.model_name}_{cfg.start_year}_alpha_{int(alpha*10)}.npy', train_loss)
            torch.distributed.destroy_process_group()
            save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH
            if cfg.DELETE_OLD_MODEL and os.path.exists(save_path):
                shutil.rmtree(save_path)
            # model_save_path = os.path.join(save_path, 'models')
            log_save_path = os.path.join(save_path, 'logs')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                os.makedirs(model_save_path)
                os.makedirs(log_save_path)
            print(f'Saving model to {model_save_path}')
            torch.save(model.state_dict(), f'{model_save_path}')

    elif cfg.nn_mode == 'test':
        flux_quantiles = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_diff_quantiles.npy')
        sst_quantiles = np.load(cfg.root_path + f'DATA/SST_1979-2025_diff_quantiles.npy')
        press_quantiles = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_diff_quantiles.npy')
        quantiles = [flux_quantiles, sst_quantiles, press_quantiles]

        test_sampler = DistributedSampler(test_data, shuffle=False)
        test_loader = DataLoader(test_data, num_workers=threads, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                                  sampler=test_sampler)

        if is_master_proc():
            with ((torch.no_grad())):
                ssim_flux = 0.0
                ssim_sst = 0.0
                ssim_press = 0.0
                mse = 0.0
                amount = 0
                for idx, test_batch in enumerate(test_loader):
                    print(f'batch {idx}')
                    test_batch = normalize_data_cuda(test_batch, cfg.min_vals, cfg.max_vals)
                    input = test_batch[:, :cfg.in_len, :3].clone()
                    test_batch = test_batch.cuda()
                    input = input.cuda()
                    test_pred_values = model(input)
                    # test_pred_values = model(test_batch[:, :cfg.in_len, :3])
                    if criterion_simple:
                        loss = criterion(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3], test_pred_values[:, :, :3])
                    else:
                        loss = criterion(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3],
                                     test_pred_values[:, :, :3], test_batch[:, :cfg.out_len, 3:], alpha, beta)
                        loss_simple = criterion_MSE(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3], test_pred_values[:, :, :3])

                    mse += loss_simple.item()
                    test_batch_scaled = test_batch.clone().detach()
                    truth = test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len].detach().clone()
                    prediction = test_pred_values.detach().clone()
                    truth = truth.cpu().numpy()
                    prediction = prediction.cpu().numpy()

                    ssim = get_SSIM(prediction, truth)
                    ssim_arr = np.mean(ssim, axis=(0, 1))
                    ssim_flux += ssim_arr[0]
                    ssim_sst += ssim_arr[1]
                    ssim_press += ssim_arr[2]

                    amount += 1

                    for channel in range(3):
                        test_batch_scaled[:, :cfg.in_len + cfg.out_len, channel] *= (cfg.max_vals[channel] - cfg.min_vals[channel])
                        test_batch_scaled[:, :cfg.in_len + cfg.out_len, channel] += cfg.min_vals[channel]

                        test_pred_values[:, :, channel] *= (cfg.max_vals[channel] - cfg.min_vals[channel])
                        test_pred_values[:, :, channel] += cfg.min_vals[channel]

                    print(cfg.min_vals)
                    print(cfg.max_vals)
                    print(tuple(torch.amin(test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len, :3], dim=(0, 1, 3, 4))))
                    print(tuple(torch.amax(test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len, :3], dim=(0, 1, 3, 4))))
                    print(tuple(torch.amin(test_pred_values[:, :, :3], dim=(0, 1, 3, 4))))
                    print(tuple(torch.amax(test_pred_values[:, :, :3], dim=(0, 1, 3, 4))))

                    if criterion_simple:
                        loss_values = criterion(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3],
                                                test_pred_values[:, :, :3])
                    else:
                        loss_values = criterion(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3],
                                  test_pred_values[:, :, :3], test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, 3:6])

                    differ = test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len, :3] - test_pred_values
                    loss_divided = torch.mean(torch.sum(differ ** 2, (3, 4)), (0, 1))

                    if idx == 0:
                        for i in range(cfg.batch):
                            day = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=days_delta1 + offset + idx + i)
                            test_batch_numpy = test_batch_scaled[i, :, :3].cpu().numpy()
                            test_pred_numpy = test_pred_values[i, :, :3].cpu().numpy()
                            plot_predictions(cfg.root_path, test_batch_numpy[cfg.in_len:cfg.in_len + cfg.out_len],
                                             test_pred_numpy, cfg.model_name,
                                             cfg.features_amount, day, mask, cfg, postfix_simple=False)
                        # break
            print(f'SSIM flux = {ssim_flux/amount:.4f}')
            print(f'SSIM sst = {ssim_sst / amount:.4f}')
            print(f'SSIM press = {ssim_press/ amount:.4f}')
            print(f'MSE = {mse / amount}')