# ..\Venv\Scripts\activate
# run: torchrun --nproc_per_node=1 --master_port 39985 main_unet.py
import datetime
import os
from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

import torch
from torch import nn
from model import Model
# from models.encoder_decoder import Encoder_Decoder
from models.attetion_unet import AttU_Net
from loss import *
# from train_and_test import train_and_test, test
# from net_params import *
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loader import count_offset, load_mask, create_dataloaders
import argparse
from collections import OrderedDict
from utils import *
from plotter import plot_train_loss, plot_predictions
from utils import normalize_data_cuda
from loader import scale_to_bins, load_np_data


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
    mask = load_mask(cfg.root_path)

    LR = cfg.LR
    parser = argparse.ArgumentParser()
    start_year = cfg.start_year
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

    model = AttU_Net(cfg.in_len*cfg.features_amount, cfg.out_len*cfg.features_amount, (cfg.batch, cfg.height, cfg.width), 3, 1, 0)

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
    train_data, valid_data, test_data = create_dataloaders(cfg.root_path, start_year, end_year, cfg)

    # loss
    criterion = Loss_MSE().cuda()

    if cfg.nn_mode == 'train':
        train_sampler = DistributedSampler(train_data, shuffle=True)
        train_loader = DataLoader(train_data, num_workers=threads, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                                  sampler=train_sampler)
        train_loss = np.zeros(cfg.epoch, dtype=float)
        for epoch in range(1, cfg.epoch + 1):
            epoch_loss = 0.0
            print(f'Epoch {epoch}/{cfg.epoch}', flush=True)
            for idx, train_batch in enumerate(train_loader):
                # print(train_batch.shape)
                train_batch = normalize_data_cuda(train_batch, cfg.min_vals, cfg.max_vals)
                optimizer.zero_grad()
                train_pred = model(train_batch[:, :cfg.in_len])

                if cfg.model_name == 'Attention U-net labels':
                    loss_labels = criterion(train_batch[:, cfg.in_len + cfg.out_len:, :3], train_pred[:, :, :3])
                    loss_labels.backward()
                    optimizer.step()
                    loss_labels = reduce_tensor(loss_labels)  # all reduce
                    epoch_loss += loss_labels.item()
                else:
                    loss = criterion(train_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3], train_pred[:, :, :3])
                    loss.backward()
                    optimizer.step()
                    loss = reduce_tensor(loss)  # all reduce
                    epoch_loss += loss.item()

            train_loss[epoch-1] = epoch_loss
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
            print(f'Saving model to {model_save_path}/features_{cfg.features_amount}_epoch_{cfg.epoch}.pth')
            torch.save(model.state_dict(), f'{model_save_path}/features_{cfg.features_amount}_epoch_{cfg.epoch}.pth')

    elif cfg.nn_mode == 'test':
        flux_array, SST_array, press_array = load_np_data(cfg.root_path, start_year, end_year)
        flux_array_scaled, flux_quantiles = scale_to_bins(flux_array, bins=cfg.bins)
        SST_array_scaled, SST_quantiles = scale_to_bins(SST_array, bins=cfg.bins)
        press_array_scaled, press_quantiles = scale_to_bins(press_array, bins=cfg.bins)
        quantiles = [flux_quantiles, SST_quantiles, press_quantiles]

        test_sampler = DistributedSampler(test_data, shuffle=False)
        test_loader = DataLoader(test_data, num_workers=threads, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                                  sampler=test_sampler)
        if is_master_proc():
            with ((torch.no_grad())):
                for idx, test_batch in enumerate(test_loader):
                    test_batch = normalize_data_cuda(test_batch, cfg.min_vals, cfg.max_vals)

                    if cfg.model_name == 'Attention U-net labels':
                        test_pred = model(test_batch[:, :cfg.in_len])
                        loss_labels = criterion(test_batch[:, cfg.in_len + cfg.out_len:, :3], test_pred[:, :, :3])
                        test_pred_labels = test_pred * cfg.bins
                        test_pred_labels = test_pred_labels.int()
                        test_pred_values = torch.zeros_like(test_pred)
                        for channel in range(3):
                            for q in range(cfg.bins):
                                test_pred_values[:, :, channel][test_pred_labels[:, :, channel] == q+1] = (quantiles[channel][q] + quantiles[channel][q+1])/2

                        for channel in range(3):
                            test_pred_values[:, :, channel] = (test_pred_values[:, :, channel] - cfg.min_vals[channel]) / (cfg.max_vals[channel] - cfg.min_vals[channel])

                    else:
                        test_pred_values = model(test_batch[:, :cfg.in_len])
                        loss = criterion(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3], test_pred_values[:, :, :3])

                    test_batch_scaled = test_batch.clone().detach()

                    truth = test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len].detach().clone()
                    prediction = test_pred_values.detach().clone()
                    truth = truth.cpu().numpy()
                    prediction = prediction.cpu().numpy()
                    ssim = get_SSIM(prediction, truth)
                    # print(ssim.shape)
                    # print(ssim)
                    print(np.mean(ssim, axis=(0, 1)))
                    print(np.mean(ssim))


                    for channel in range(3):
                        test_batch_scaled[:, :cfg.in_len + cfg.out_len, channel] = \
                            test_batch_scaled[:, :cfg.in_len + cfg.out_len, channel] * (cfg.max_vals[channel] - cfg.min_vals[channel]) + cfg.min_vals[channel]

                        test_pred_values[:, :, channel] = test_pred_values[:, :, channel] * (cfg.max_vals[channel] - cfg.min_vals[channel]) + cfg.min_vals[channel]

                    # print(cfg.min_vals)
                    # print(cfg.max_vals)
                    # print(tuple(torch.amin(test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len, :3], dim=(0, 1, 3, 4))))
                    # print(tuple(torch.amax(test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len, :3], dim=(0, 1, 3, 4))))
                    # print(tuple(torch.amin(test_pred_values[:, :, :3], dim=(0, 1, 3, 4))))
                    # print(tuple(torch.amax(test_pred_values[:, :, :3], dim=(0, 1, 3, 4))))

                    loss_values = criterion(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3],
                                            test_pred_values[:, :, :3])


                    differ = test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len] - test_pred_values
                    loss_divided = torch.mean(torch.sum(differ ** 2, (3, 4)), (0, 1))


                    # raise ValueError

                    if cfg.model_name == 'Attention U-net labels':
                        print(f'Idx {idx}, labels loss: {loss_labels}', flush=True)
                    print(loss_divided)
                    print(f'Idx {idx}, values loss: {loss_values}', flush=True)
                    for i in range(cfg.batch):
                        day = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=offset + idx + i)
                        test_batch_numpy = test_batch_scaled[i, :, :3].cpu().numpy()
                        test_pred_numpy = test_pred_values[i, :, :3].cpu().numpy()
                        test_pred_labels_numpy = test_pred_labels[i, :, :3].cpu().numpy()
                        plot_predictions(cfg.root_path, test_batch_numpy,
                                         test_pred_numpy, cfg.model_name,
                                         cfg.features_amount, day, mask, cfg)

                        plot_predictions(cfg.root_path, np.array(test_pred_labels_numpy, dtype=float),
                                         np.array(test_pred_labels_numpy, dtype=float), cfg.model_name + ' labels',
                                         cfg.features_amount, day, mask, cfg)
                    if idx >= 0:
                        break
