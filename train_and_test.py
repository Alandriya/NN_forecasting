import datetime
from copy import deepcopy
from config import cfg
import numpy as np
from utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import shutil
import pandas as pd
import time
from thop import profile
from plotter import plot_predictions
from loader import load_mask


def train_and_test(model, optimizer, criterion, train_epoch, valid_epoch, loader, train_sampler):
    train_valid_metrics_save_path, model_save_path, writer, save_path, test_metrics_save_path = [None] * 5
    train_loader, test_loader, valid_loader = loader
    start = time.time()
    eval_ = Evaluation(seq_len=OUT_LEN, use_central=False)
    if is_master_proc():
        save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH
        if cfg.DELETE_OLD_MODEL and os.path.exists(save_path):
            shutil.rmtree(save_path)
        model_save_path = os.path.join(save_path, 'models')
        log_save_path = os.path.join(save_path, 'logs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(model_save_path)
            os.makedirs(log_save_path)
        # test_metrics_save_path = os.path.join(save_path, "test_metrics.xlsx")
        writer = SummaryWriter(log_save_path)
    train_loss = 0.0
    params_lis = []
    eta = 1.0
    delta = 1 / (train_epoch * len(train_loader))
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True)
    for epoch in range(1, train_epoch + 1):
        if is_master_proc():
            print('epoch: ', epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        for idx, train_batch in enumerate(train_loader, 1):
            #print(type(train_batch))
            train_batch = normalize_data_cuda(train_batch, cfg.min_vals, cfg.max_vals)
            optimizer.zero_grad()
            train_pred, decouple_loss = model([train_batch, eta, epoch], mode='train')
            # for channel in range(3):
            #     train_pred[:, :, channel] = (train_pred[:, :, channel] * (cfg.max_vals[channel] - cfg.min_vals[channel]) +
            #                                  cfg.min_vals[channel])
            loss = criterion(train_batch[:, IN_LEN:, :3], train_pred, decouple_loss)
            loss.backward()
            optimizer.step()
            loss = reduce_tensor(loss)  # all reduce
            train_loss += loss.item()
            eta -= delta
            eta = max(eta, 0)
            # pbar.update(1)

            # compute Params and FLOPs for Generator
            if epoch == 1 and idx == 1 and is_master_proc():
                Total_params = 0
                Trainable_params = 0
                NonTrainable_params = 0
                for param in model.parameters():
                    mulValue = param.numel()
                    Total_params += mulValue
                    if param.requires_grad:
                        Trainable_params += mulValue
                    else:
                        NonTrainable_params += mulValue
                Total_params = np.around(Total_params / 1e+6, decimals=decimals)
                Trainable_params = np.around(Trainable_params / 1e+6, decimals=decimals)
                NonTrainable_params = np.around(NonTrainable_params / 1e+6, decimals=decimals)
                flops, _ = profile(model.module, inputs=([train_batch, eta, epoch], 'train',))
                flops = np.around(flops / 1e+9, decimals=decimals)
                params_lis.append(Total_params)
                params_lis.append(Trainable_params)
                params_lis.append(NonTrainable_params)
                params_lis.append(flops)
                print(f'Total params: {Total_params}M')
                print(f'Trained params: {Trainable_params}M')
                print(f'Untrained params: {NonTrainable_params}M')
                print(f'FLOPs: {flops}G')
        # pbar.close()

        # valid
        if epoch % valid_epoch == 0:
            train_loss = train_loss / (len(train_loader) * valid_epoch)
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for valid_batch in valid_loader:
                    valid_batch = normalize_data_cuda(valid_batch, cfg.min_vals, cfg.max_vals)
                    valid_pred, decouple_loss = model([valid_batch, 0, train_epoch], mode='test')
                    # for channel in range(3):
                    #     valid_pred[:, :, channel] = (valid_pred[:, :, channel] * (cfg.max_vals[channel] - cfg.min_vals[channel]) +
                    #                 cfg.min_vals[channel])
                    loss = criterion(valid_batch[:, IN_LEN:, :3], valid_pred, decouple_loss)
                    loss = reduce_tensor(loss)  # all reduce
                    valid_loss += loss.item()
            valid_loss = valid_loss / len(valid_loader)
            if is_master_proc():
                writer.add_scalars("loss", {"train": train_loss, "valid": valid_loss}, epoch)  # plot loss
                torch.save(model.state_dict(), f'{model_save_path}/features_{cfg.features_amount}_epoch_{epoch}.pth')
                # print(f'Saving model to {model_save_path}/epoch_{epoch}.pth')
            train_loss = 0.0
            # early stopping
            if cfg.early_stopping:
                early_stopping(valid_loss, model, is_master_proc())
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break

    # test
    eval_.clear_all()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = normalize_data_cuda(test_batch, cfg.min_vals, cfg.max_vals)
            test_pred, decouple_loss = model([test_batch, 0, train_epoch], mode='test')
            # for channel in range(3):
            #     test_pred[:, :, channel] = (test_pred[:, :, channel] * (cfg.max_vals[channel] - cfg.min_vals[channel]) +
            #                                  cfg.min_vals[channel])
            loss = criterion(test_batch[:, IN_LEN:, :3], test_pred, decouple_loss)
            test_loss += loss.item()
            test_batch_numpy = test_batch.cpu().numpy()
            test_pred_numpy = test_pred.cpu().numpy()
            eval_.update(test_batch_numpy[:, IN_LEN:, :3], test_pred_numpy)

    if is_master_proc():
        test_metrics_lis = eval_.get_metrics()
        test_loss = test_loss / len(test_loader)
        test_metrics_lis.append(test_loss)
        end = time.time()
        running_time = np.around((end - start) / 3600, decimals=decimals)
        print("===============================")
        print('Running time: {} hours'.format(running_time))
        print("===============================")
        print(f'Test loss: {test_loss}')
        print(f'Test SSIM: {np.sum(test_metrics_lis[0], axis=0)/cfg.out_len}')
        eval_.clear_all()

    if is_master_proc():
        writer.close()
        torch.distributed.destroy_process_group()


def test(model, criterion, test_loader, train_epoch, cfg, train_len):
    if not is_master_proc():
        return
    test_loss = 0.0
    eval_ = Evaluation(seq_len=OUT_LEN, use_central=False)
    eval_.clear_all()
    model.eval()
    mask = load_mask(cfg.root_path)
    # ssim = 0.0
    with torch.no_grad():
        # batch_count = 0
        for idx, test_batch in enumerate(test_loader):
            test_batch = normalize_data_cuda(test_batch, cfg.min_vals, cfg.max_vals)
            test_pred, decouple_loss = model([test_batch, 0, train_epoch], mode='test')
            # for channel in range(3):
            #     test_pred[:, :, channel] = (test_pred[:, :, channel] * (cfg.max_vals[channel] - cfg.min_vals[channel]) +
            #                                  cfg.min_vals[channel])
            loss = criterion(test_batch[:, IN_LEN:, :3], test_pred, decouple_loss)
            test_loss += loss.item()
            test_batch_numpy = test_batch.cpu().numpy()
            test_pred_numpy = test_pred.cpu().numpy()
            eval_.update(test_batch_numpy[:, IN_LEN:, :3], test_pred_numpy)

            if is_master_proc() and idx == 0:
                print('Plotting', flush=True)
                # if os.path.exists(cfg.root_path + f'videos/Forecast/{cfg.model_name}'):
                #     shutil.rmtree(cfg.root_path + f'videos/Forecast/{cfg.model_name}')
                # print(test_batch_numpy.shape) (12, 8, 3, 81, 91)
                for batch_day in range(test_batch.shape[0]):
                    day = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=train_len + batch_day)
                    real_values = deepcopy(test_batch_numpy[:, IN_LEN:, :3].reshape(-1, cfg.out_len, test_pred_numpy.shape[2],
                                                                         test_pred_numpy.shape[3], test_pred_numpy.shape[4]))

                    predictions = deepcopy(test_pred_numpy.reshape(-1, cfg.out_len, test_pred_numpy.shape[2],
                                                                         test_pred_numpy.shape[3], test_pred_numpy.shape[4]))
                    for channel in range(3):
                        max_val = cfg.max_vals[channel].numpy()
                        min_val = cfg.min_vals[channel].numpy()
                        real_values[:, :, channel] = real_values[:, :, channel] * (max_val - min_val) + min_val
                        predictions[:, :, channel] = predictions[:, :, channel] * (max_val - min_val) + min_val
                    plot_predictions(cfg.root_path, real_values[batch_day], predictions[batch_day], cfg.model_name,
                                     cfg.features_amount, day, mask, cfg)
    # print(f'SSIM test full = {ssim/batch_count}')
    # print(ssim.shape)
    # print(f'SSIM test Flux = {np.sum(ssim[:, :, 0])/ batch_count /cfg.batch / OUT_LEN :.2f}')
    # print(f'SSIM test SST = {np.sum(ssim[:, :, 1]) / batch_count /cfg.batch / OUT_LEN :.2f}')
    # print(f'SSIM test Pressure = {np.sum(ssim[:, :, 2]) / batch_count /cfg.batch / OUT_LEN: .2f}')
    if is_master_proc():
        test_metrics_lis = eval_.get_metrics()
        test_loss = test_loss / len(test_loader)
        test_metrics_lis.append(test_loss)
        print("===============================")
        print(f'Test loss: {test_loss}')
        print(f'Test MSE by points: {np.sum(test_metrics_lis[1])/np.sum(mask)/cfg.batch/3/cfg.out_len:.5f}')
        print(f'Test MAE by points: {np.sum(test_metrics_lis[2]) /np.sum(mask)/cfg.batch/3/cfg.out_len:.5f}')
        print(f'Test SSIM: {np.mean(test_metrics_lis[0], axis=0)}')
        print(f'Test CSI: {test_metrics_lis[3]}')
        eval_.clear_all()
        torch.distributed.destroy_process_group()
    return
