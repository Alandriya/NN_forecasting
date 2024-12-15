from struct import unpack
import datetime
import numpy as np
import tqdm
from sklearn.metrics import mean_absolute_percentage_error
from plotter import plot_predictions
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from skimage.metrics import structural_similarity as ssim
from SSIM import get_SSIM
# import tensorflow as tf
import sys
import torch
# SHORT_POSTFIX = '_short'
SHORT_POSTFIX = ''
files_path_prefix = 'E:/Nastya/Data/OceanFull/'
from config import cfg
import copy


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    # --------------------------------------------------------------------------------
    # Mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    mask = mask.reshape((161, 181))[::2, ::2]
    print(np.sum(mask))
    # print(cfg.batch)
    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    start_year = 2019
    if start_year == 2019:
        end_year = 2025
    else:
        end_year = start_year + 10

    offset = days_delta1 + days_delta2 + days_delta3 + days_delta4
    # ---------------------------------------------------------------------------------------
    # configs
    width = 91
    height = 81
    batch_size = cfg.batch
    days_known = 7
    days_prediction = 5
    features_amount = 6
    # ---------------------------------------------------------------------------------------
    x_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{features_amount}{SHORT_POSTFIX}.pt')
    y_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{features_amount}{SHORT_POSTFIX}.pt')

    x_test = torch.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{features_amount}{SHORT_POSTFIX}.pt')
    y_test = torch.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{features_amount}{SHORT_POSTFIX}.pt')

    min_vals = torch.amin(y_train, dim=(0, 1, 2, 3)).numpy()
    max_vals = torch.amax(y_train, dim=(0, 1, 2, 3)).numpy()
    # print(min_vals)
    # print(max_vals)

    x_train = torch.permute(x_train, dims=(0, 3, 4, 1, 2)).numpy()
    y_train = torch.permute(y_train, dims=(0, 3, 4, 1, 2)).numpy()
    x_test = torch.permute(x_test, dims=(0, 3, 4, 1, 2)).numpy()
    y_test = torch.permute(y_test, dims=(0, 3, 4, 1, 2)).numpy()
    y_test_copy = copy.deepcopy(y_test)

    for k in range(3):
        x_train[:, :, k] = (x_train[:, :, k] - min_vals[k]) / (max_vals[k] - min_vals[k])
        y_train[:, :, k] = (y_train[:, :, k] - min_vals[k]) / (max_vals[k] - min_vals[k])
        x_test[:, :, k] = (x_test[:, :, k] - min_vals[k]) / (max_vals[k] - min_vals[k])
        y_test[:, :, k] = (y_test[:, :, k] - min_vals[k]) / (max_vals[k] - min_vals[k])

    print(x_train.shape)
    # mse = 0
    # mae = 0
    # ssim_arr = np.zeros((batch_size, days_prediction, 3))
    # # get dumb prediction and plot it
    # for t in range(batch_size):
    #     y_pred = np.zeros((days_prediction, 3, height, width))
    #     y_pred[:, :, np.logical_not(mask)] = 0
    #     for t1 in range(days_prediction):
    #         y_pred[t1] = np.mean(y_train[t], axis=0)[:3]
    #         for k in range(3):
    #             ssim_arr[t, t1, k] = ssim(y_pred[t1, k], y_test[t, t1, k])
    #
    #     start_day = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=x_train.shape[0] + t)
    #     mse += np.sum((y_pred - y_test[t]) ** 2)
    #     mae += np.sum(np.abs(y_pred-y_test[t]))
    #     # print(start_day)
    #     for k in range(3):
    #         y_pred[:, k] = y_pred[:, k] * (max_vals[k] - min_vals[k]) + min_vals[k]
    #     plot_predictions(files_path_prefix, y_test_copy[t], y_pred, 'Dumb', features_amount, start_day, mask, cfg)
    #
    # print(f'Test SSIM DUMB: {np.mean(ssim_arr, axis=(0, 1))}')
    # mse = mse/np.sum(mask)/batch_size/3/cfg.out_len
    # mae = mae/np.sum(mask)/batch_size/3/cfg.out_len
    # print(f'Test MSE DUMB: {mse:.5f}')
    # print(f'Test MAE DUMB: {mae:.5f}')

    ssim_arr = np.zeros((batch_size, days_prediction, 3))
    mse = 0
    mae = 0
    # get regression prediction and plot it
    for t in range(batch_size):
        y_pred = np.zeros((days_prediction, 3, height, width), dtype=float)
        for k in range(3):
            for i in range(height):
                for j in range(width):
                    if mask[i, j]:
                        regr = linear_model.LinearRegression()
                        x_tmp = x_train[:, -days_prediction:, :, i, j].reshape((-1, features_amount))
                        y_tmp = y_train[:, -days_prediction:, k, i, j].flatten()
                        # print(x_tmp.shape)
                        # print(y_tmp.shape)
                        regr.fit(x_tmp, y_tmp)
                        y_pred[:, k, i, j] = regr.predict(x_test[0:1, -days_prediction:, :, i, j].reshape((-1, features_amount)))
        start_day = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=x_train.shape[0] + t)
        mse += np.sum((y_pred - y_test[t][:, :3]) ** 2)
        mae += np.sum(np.abs(y_pred-y_test[t][:, :3]))
        y_pred[:, :, np.logical_not(mask)] = 0
        # for t1 in range(days_prediction):
        #     for k in range(3):
        #         ssim_arr[t, t1, k] = ssim(y_pred[t1, k], y_test[t, t1, k])

        for k in range(3):
            y_pred[:, k] = y_pred[:, k] * (max_vals[k] - min_vals[k]) + min_vals[k]
        plot_predictions(files_path_prefix, y_test_copy[t][:, :3], y_pred, 'Linear regression', features_amount, start_day, mask, cfg)


    # mse = mse/np.sum(mask)/batch_size/3/cfg.out_len
    # mae = mae/np.sum(mask)/batch_size/3/cfg.out_len
    # print(f'Test MSE regression: {mse:.5f}')
    # print(f'Test MAE regression: {mae:.5f}')
    # print(f'Test SSIM regression: {np.mean(ssim_arr, axis=(0, 1))}')