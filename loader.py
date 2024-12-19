from torch.utils.data import Dataset
import os
# import cv2
import sys
sys.path.append("..")
from config import cfg
import numpy as np
from datetime import datetime
import torch
from struct import unpack
from torch.utils.data import TensorDataset, DataLoader


def scale_to_bins(arr, bins=100):
    quantiles = list(np.nanquantile(arr, np.linspace(0, 1, bins, endpoint=False)))

    arr_scaled = np.zeros_like(arr, dtype=float)
    # arr_scaled[np.isnan(arr)] = 0
    # for j in tqdm.tqdm(range(bins - 1)):
    for j in range(bins - 1):
        arr_scaled[np.where((np.logical_not(np.isnan(arr))) & (quantiles[j] <= arr) & (arr < quantiles[j + 1]))] = \
            (j + 1) / bins
            # (quantiles[j] + quantiles[j + 1]) / 2

    quantiles += [np.nanmax(arr)]
    return arr_scaled, quantiles


def load_np_data(files_path_prefix, start_year, end_year):
    # load data
    flux_array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    # flux_array = np.diff(flux_array, axis=1)

    SST_array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    # SST_array = np.diff(SST_array, axis=1)

    press_array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    # press_array = np.diff(press_array, axis=1)

    flux_array = flux_array.reshape((161, 181, -1))
    flux_array = flux_array[::2, ::2, :]
    SST_array = SST_array.reshape((161, 181, -1))
    SST_array = SST_array[::2, ::2, :]
    press_array = press_array.reshape((161, 181, -1))
    press_array = press_array[::2, ::2, :]

    return flux_array, SST_array, press_array


def load_mask(files_path_prefix):
    # Mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)
    mask = mask.reshape((161, 181))[::2, ::2]
    return mask


def count_offset(start_year):
    # --------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime(1989, 1, 1, 0, 0) - datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime(1999, 1, 1, 0, 0) - datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime(2009, 1, 1, 0, 0) - datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime(2019, 1, 1, 0, 0) - datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime(2024, 1, 1, 0, 0) - datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime(2024, 4, 28, 0, 0) - datetime(2019, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------

    if start_year == 1979:
        offset = 0
    elif start_year == 1989:
        offset = days_delta1
    elif start_year == 1999:
        offset = days_delta1 + days_delta2
    elif start_year == 2009:
        offset = days_delta1 + days_delta2 + days_delta3
    else:
        offset = days_delta1 + days_delta2 + days_delta3 + days_delta4

    if start_year == 2019:
        end_year = 2025
    else:
        end_year = start_year + 10
    return end_year, offset


def create_torch_data(files_path_prefix, start_year, end_year, cfg):
    flux_array, SST_array, press_array = load_np_data(files_path_prefix, start_year, end_year)
    flux_array_scaled, flux_quantiles = scale_to_bins(flux_array, bins=cfg.bins)
    SST_array_scaled, SST_quantiles = scale_to_bins(SST_array, bins=cfg.bins)
    press_array_scaled, press_quantiles = scale_to_bins(press_array, bins=cfg.bins)
    if start_year == 2019:
        train_len = int(flux_array.shape[2] * 4 / 5)
        test_len = flux_array.shape[2] - train_len - cfg.in_len - cfg.out_len
    else:
        train_len = int(flux_array.shape[2]) - cfg.in_len - cfg.out_len - cfg.batch * 2
        test_len = cfg.batch * 2
    _,  offset = count_offset(start_year)
    # mask = load_mask(files_path_prefix)
    if cfg.postfix_short == '_short':
        train_len = 100
        test_len = 10
    x_train = np.zeros((train_len, cfg.height, cfg.width, cfg.in_len, cfg.features_amount), dtype=float)
    y_train = np.zeros((train_len, cfg.height, cfg.width, cfg.out_len, cfg.features_amount), dtype=float)
    labels_train = np.zeros((train_len, cfg.height, cfg.width, cfg.out_len, cfg.features_amount), dtype=float)

    # eigenvectors_flux_sst = np.zeros((cfg.height, cfg.width, cfg.in_len))
    # eigenvectors_sst_press = np.zeros((cfg.height, cfg.width, cfg.in_len))
    # eigenvectors_flux_press = np.zeros((cfg.height, cfg.width, cfg.in_len))

    eigenvectors_flux_flux = np.zeros((cfg.height, cfg.width, cfg.in_len))
    eigenvectors_sst_sst = np.zeros((cfg.height, cfg.width, cfg.in_len))
    eigenvectors_press_press = np.zeros((cfg.height, cfg.width, cfg.in_len))

    a_flux = np.zeros((cfg.height, cfg.width, cfg.in_len))
    a_sst = np.zeros((cfg.height, cfg.width, cfg.in_len))
    a_press = np.zeros((cfg.height, cfg.width, cfg.in_len))

    print('Preparing train', flush=True)
    for t in range(train_len):
        # flux
        x_train[t, :, :, :, 0] = flux_array[:, :, t:t+cfg.in_len]
        y_train[t, :, :, :, 0] = flux_array[:, :, t + cfg.in_len:t + cfg.in_len + cfg.out_len]
        labels_train[t, :, :, :, 0] = flux_array_scaled[:, :, t + cfg.in_len:t + cfg.in_len + cfg.out_len]

        # sst
        x_train[t, :, :, :, 1] = SST_array[:, :, t:t+cfg.in_len]
        y_train[t, :, :, :, 1] = SST_array[:, :, t + cfg.in_len:t + cfg.in_len + cfg.out_len]
        labels_train[t, :, :, :, 1] = SST_array_scaled[:, :, t + cfg.in_len:t + cfg.in_len + cfg.out_len]

        # press
        x_train[t, :, :, :, 2] = press_array[:, :, t:t+cfg.in_len]
        y_train[t, :, :, :, 2] = press_array[:, :, t + cfg.in_len:t + cfg.in_len + cfg.out_len]
        labels_train[t, :, :, :, 2] = press_array_scaled[:, :, t + cfg.in_len:t + cfg.in_len + cfg.out_len]

        if cfg.features_amount >= 6:
            for t_lag in range(cfg.in_len):
            #     # flux - sst
            #     try:
            #         eigenvectors_flux_sst[:, :, t_lag] = np.load(
            #             files_path_prefix + f'Eigenvalues/Flux-SST/eigen0_{t + offset + t_lag}.npy').reshape(
            #             (161, 181))[::2, ::2]
            #     except FileNotFoundError:
            #         print(f'Not existing Eigenvalues/Flux-SST/eigen0_{t + offset + t_lag}.npy')
            #         eigenvectors_flux_sst[:, :, t_lag] = np.zeros((cfg.height, cfg.width))
            #
            #     # sst - press
            #     try:
            #         eigenvectors_sst_press[:, :, t_lag] = np.load(
            #             files_path_prefix + f'Eigenvalues/SST-Pressure/eigen0_{t + offset + t_lag}.npy').reshape(
            #             (161, 181))[::2, ::2]
            #     except FileNotFoundError:
            #         print(f'Not existing Eigenvalues/SST-Pressure/eigen0_{t + offset + t_lag}.npy')
            #         eigenvectors_sst_press[:, :, t_lag] = np.zeros((cfg.height, cfg.width))
            #
            #     # flux - press
            #     try:
            #         eigenvectors_flux_press[:, :, t_lag] = np.load(
            #             files_path_prefix + f'Eigenvalues/Flux-Pressure/eigen0_{t + offset + t_lag}.npy').reshape(
            #             (161, 181))[::2, ::2]
            #     except FileNotFoundError:
            #         print(f'Not existing Eigenvalues/Flux-Pressure/eigen0_{t + offset + t_lag}.npy')
            #         eigenvectors_flux_press[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # flux - flux
                try:
                    eigenvectors_flux_flux[:, :, t_lag] = np.load(
                        files_path_prefix + f'Eigenvalues/Flux-Flux/eigen0_{t + offset + t_lag}.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Eigenvalues/Flux-Flux/eigen0_{t + offset + t_lag}.npy')
                    eigenvectors_flux_flux[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # sst - sst
                try:
                    eigenvectors_sst_sst[:, :, t_lag] = np.load(
                        files_path_prefix + f'Eigenvalues/SST-SST/eigen0_{t + offset + t_lag}.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Eigenvalues/SST-SST/eigen0_{t + offset + t_lag}.npy')
                    eigenvectors_sst_sst[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # press - press
                try:
                    eigenvectors_press_press[:, :, t_lag] = np.load(
                        files_path_prefix + f'Eigenvalues/Pressure-Pressure/eigen0_{t + offset + t_lag}.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Eigenvalues/Pressure-Pressure/eigen0_{t + offset + t_lag}.npy')
                    eigenvectors_press_press[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

            x_train[t, :, :, :, 3] = eigenvectors_flux_flux
            x_train[t, :, :, :, 4] = eigenvectors_sst_sst
            x_train[t, :, :, :, 5] = eigenvectors_press_press

        if cfg.features_amount >= 9:
            for t_lag in range(cfg.in_len):
                # a flux
                try:
                    a_flux[:, :, t_lag] = np.load(files_path_prefix + f'Coeff_data_3d/flux-sst/{t + offset + t_lag}_A_sens.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Coeff_data_3d/flux-sst/{t + offset + t_lag}_A_sens.npy')
                    a_flux[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # a sst
                try:
                    a_sst[:, :, t_lag] = np.load(files_path_prefix + f'Coeff_data_3d/flux-sst/{t + offset + t_lag}_A_lat.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Coeff_data_3d/flux-sst/{t + offset + t_lag}_A_lat.npy')
                    a_sst[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # a press
                try:
                    a_press[:, :, t_lag] = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{t + offset + t_lag}_A_lat.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Coeff_data_3d/flux-press/{t + offset + t_lag}_A_lat.npy')
                    a_press[:, :, t_lag] = np.zeros((cfg.height, cfg.width))
            x_train[t, :, :, :, 6] = a_flux
            x_train[t, :, :, :, 7] = a_sst
            x_train[t, :, :, :, 8] = a_press

    np.nan_to_num(x_train, copy=False)
    np.nan_to_num(y_train, copy=False)
    x_train = torch.from_numpy(x_train)
    torch.save(x_train, files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{cfg.features_amount}{cfg.postfix_short}.pt')
    del x_train
    y_train = torch.from_numpy(y_train)
    torch.save(y_train, files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{cfg.features_amount}{cfg.postfix_short}.pt')
    del y_train
    labels_train = torch.from_numpy(labels_train)
    torch.save(labels_train, files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_labels_train_{cfg.features_amount}{cfg.postfix_short}.pt')
    del labels_train

    # ---------------------------------------------------------------------------------------------------------------
    print('Preparing test', flush=True)
    x_test = np.zeros((test_len, cfg.height, cfg.width, cfg.in_len, cfg.features_amount), dtype=float)
    y_test = np.zeros((test_len, cfg.height, cfg.width, cfg.out_len, cfg.features_amount), dtype=float)
    labels_test = np.zeros((test_len, cfg.height, cfg.width, cfg.out_len, cfg.features_amount), dtype=float)
    for t in range(test_len):
        t_absolute = train_len + t
        # flux
        x_test[t, :, :, :, 0] = flux_array[:, :, t_absolute:t_absolute+cfg.in_len]
        y_test[t, :, :, :, 0] = flux_array[:, :, t_absolute + cfg.in_len:t_absolute + cfg.in_len + cfg.out_len]
        labels_test[t, :, :, :, 0] = flux_array_scaled[:, :, t_absolute + cfg.in_len:t_absolute + cfg.in_len + cfg.out_len]

        # sst
        x_test[t, :, :, :, 1] = SST_array[:, :, t_absolute:t_absolute+cfg.in_len]
        y_test[t, :, :, :, 1] = SST_array[:, :, t_absolute + cfg.in_len:t_absolute + cfg.in_len + cfg.out_len]
        labels_test[t, :, :, :, 1] = SST_array_scaled[:, :, t_absolute + cfg.in_len:t_absolute + cfg.in_len + cfg.out_len]

        # press
        x_test[t, :, :, :, 2] = press_array[:, :, t_absolute:t_absolute+cfg.in_len]
        y_test[t, :, :, :, 2] = press_array[:, :, t_absolute + cfg.in_len:t_absolute + cfg.in_len + cfg.out_len]
        labels_test[t, :, :, :, 2] = press_array_scaled[:, :, t_absolute + cfg.in_len:t_absolute + cfg.in_len + cfg.out_len]

        if cfg.features_amount >= 6:
            for t_lag in range(cfg.in_len):
                # flux - sst
                # try:
                #     eigenvectors_flux_sst[:, :, t_lag] = np.load(
                #         files_path_prefix + f'Eigenvalues/Flux-SST/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
                #         (161, 181))[::2, ::2]
                # except FileNotFoundError:
                #     print(f'Not existing Eigenvalues/Flux-SST/eigen0_{t_absolute + offset + t_lag}.npy')
                #     eigenvectors_flux_sst[:, :, t_lag] = np.zeros((cfg.height, cfg.width))
                #
                # # sst - press
                # try:
                #     eigenvectors_sst_press[:, :, t_lag] = np.load(
                #         files_path_prefix + f'Eigenvalues/SST-Pressure/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
                #         (161, 181))[::2, ::2]
                # except FileNotFoundError:
                #     print(f'Not existing Eigenvalues/SST-Pressure/eigen0_{t_absolute + offset + t_lag}.npy')
                #     eigenvectors_sst_press[:, :, t_lag] = np.zeros((cfg.height, cfg.width))
                #
                # # flux - press
                # try:
                #     eigenvectors_flux_press[:, :, t_lag] = np.load(
                #         files_path_prefix + f'Eigenvalues/Flux-Pressure/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
                #         (161, 181))[::2, ::2]
                # except FileNotFoundError:
                #     print(f'Not existing Eigenvalues/Flux-Pressure/eigen0_{t_absolute + offset + t_lag}.npy')
                #     eigenvectors_flux_press[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # flux - flux
                try:
                    eigenvectors_flux_flux[:, :, t_lag] = np.load(
                        files_path_prefix + f'Eigenvalues/Flux-Flux/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Eigenvalues/Flux-Flux/eigen0_{t_absolute + offset + t_lag}.npy')
                    eigenvectors_flux_flux[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # sst - sst
                try:
                    eigenvectors_sst_sst[:, :, t_lag] = np.load(
                        files_path_prefix + f'Eigenvalues/SST-SST/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Eigenvalues/SST-SST/eigen0_{t_absolute + offset + t_lag}.npy')
                    eigenvectors_sst_sst[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # press - press
                try:
                    eigenvectors_press_press[:, :, t_lag] = np.load(
                        files_path_prefix + f'Eigenvalues/Pressure-Pressure/eigen0_{t_absolute + offset + t_lag}.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Eigenvalues/Pressure-Pressure/eigen0_{t_absolute + offset + t_lag}.npy')
                    eigenvectors_press_press[:, :, t_lag] = np.zeros((cfg.height, cfg.width))


            # x_test[t, :, :, :, 3] = eigenvectors_flux_sst
            # x_test[t, :, :, :, 4] = eigenvectors_sst_press
            # x_test[t, :, :, :, 5] = eigenvectors_flux_press

            x_test[t, :, :, :, 3] = eigenvectors_flux_flux
            x_test[t, :, :, :, 4] = eigenvectors_sst_sst
            x_test[t, :, :, :, 5] = eigenvectors_press_press

        if cfg.features_amount >= 9:
            for t_lag in range(cfg.in_len):
                # a flux
                try:
                    a_flux[:, :, t_lag] = np.load(files_path_prefix + f'Coeff_data_3d/flux-sst/{t_absolute + offset + t_lag}_A_sens.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Coeff_data_3d/flux-sst/{t_absolute + offset + t_lag}_A_sens.npy')
                    a_flux[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # a sst
                try:
                    a_sst[:, :, t_lag] = np.load(files_path_prefix + f'Coeff_data_3d/flux-sst/{t_absolute + offset + t_lag}_A_lat.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Coeff_data_3d/flux-sst/{t_absolute + offset + t_lag}_A_lat.npy')
                    a_sst[:, :, t_lag] = np.zeros((cfg.height, cfg.width))

                # a press
                try:
                    a_press[:, :, t_lag] = np.load(files_path_prefix + f'Coeff_data_3d/flux-press/{t_absolute + offset + t_lag}_A_lat.npy').reshape(
                        (161, 181))[::2, ::2]
                except FileNotFoundError:
                    print(f'Not existing Coeff_data_3d/flux-press/{t_absolute + offset + t_lag}_A_lat.npy')
                    a_press[:, :, t_lag] = np.zeros((cfg.height, cfg.width))
            x_test[t, :, :, :, 6] = a_flux
            x_test[t, :, :, :, 7] = a_sst
            x_test[t, :, :, :, 8] = a_press

    np.nan_to_num(x_test, copy=False)
    np.nan_to_num(y_test, copy=False)
    x_test = torch.from_numpy(x_test)
    torch.save(x_test, files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{cfg.features_amount}{cfg.postfix_short}.pt')
    del x_test
    y_test = torch.from_numpy(y_test)
    torch.save(y_test, files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{cfg.features_amount}{cfg.postfix_short}.pt')
    del y_test
    labels_test = torch.from_numpy(labels_test)
    torch.save(labels_test, files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_labels_test_{cfg.features_amount}{cfg.postfix_short}.pt')
    del labels_test
    return


def create_dataloaders(files_path_prefix, start_year, end_year, cfg):
    if not os.path.exists(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{cfg.features_amount}{cfg.postfix_short}.pt'):
        create_torch_data(files_path_prefix, start_year, end_year, cfg)
    x_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{cfg.features_amount}{cfg.postfix_short}.pt')
    y_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{cfg.features_amount}{cfg.postfix_short}.pt')
    labels_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_labels_train_{cfg.features_amount}{cfg.postfix_short}.pt')
    # print(x_train.shape)
    #print(y_train.shape)
    x_test = torch.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{cfg.features_amount}{cfg.postfix_short}.pt')
    y_test = torch.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{cfg.features_amount}{cfg.postfix_short}.pt')
    labels_test = torch.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_labels_test_{cfg.features_amount}{cfg.postfix_short}.pt')

    cfg.min_vals = tuple(torch.amin(x_train, dim=(0, 1, 2, 3)))
    cfg.max_vals = tuple(torch.amax(x_train, dim=(0, 1, 2, 3)))

    # train_dataset = Data(x_train, y_train)
    # test_dataset = Data(x_test, y_test)
    train_dataset = Data_Labelled(x_train, y_train, labels_train)
    test_dataset = Data_Labelled(x_test, y_test, labels_test)

    return train_dataset, test_dataset, test_dataset


def create_dataloader_encoder_decoder(files_path_prefix, start_year, end_year, cfg):
    if not os.path.exists(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{cfg.features_amount}{cfg.postfix_short}.pt'):
        create_torch_data(files_path_prefix, start_year, end_year, cfg)
    x_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{cfg.features_amount}{cfg.postfix_short}.pt')
    # print(x_train.shape)
    cfg.min_vals = tuple(torch.amin(x_train, dim=(0, 1, 2, 3)))
    cfg.max_vals = tuple(torch.amax(x_train, dim=(0, 1, 2, 3)))
    train_dataset = Data(x_train, x_train)
    return train_dataset


class Data(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.data = torch.cat((x, y), dim=3).permute((3, 0, 4, 1, 2)).float()

    def __getitem__(self, index):
        sample = self.data[:, index, :, :, :]
        return sample  # S*C*H*W

    def __len__(self):
        return self.data.shape[1]


class Data_Labelled(Dataset):
    def __init__(self, x, y, labels):
        super().__init__()
        self.data = torch.cat((x, y, labels), dim=3).permute((3, 0, 4, 1, 2)).float()

    def __getitem__(self, index):
        sample = self.data[:, index, :, :, :]
        return sample  # S*C*H*W

    def __len__(self):
        return self.data.shape[1]


class Data2(Dataset):
    def __init__(self, cfg, start_idx, end_idx):
        super().__init__()
        self.features_amount = cfg.features_amount
        self.in_len = cfg.in_len
        self.out_len = cfg.out_len
        self.height = cfg.height
        self.width = cfg.width
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.flux_array = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_grouped_diff.npy')[start_idx:end_idx]
        self.sst_array = np.load(cfg.root_path + f'DATA/SST_1979-2025_grouped_diff.npy')[start_idx:end_idx]
        self.press_array = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_grouped_diff.npy')[start_idx:end_idx]
        np.nan_to_num(self.flux_array, copy=False)
        np.nan_to_num(self.sst_array, copy=False)
        np.nan_to_num(self.press_array, copy=False)

        self.flux_quantiles = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_grouped_diff_scaled.npy')[start_idx:end_idx]
        self.sst_quantiles = np.load(cfg.root_path + f'DATA/SST_1979-2025_grouped_diff_scaled.npy')[start_idx:end_idx]
        self.press_quantiles = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_grouped_diff_scaled.npy')[start_idx:end_idx]

        self.eigen_flux = None
        self.eigen_sst = None
        self.eigen_press = None

        self.A_flux = None
        self.A_sst = None
        self.A_press = None

        if cfg.features_amount == 6 or cfg.features_amount == 9:
            self.eigen_flux = np.load(cfg.root_path + f'DATA/FLUX_FLUX_1979-2025_eigen0.npy')[start_idx:end_idx]
            self.eigen_sst = np.load(cfg.root_path + f'DATA/SST_SST_1979-2025_eigen0.npy')[start_idx:end_idx]
            self.eigen_press = np.load(cfg.root_path + f'DATA/PRESS_PRESS_1979-2025_eigen0.npy')[start_idx:end_idx]

            np.nan_to_num(self.eigen_flux, copy=False)
            np.nan_to_num(self.eigen_sst, copy=False)
            np.nan_to_num(self.eigen_press, copy=False)

        if cfg.features_amount == 9:
            self.A_flux = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_A.npy')[start_idx:end_idx]
            self.A_sst = np.load(cfg.root_path + f'DATA/SST_1979-2025_A.npy')[start_idx:end_idx]
            self.A_press = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_A.npy')[start_idx:end_idx]

            np.nan_to_num(self.A_flux, copy=False)
            np.nan_to_num(self.A_sst, copy=False)
            np.nan_to_num(self.A_press, copy=False)

    def __getitem__(self, index):
        sample = np.zeros((self.in_len + self.out_len + self.out_len, self.features_amount, self.height, self.width), dtype=float)
        for day in range(self.in_len + self.out_len):
            sample[day, 0] = self.flux_array[index + day]
            sample[day, 1] = self.sst_array[index + day]
            sample[day, 2] = self.press_array[index + day]
        for day in range(self.out_len):
            sample[self.in_len + self.out_len + day, 0] = self.flux_quantiles[index + day]
            sample[self.in_len + self.out_len + day, 1] = self.sst_quantiles[index + day]
            sample[self.in_len + self.out_len + day, 2] = self.press_quantiles[index + day]

        if self.features_amount == 6:
            for day in range(self.in_len + self.out_len):
                sample[day, 3] = self.eigen_flux[index + day]
                sample[day, 4] = self.eigen_sst[index + day]
                sample[day, 5] = self.eigen_press[index + day]
        elif self.features_amount == 9:
            for day in range(self.in_len + self.out_len):
                sample[day, 3] = self.eigen_flux[index + day]
                sample[day, 4] = self.eigen_sst[index + day]
                sample[day, 5] = self.eigen_press[index + day]

                sample[day, 6] = self.A_flux[index + day]
                sample[day, 7] = self.A_sst[index + day]
                sample[day, 8] = self.A_press[index + day]

        return torch.from_numpy(sample).float()  # S*C*H*W

    def __len__(self):
        return self.flux_array.shape[0] - self.in_len - self.out_len