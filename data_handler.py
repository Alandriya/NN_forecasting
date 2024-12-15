import os.path

import numpy as np
from loader import count_offset, scale_to_bins
from config import cfg
import datetime
from struct import unpack
files_path_prefix = '/home/aosipova/EM_ocean/'

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    # Mask
    maskfile = open(files_path_prefix + "mask", "rb")
    binary_values = maskfile.read(29141)
    maskfile.close()
    mask = unpack('?' * 29141, binary_values)
    mask = np.array(mask, dtype=int)

    # ---------------------------------------------------------------------------------------
    # Days deltas
    days_delta1 = (datetime.datetime(1989, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(1999, 1, 1, 0, 0) - datetime.datetime(1989, 1, 1, 0, 0)).days
    days_delta3 = (datetime.datetime(2009, 1, 1, 0, 0) - datetime.datetime(1999, 1, 1, 0, 0)).days
    days_delta4 = (datetime.datetime(2019, 1, 1, 0, 0) - datetime.datetime(2009, 1, 1, 0, 0)).days
    days_delta5 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta6 = (datetime.datetime(2024, 4, 28, 0, 0) - datetime.datetime(2019, 1, 1, 0, 0)).days
    days_delta7 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # ----------------------------------------------------------------------------------------------
    days_delta8 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    variable = 'FLUX'

    # full_array = np.zeros((days_delta8, cfg.height, cfg.width), dtype=float)
    # for start_year in [1979, 1989, 1999, 2009, 2019]:
    #     end_year, offset = count_offset(start_year)
    #     if variable == 'FLUX':
    #         array = np.load(files_path_prefix + f'Fluxes/FLUX_{start_year}-{end_year}_grouped.npy')
    #     elif variable == 'SST':
    #         array = np.load(files_path_prefix + f'SST/SST_{start_year}-{end_year}_grouped.npy')
    #     else:
    #         array = np.load(files_path_prefix + f'Pressure/PRESS_{start_year}-{end_year}_grouped.npy')
    #
    #     array = array.reshape((161, 181, -1))
    #     array = array[::2, ::2, :]
    #     array = np.swapaxes(array, 0, 2)
    #     array = np.swapaxes(array, 1, 2)
    #     print(array.shape, flush=True)
    #     if start_year == 2019:
    #         full_array[offset:] = array
    #     else:
    #         end_year2, offset2 = count_offset(start_year + 10)
    #         full_array[offset:offset2] = array
    #
    # if not os.path.exists(files_path_prefix + f'DATA'):
    #     os.mkdir(files_path_prefix + f'DATA')
    #
    # np.save(files_path_prefix + f'DATA/{variable}_1979-2025_grouped.npy', full_array)
    # print('Beginning scaling')
    # array_scaled, quantiles = scale_to_bins(full_array, bins=cfg.bins)
    # np.save(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_scaled.npy', array_scaled)
    # np.save(files_path_prefix + f'DATA/{variable}_1979-2025_quantiles.npy', quantiles)
    # print('Ended')
    array_scaled = np.load(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_scaled.npy')
    print(np.isnan(array_scaled).any())