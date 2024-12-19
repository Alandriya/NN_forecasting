import os.path
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from loader import count_offset, scale_to_bins, load_mask
from config import cfg
import datetime
from struct import unpack
from scipy.linalg import sqrtm
import gc
from sklearn.mixture import GaussianMixture
import math

files_path_prefix = '/home/aosipova/EM_ocean/'

# def collect_eigenvalues(files_path_prefix: str,
#                      n_lambdas: int,
#                      mask: np.ndarray,
#                      t_start: int,
#                      t_end: int,
#                      offset: int,
#                      array2,
#                      array2_quantiles,
#                      names: tuple = ('Sensible', 'Latent'),
#                      shape: tuple = (161, 181),
#                      ):
#     print(f'Collecting {names[0]}-{names[1]}', flush=True)
#     for t in range(t_start, t_end):
#         if t % 100 == 0:
#             print(t)
#         try:
#             eigenvalues = np.load(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvalues_{t+offset}.npy')
#             eigenvalues = np.real(eigenvalues)
#             eigenvectors = np.load(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvectors_{t+offset}.npy')
#             eigenvectors = np.real(eigenvectors)
#             print(f'Plot timestep {t + offset}', flush=True)
#         except FileNotFoundError:
#             print(f'No file step {t + offset}', flush=True)
#             continue
#
#         width, height = shape
#         matrix_list = [np.zeros(height * width) for _ in range(n_lambdas)]
#         lambda_list = []
#         max_list = []
#         min_list = []
#
#         n_bins = 100
#         for l in range(n_lambdas):
#             for j1 in range(0, n_bins):
#                 points_y1 = np.where((array2_quantiles[j1] <= array2[t+1]) & (array2[t+1] < array2_quantiles[j1 + 1]))[0]
#                 matrix_list[l][points_y1] = np.real(eigenvectors[j1, l])
#
#             matrix_list[l][np.logical_not(mask)] = np.nan
#             max_list.append(np.nanmax(matrix_list[l]))
#             min_list.append(np.nanmin(matrix_list[l]))
#             lambda_list.append(eigenvalues[l])
#
#         np.save(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigen0_{t+offset}.npy', matrix_list[0])
#     return


# def get_eig(B: np.ndarray,
#             names: tuple):
#     """
#     Counts eigenvalues for the covariances matrix B for two cases: if both the variables in the data arrays are the
#     same, e.g. (Flux, Flux) and for different, e.g. (Flux, SST)
#     :param B: np.array with shape (n_bins, n_bins), two-dimensional
#     :param names: tuple with names of the data, e.g. ('Flux', 'SST'), ('Flux', 'Flux')
#     :return:
#     """
#     if names[0] == names[1]:
#         A = B
#     else:
#         # print('Performing A = B*B^T', flush=True)
#         A = np.dot(B, B.transpose())
#         # print('Getting sqrt(A)', flush=True)
#         A = sqrtm(A)
#
#     gc.collect()
#     # print('Counting eigenvalues', flush=True)
#     eigenvalues, eigenvectors = np.linalg.eig(A)
#     # sort by absolute value of the eigenvalues
#     eigenvalues = np.real(eigenvalues)
#     eigenvalues = [0 if np.isnan(e) else e for e in eigenvalues]
#     positions = [x for x in range(len(eigenvalues))]
#     positions = [x for _, x in reversed(sorted(zip(np.abs(eigenvalues), positions)))]
#     return np.take(eigenvalues, positions), np.take(eigenvectors, positions, axis=1), positions

# def count_eigenvalues_pair(files_path_prefix: str,
#                            array1: np.ndarray,
#                            array2: np.ndarray,
#                            array1_quantiles: list,
#                            array2_quantiles: list,
#                            t: int,
#                            n_bins: int,
#                            offset: int,
#                            names: tuple):
#     """
#
#     :param files_path_prefix: path to the working directory
#     :param array1: array with shape (height*width, n_days): e.g. (29141, 1410)
#     :param array2: array with shape (height*width, n_days): e.g. (29141, 1410)
#     :param array1_quantiles: list with length = n_bins + 1 of the quantiles built by scale_to_bins function
#     :param array2_quantiles: list with length = n_bins + 1 of the quantiles built by scale_to_bins function
#     :param t: relative time moment from the beginning of the array
#     :param n_bins: amount of bins to divide the values of each array
#     :param offset: shift of the beginning of the data arrays in days from 01.01.1979, for 01.01.2019 is 14610
#     :param names: tuple with names of the data arrays, e.g. ('Flux', 'SST')
#     :return:
#     """
#     if not os.path.exists(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}'):
#         os.mkdir(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}')
#
#     if os.path.exists(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy'):
#         return
#
#     b_matrix = np.zeros((n_bins, n_bins))
#     for i1 in range(0, n_bins):
#         points_x1 = np.where((array1_quantiles[i1] <= array1[0]) & (array1[0] < array1_quantiles[i1 + 1]))[0]
#         for j1 in range(0, n_bins):
#             points_y1 = np.where((array2_quantiles[j1] <= array2[0]) & (array2[0] < array2_quantiles[j1 + 1]))[0]
#             if len(points_x1) and len(points_y1):
#                 mean1 = np.mean(array1[0, points_x1])
#                 mean2 = np.mean(array2[0, points_y1])
#                 vec1 = array1[1, points_x1] - mean1
#                 vec2 = array2[1, points_y1] - mean2
#                 b_matrix[i1, j1] = np.sum(np.multiply.outer(vec1, vec2).ravel())
#
#     b_matrix = np.nan_to_num(b_matrix)
#
#     # count eigenvalues
#     eigenvalues, eigenvectors, positions = get_eig(b_matrix, (names[0], names[1]))
#     # np.save(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvalues_{t + offset}.npy', eigenvalues)
#     # np.save(files_path_prefix + f'Eigenvalues-mini/{names[0]}-{names[1]}/eigenvectors_{t + offset}.npy', eigenvectors)
#     return eigenvalues, eigenvectors, positions
#
# def count_eigenvalues_triplets(files_path_prefix: str,
#                                t_start: int,
#                                flux_array: np.ndarray,
#                                SST_array: np.ndarray,
#                                press_array: np.ndarray,
#                                mask: np.ndarray,
#                                offset: int = 14610,
#                                n_bins: int = 100,
#                                ):
#     """
#     Counts and plots eigenvalues and eigenvectors for pairs Flux-Flux, SST-SST, Flux-SST, Flux-Pressure for time range
#     offset + t_start, offset + len(flux_array)
#     :param files_path_prefix: path to the working directory
#     :param t_start: relative offset from the beginning of the array for time cycle
#     :param flux_array: array with shape (height*width, n_days): e.g. (29141, 1410) with flux values
#     :param SST_array: array with shape (height*width, n_days): e.g. (29141, 1410) with SST values
#     :param press_array: array with shape (height*width, n_days): e.g. (29141, 1410) with pressure values
#     :param mask:
#     :param offset: shift of the beginning of the data arrays in days from 01.01.1979, for 01.01.2019 is 14610
#     :param n_bins: amount of bins to divide the values of each array
#     :return:
#     """
#
#     # flux_array_grouped, quantiles_flux = scale_to_bins(flux_array, n_bins)
#     # SST_array_grouped, quantiles_sst = scale_to_bins(SST_array, n_bins)
#     # press_array_grouped, quantiles_press = scale_to_bins(press_array, n_bins)
#
#     quantiles_flux = np.load(cfg.root_path + f'DATA/FLUX_1979-2025_quantiles.npy')
#     quantiles_sst = np.load(cfg.root_path + f'DATA/SST_1979-2025_quantiles.npy')
#     quantiles_press = np.load(cfg.root_path + f'DATA/PRESS_1979-2025_quantiles.npy')
#
#     if not os.path.exists(files_path_prefix + f'Eigenvalues-mini'):
#         os.mkdir(files_path_prefix + f'Eigenvalues-mini')
#
#     def count_pair(pair_name, array1, array2, quantiles1, quantiles2):
#         print(f'Pair {pair_name}')
#         n_lambdas = 3
#         n_bins = 100
#         for t in range(t_start, flux_array.shape[0] - 1):
#             if t % 100 == 0:
#                 print(f'Timestep {t}', flush=True)
#             eigenvalues, eigenvectors, positions = count_eigenvalues_pair(files_path_prefix, array1[t:t + 2],
#                                                                           array2[t:t + 2], quantiles1,
#                                                                           quantiles2, t, n_bins,
#                                                                           offset, pair_name)
#             eigenvalues = np.real(eigenvalues)
#             eigenvectors = np.real(eigenvectors)
#             print(array1.shape)
#             width, height = array1.shape[2], array1.shape[1]
#             matrix_list = [np.zeros((height, width)) for _ in range(n_lambdas)]
#             lambda_list = []
#             max_list = []
#             min_list = []
#
#             n_bins = 100
#             for l in range(n_lambdas):
#                 for j1 in range(0, n_bins):
#                     points_y1 = \
#                     np.where((quantiles2[j1] <= array2[t + 1]) & (array2[t + 1] < quantiles2[j1 + 1]))[0]
#                     matrix_list[l][points_y1] = np.real(eigenvectors[j1, l])
#
#                 matrix_list[l][np.logical_not(mask)] = np.nan
#                 max_list.append(np.nanmax(matrix_list[l]))
#                 min_list.append(np.nanmin(matrix_list[l]))
#                 lambda_list.append(eigenvalues[l])
#
#             np.save(files_path_prefix + f'Eigenvalues-mini/{pair_name[0]}-{pair_name[1]}/eigen0_{t + offset}.npy',
#                     matrix_list[0])
#         return
#
#     count_pair(('Flux', 'Flux'), flux_array, flux_array, quantiles_flux, quantiles_flux)
#     count_pair(('SST', 'SST'), sst_array, sst_array, quantiles_sst, quantiles_sst)
#     count_pair(('Pressure', 'Pressure'), press_array, press_array, quantiles_press, quantiles_press)
#
#     count_pair(('Flux', 'SST'), flux_array, sst_array, quantiles_flux, quantiles_sst)
#     count_pair(('Flux', 'Pressure'), flux_array, press_array, quantiles_flux, quantiles_press)
#     count_pair(('SST', 'Pressure'), sst_array, press_array, quantiles_sst, quantiles_press)
#     return


def count_1d_Korolev(files_path_prefix: str,
                     flux: np.ndarray,
                     time_start: int,
                     time_end: int,
                     path: str = 'Synthetic/',
                     quantiles_amount: int = 50,
                     n_components: int = 2,
                     start_index: int = 0,
                     ):
    """
    Counts and saves to files_path_prefix + path + 'Kor/daily' A and B estimates for flux array for each day
    t+start index for t in (time_start, time_end)
    :param files_path_prefix: path to the working directory
    :param flux: np.array with shape [time_steps, height, width]
    :param time_start: int counter of start day
    :param time_end: int counter of end day
    :param path: additional path to the folder from files_path_prefix, like 'Synthetic/', 'Components/sensible/',
    'Components/latent/'
    :param quantiles_amount: how many quantiles to use (for one step)
    :param n_components: amount of components for EM
    :param start_index: offset index when saving maps
    :return:
    """
    if not os.path.exists(files_path_prefix + '3D_coeff_Kor'):
        os.mkdir(files_path_prefix + '3D_coeff_Kor')

    if not os.path.exists(files_path_prefix + '3D_coeff_Kor/' + path):
        os.mkdir(files_path_prefix + '3D_coeff_Kor/' + path)

    if not os.path.exists(files_path_prefix + f'3D_coeff_Kor/{path}/daily-mini'):
        os.mkdir(files_path_prefix + f'3D_coeff_Kor/{path}/daily-mini')

    a_map = np.zeros((flux.shape[1], flux.shape[2]), dtype=float)
    b_map = np.zeros((flux.shape[1], flux.shape[2]), dtype=float)
    a_map[np.isnan(flux[0])] = np.nan
    b_map[np.isnan(flux[0])] = np.nan
    # start_time = time.time()
    for t in range(time_start + 1, time_end):
        if t % 100 == 0:
            print(f't = {t}', flush=True)
        flux_array, quantiles = scale_to_bins(flux[t - 1], quantiles_amount)
        flux_set = list(set(flux_array[np.logical_not(np.isnan(flux_array))].flat))
        for group in range(len(flux_set)):
            value_t0 = flux_set[group]
            if np.isnan(value_t0):
                continue
            day_sample = (flux[t][np.where(flux_array == value_t0)] -
                          flux[t - 1][np.where(flux_array == value_t0)]).flatten()
            # print(len(day_sample))
            # plot_hist(day_sample, group)

            window = day_sample
            try:
                gm = GaussianMixture(n_components=n_components,
                                     tol=1e-4,
                                     covariance_type='spherical',
                                     max_iter=1000,
                                     init_params='random',
                                     n_init=5
                                     ).fit(window.reshape(-1, 1))
                means = gm.means_.flatten()
                sigmas_squared = gm.covariances_.flatten()
                weights = gm.weights_.flatten()
                weights /= sum(weights)
            except ValueError:
                means = np.mean(window)
                weights = np.ones(2)
                sigmas_squared = np.array([1, 0])

            a_sum = sum(means * weights)
            b_sum = math.sqrt(sum(weights * (means ** 2 + sigmas_squared)))

            a_map[np.where(flux_array == value_t0)] = a_sum
            b_map[np.where(flux_array == value_t0)] = b_sum

        np.save(files_path_prefix + f'3D_coeff_Kor/{path}/daily-mini/A_{t+start_index}.npy', a_map)
        np.save(files_path_prefix + f'3D_coeff_Kor/{path}/daily-mini/B_{t+start_index}.npy', b_map)
        # print(f'Iteration {t}: {(time.time() - start_time):.1f} seconds')
        # start_time = time.time()
    return


if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------
    mask = load_mask(files_path_prefix)
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
    # variable = 'SST'

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
    # array_scaled = np.load(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_scaled.npy')
    # print(np.isnan(array_scaled).any())

    # # count A and B coeff mini
    # array = np.load(files_path_prefix + f'DATA/{variable}_1979-2025_grouped.npy')
    # np.nan_to_num(array, copy=False)
    # count_1d_Korolev(files_path_prefix, array, 0, array.shape[0], variable, 15)
    #
    # a_coeff = np.zeros_like(array)
    # for day in range(1, a_coeff.shape[0] - 1):
    #     a_day = np.load(files_path_prefix + f'3D_coeff_Kor/{variable}/daily-mini/A_{day}.npy')
    #     a_coeff[day] = a_day
    # np.save(files_path_prefix + f'DATA/{variable}_1979-2025_a_coeff.npy', a_coeff)

    flux_array = np.load(files_path_prefix + f'DATA/FLUX_1979-2025_grouped.npy')
    sst_array = np.load(files_path_prefix + f'DATA/SST_1979-2025_grouped.npy')
    press_array = np.load(files_path_prefix + f'DATA/PRESS_1979-2025_grouped.npy')

    np.save(files_path_prefix + f'DATA/FLUX_1979-2025_grouped_diff.npy', np.diff(flux_array, axis=0))
    np.save(files_path_prefix + f'DATA/SST_1979-2025_grouped_diff.npy', np.diff(sst_array, axis=0))
    np.save(files_path_prefix + f'DATA/PRESS_1979-2025_grouped_diff.npy', np.diff(press_array, axis=0))

    for variable in ['FLUX', 'SST', 'PRESS']:
        array = np.load(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_diff.npy')
        array_scaled, quantiles = scale_to_bins(array, bins=cfg.bins)
        np.save(files_path_prefix + f'DATA/{variable}_1979-2025_grouped_diff_scaled.npy', array_scaled)
        np.save(files_path_prefix + f'DATA/{variable}_1979-2025_diff_quantiles.npy', quantiles)

    # count_eigenvalues_triplets(files_path_prefix, 0, flux_array, sst_array, press_array, mask, 0, 100)

    # for var1 in ['Flux', 'SST', 'Pressure']:
    #     for var2 in ['Flux', 'SST', 'Pressure']:
    #         print(f'Collecting {var1}-{var2}', flush=True)
    #         eigenarray = np.zeros_like(flux_array)
    #         for t in range(flux_array.shape[0]):
    #             try:
    #                 eigenarray[t] = np.load(files_path_prefix + f'Eigenvalues/{var1}-{var2}/eigen0_{t}.npy').reshape((161, 181))[::2, ::2]
    #             except FileNotFoundError:
    #                 print(f'Not existing {files_path_prefix}/Eigenvalues/{var1}-{var2}/eigen0_{t}.npy', flush=True)
    #         np.save(files_path_prefix + f'DATA/{var1}-{var2}_eigen.npy', eigenarray)


