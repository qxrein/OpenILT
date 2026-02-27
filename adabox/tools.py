"""
Local copy of the adaptive-boxes `tools` module so that OpenILT can import

    from adabox import tools

without requiring adaptive-boxes to be installed as a separate package.
"""

import numpy as np
from scipy import stats
import json
import pandas as pd


def is_broken(vector_to_test, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    it_is = False

    for i in range(len(vector_to_test) - 1):
        diff_val = abs(vector_to_test[i] - vector_to_test[i + 1])
        if diff_val <= error_ratio_sup:
            if diff_val >= error_ratio_inf:
                it_is = False
            else:
                it_is = True
                break
        else:
            it_is = True
            break

    return it_is


def get_dist_left(all_x_points_arg, init_x_index_arg, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    l_lim = 0
    index_val = init_x_index_arg
    while index_val > l_lim:
        diff_bound_val = abs(all_x_points_arg[index_val, 0] - all_x_points_arg[index_val - 1, 0])
        if diff_bound_val >= error_ratio_sup or diff_bound_val <= error_ratio_inf:
            break
        index_val = index_val - 1

    f_index_l_val = index_val
    dist_l_val = init_x_index_arg - f_index_l_val
    return dist_l_val


def get_dist_right(all_x_points_arg, init_x_index_arg, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    r_lim = len(all_x_points_arg) - 1
    index_val = init_x_index_arg
    while index_val < r_lim:
        diff_bound = abs(all_x_points_arg[index_val, 0] - all_x_points_arg[index_val + 1, 0])
        if diff_bound >= error_ratio_sup or diff_bound <= error_ratio_inf:
            break
        index_val = index_val + 1

    f_index_r_val = index_val + 1
    dist_r_val = f_index_r_val - init_x_index_arg
    return dist_r_val


def get_dist_down(all_y_points_arg, init_y_index_arg, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    d_lim = 0
    index_val = init_y_index_arg
    while index_val > d_lim:
        diff_bound = abs(all_y_points_arg[index_val, 1] - all_y_points_arg[index_val - 1, 1])
        if diff_bound >= error_ratio_sup or diff_bound <= error_ratio_inf:
            break
        index_val = index_val - 1

    f_index_d_val = index_val
    dist_d_val = init_y_index_arg - f_index_d_val
    return dist_d_val


def get_dist_up(all_y_points_arg, init_y_index_arg, sep_value):
    error_ratio = 0.05
    error_ratio_sup = sep_value * (1 + error_ratio)
    error_ratio_inf = sep_value * (1 - error_ratio)
    u_lim = len(all_y_points_arg) - 1
    index_val = init_y_index_arg
    while index_val < u_lim:
        diff_bound = abs(all_y_points_arg[index_val, 1] - all_y_points_arg[index_val + 1, 1])
        if diff_bound >= error_ratio_sup or diff_bound <= error_ratio_inf:
            break
        index_val = index_val + 1

    f_index_u_val = index_val + 1
    dist_u_val = f_index_u_val - init_y_index_arg
    return dist_u_val


def get_final_index_down(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
    down_lim = 0
    index = init_y_index_arg
    while index >= down_lim:
        temp_y = all_y_points_arg[index, 1]
        all_x_points_arg = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)
        temp_x = all_y_points_arg[index, 0]
        temp_x_index = np.where(all_x_points_arg[:, 0] == temp_x)[0][0]

        index_lim_sup = temp_x_index + dist_r_arg
        index_lim_inf = temp_x_index - dist_l_arg

        if index_lim_inf < 0:
            index_lim_inf = 0

        if index_lim_sup > len(all_x_points_arg):
            index_lim_sup = len(all_x_points_arg)

        temp_range_lr = range(index_lim_inf, index_lim_sup)
        just_x = all_x_points_arg[temp_range_lr, 0]
        if is_broken(just_x, sep_value):
            break
        index = index - 1

    final_index_val = index + 1
    return final_index_val


def get_final_index_up(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
    up_lim = len(all_y_points_arg) - 1
    index = init_y_index_arg
    while index <= up_lim:
        temp_y = all_y_points_arg[index, 1]
        all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

        temp_x = all_y_points_arg[index, 0]
        temp_x_index = np.where(all_x_points[:, 0] == temp_x)[0][0]

        index_lim_sup = temp_x_index + dist_r_arg
        index_lim_inf = temp_x_index - dist_l_arg

        if index_lim_inf < 0:
            index_lim_inf = 0

        if index_lim_sup > len(all_x_points):
            index_lim_sup = len(all_x_points)

        temp_range_lr = range(index_lim_inf, index_lim_sup)
        just_x = all_x_points[temp_range_lr, 0]
        if is_broken(just_x, sep_value):
            break
        index = index + 1

    final_index_val = index - 1
    return final_index_val


def get_final_xy_index_down(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
    final_index = get_final_index_down(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value)

    temp_y = all_y_points_arg[final_index, 1]
    all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

    temp_x = all_y_points_arg[final_index, 0]
    temp_x_index = np.where(all_x_points[:, 0] == temp_x)[0][0]

    index_lim_sup = temp_x_index + dist_r_arg
    index_lim_inf = temp_x_index - dist_l_arg

    if index_lim_inf < 0:
        index_lim_inf = 0

    if index_lim_sup > len(all_x_points):
        index_lim_sup = len(all_x_points)

    temp_range_lr = range(index_lim_inf, index_lim_sup)
    final_x_min = all_x_points[temp_range_lr, 0].min()
    final_x_max = all_x_points[temp_range_lr, 0].max()
    final_y_down = temp_y
    return final_x_min, final_x_max, final_y_down


def get_final_xy_index_up(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
    final_index = get_final_index_up(data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value)

    temp_y = all_y_points_arg[final_index, 1]
    all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

    temp_x = all_y_points_arg[final_index, 0]
    temp_x_index = np.where(all_x_points[:, 0] == temp_x)[0][0]

    index_lim_sup = temp_x_index + dist_r_arg
    index_lim_inf = temp_x_index - dist_l_arg

    if index_lim_inf < 0:
        index_lim_inf = 0

    if index_lim_sup > len(all_x_points):
        index_lim_sup = len(all_x_points)

    temp_range_lr = range(index_lim_inf, index_lim_sup)
    final_x_min = all_x_points[temp_range_lr, 0].min()
    final_x_max = all_x_points[temp_range_lr, 0].max()
    final_y_up = temp_y
    return final_x_min, final_x_max, final_y_up


def get_separation_value(data_2d_global_arg):
    n_sample = 100
    x_data = np.unique(np.sort(data_2d_global_arg[:, 0]))
    y_data = np.unique(np.sort(data_2d_global_arg[:, 1]))

    diffs_x = np.zeros(shape=[n_sample])
    diffs_y = np.zeros(shape=[n_sample])

    for p in range(n_sample):
        x_rand_num = int(np.random.rand() * (len(x_data) - 1))
        y_rand_num = int(np.random.rand() * (len(y_data) - 1))
        diffs_x[p] = np.abs(x_data[x_rand_num] - x_data[x_rand_num + 1])
        diffs_y[p] = np.abs(y_data[y_rand_num] - y_data[y_rand_num + 1])

    sep_value_val = (stats.mode(diffs_x, keepdims=True).mode[0] + stats.mode(diffs_y, keepdims=True).mode[0]) / 2
    return sep_value_val


def create_2d_data_from_vertex(vertex_2d_data):
    shape_vertex_data = vertex_2d_data.shape
    data_2d_global_val = np.zeros(shape=[shape_vertex_data[0], (shape_vertex_data[1] - 1) + 1])
    data_2d_global_val[:, [0, 1]] = np.array(vertex_2d_data.loc[:, ["x", "y"]])
    data_2d_global_val = np.unique(data_2d_global_val, axis=0)
    return data_2d_global_val


class Rectangle:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def get_area(self):
        return abs(self.x2 - self.x1) * abs(self.y2 - self.y1)

    def get_side_ratio(self):
        x_len = abs(self.x2 - self.x1)
        y_len = abs(self.y2 - self.y1)
        if y_len == 0:
            return 0
        return x_len / y_len


