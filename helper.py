import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def get_indices(n):
    i = np.array([0, 1, 2, 3])
    j = np.array([0, 1, 2, 3])
    l = np.linspace(0, n-1, n, dtype=int)

    # 3D arrays
    m_i, m_l, m_j = np.meshgrid(j, l, i)

    m_2li = 2*m_l + m_i

    m_2lj = 2*m_l + m_j

    # 2D arrays
    m_l_two_d, m_i_two_d = np.meshgrid(l, i)
    m_2li_two_d = 2*m_l_two_d + m_i_two_d

    return m_l, m_i, m_j, m_2li, m_2lj, m_l_two_d, m_i_two_d, m_2li_two_d


def get_indices_20(n, n_p_e, k):
    j = np.array([0, 1, 2, 3])
    l = np.linspace(0, n - 1, n, dtype=int)

    # create 3D arrays for Aufgabe 20
    k_s, l_s, j_s = np.meshgrid(k, l, j)

    # create indices arrays for Aufgabe 20
    m_nlk = n_p_e * l_s + k_s
    m_2lj = 2 * l_s + j_s

    rows_flatten = m_2lj.flatten()
    columns_flatten = m_nlk.flatten()

    return j_s, m_nlk, m_2lj, rows_flatten, columns_flatten


def get_indices_in_1_d(n):
    # for 3D arrays
    indices = get_indices(n)
    three_d_rows    = indices[3]
    three_d_columns = indices[4]

    flattened_rows    = three_d_rows.flatten()
    flattened_columns = three_d_columns.flatten()

    # for 2D arrays
    two_d_rows = indices[7]
    transposed_two_d_rows = np.transpose(two_d_rows)
    flattened_two_d_rows = transposed_two_d_rows.flatten()

    return flattened_rows, flattened_columns, flattened_two_d_rows


# def get_m_bar_old(length_specific_mass, h_l, n):
#     m_ = np.array([[    156.,         22.*h_l,      54.,        -13.*h_l],
#                    [ 22.*h_l,  4.*pow(h_l, 2),  13.*h_l, -3.*pow(h_l, 2)],
#                    [     54.,         13.*h_l,     156.,        -22.*h_l],
#                    [-13.*h_l, -3.*pow(h_l, 2), -22.*h_l,  4.*pow(h_l, 2)]])
#     m_ *= (length_specific_mass * h_l)/420
#
#     # make 3D array with n-times the m_ array
#     three_d_m = np.tile(m_, (n, 1, 1))
#
#     return m_, three_d_m
#
#
# def get_m_old(length_specific_mass, h_l, n):
#     # get_indices to shape correct 1D array for rows and columns
#     # and reshape the 3d arrays so that they are in 1D arrays
#     indices = get_indices_in_1_d(n)
#
#     # get the matrix m and convert it into a 1D array (n times)
#     m_bar = get_m_bar_old(length_specific_mass, h_l, n)
#     m_bar_three_d = m_bar[1]
#     m_bar_one_d = m_bar_three_d.flatten()
#
#     # create sparse matrix with shape mxm, m = 4 + (2 * (n-1))
#     m = 4 + (2 * (n-1))
#     coord_matrix = coo_matrix((m_bar_one_d, (indices[0], indices[1])), shape=(m, m)).tocsr()
#
#     return coord_matrix
#
#
# def get_s_bar_old(elast, mom_inertia, h_l, n):
#     s_ = np.array([[   12.,         6.*h_l,    -12.,         6.*h_l],
#                    [6.*h_l, 4.*pow(h_l, 2), -6.*h_l, 2.*pow(h_l, 2)],
#                    [  -12.,        -6.*h_l,     12.,        -6.*h_l],
#                    [6.*h_l, 2.*pow(h_l, 2), -6.*h_l, 4.*pow(h_l, 2)]])
#
#     s_ *= (elast * mom_inertia)/pow(h_l, 3)
#
#     # make 3D array with n-times the m_ array
#     three_d_s = np.tile(s_, (n, 1, 1))
#
#     return s_,  three_d_s
#
#
# def get_s_old(elast, mom_inertia, h_l, n):
#     # get_indices_in_1_d to shape correct 1D array for rows and columns
#     # and reshape the 3d arrays so that they are in 1D arrays
#     indices = get_indices_in_1_d(n)
#
#     # get the matrix s and convert it into a 1D array (n times)
#     s_bar = get_s_bar_old(elast, mom_inertia, h_l, n)
#     s_bar_three_d = s_bar[1]
#     s_bar_one_d = s_bar_three_d.flatten()
#
#     # create sparse matrix with shape mxm, m = 4 + (2 * (n-1))
#     m = 4 + (2 * (n - 1))
#     coord_matrix = coo_matrix((s_bar_one_d, (indices[0], indices[1])), shape=(m, m)).tocsr()
#
#     return coord_matrix
#
#
# def get_q_bar_old(q_load, h_l, n):
#     q_ = np.array([[   6.],
#                    [ h_l],
#                    [   6.],
#                    [-h_l]])
#
#     q_ *= (q_load * h_l)/12.
#
#     # make 3D array with n-times the q_ array
#     three_d_q = np.tile(q_, (n, 1, 1))
#
#     return q_, three_d_q
#
#
# def get_q_old(q_load, h_l, n):
#     # get_indices_in_1_d for rows and fill columns with zeros because it's a vector
#     indices = get_indices_in_1_d(n)
#     length_indices = len(indices[2])
#     columns = np.zeros(length_indices, dtype=int)
#
#     # get vector q and convert it to 1D data array
#     q_bar = get_q_bar_old(q_load, h_l, n)
#     q_bar_three_d = q_bar[1]
#     q_bar_one_d = q_bar_three_d.flatten()
#
#     # create sparse matrix with shape mx1, m = 4 + (2 * (n-1))
#     m = 4 + (2 * (n - 1))
#     coord_vector = coo_matrix((q_bar_one_d, (indices[2], columns)), shape=(m, 1)).tocsr()
#
#     return coord_vector


def get_number_of_fits_in_array(arr, target):
    length  = len(arr[np.where(arr == target)])
    return length


def get_c(b_matrix, n):
    # for e1 get the second column, filter for the ones and get the num of ones
    # then get the first num entries from the first column and multiply by 2 (because 2*k)
    # then create fitting data, rows and column arrays to create the sparse matrix
    sec_col = b_matrix[:, 1]
    length_of_data_1 = get_number_of_fits_in_array(sec_col, 1)
    rows_1  = 2 * b_matrix[:length_of_data_1, 0]
    columns_1 = np.arange(length_of_data_1)
    data_1 = np.ones(length_of_data_1, dtype=int)

    # creating sparse matrix e1
    m = 2 * (n + 1)
    e1 = coo_matrix((data_1, (rows_1, columns_1)), shape=(m, length_of_data_1)).tocsr()

    # for e2 we do the same as for e1 but here we have to slice different
    # also we have to multiply by two and add 1 (because 2k + 1)
    length_of_data_2 = get_number_of_fits_in_array(sec_col, 2)
    rows_2 = 2 * b_matrix[length_of_data_1:(length_of_data_1 + length_of_data_2), 0]
    rows_2 = rows_2 + 1
    columns_2 = np.arange(length_of_data_2)
    data_2 = np.ones(length_of_data_2, dtype=int)

    # creating sparse matrix e2
    e2 = coo_matrix((data_2, (rows_2, columns_2)), shape=(m, length_of_data_2)).tocsr()

    # combine e1 and e2 to get matrix C
    c = scipy.sparse.hstack([e1, e2]).tocsr()

    return c


def get_vn(b_matrix, n, t=1):
    # we will use the same method here as for 'get_c'
    sec_col = b_matrix[:, 1]
    length_of_data_1 = get_number_of_fits_in_array(sec_col, 1)
    length_of_data_2 = get_number_of_fits_in_array(sec_col, 2)

    # for e3
    length_of_data_3 = get_number_of_fits_in_array(sec_col, 3)
    min_ = (length_of_data_1 + length_of_data_2)
    rows_3    = 2 * b_matrix[min_:(min_ + length_of_data_3), 0]
    rows_3    = rows_3 + 1
    columns_3 = np.arange(length_of_data_3)
    data_3 = np.ones(length_of_data_3, dtype=int)

    # creating sparse matrix e3
    m = 2 * (n+1)
    e3 = coo_matrix((data_3, (rows_3, columns_3)), shape=(m, length_of_data_3)).tocsr()

    # for e4
    length_of_data_4 = get_number_of_fits_in_array(sec_col, 4)
    min_2 = min_ + length_of_data_3
    rows_4    = 2 * b_matrix[min_2: (min_2 + length_of_data_4), 0]
    columns_4 = np.arange(length_of_data_4)
    data_4 = np.ones(length_of_data_4, dtype=int)

    # creating sparse matrix e4
    e4 = coo_matrix((data_4, (rows_4, columns_4)), shape=(m, length_of_data_4)).tocsr()

    # for c3
    rows_5 = np.arange(length_of_data_3)
    columns_5 = np.zeros(length_of_data_3)
    data_5 = b_matrix[min_:(min_ + length_of_data_3), 2]

    c3 = coo_matrix((data_5, (rows_5, columns_5)), shape=(length_of_data_3, 1)).tocsr()

    # for c4
    rows_6    = np.arange(length_of_data_4)
    columns_6 = np.zeros(length_of_data_4, dtype=int)
    data_6 = b_matrix[min_2:(min_2 + length_of_data_4), 2]

    c4 = coo_matrix((data_6, (rows_6, columns_6)), shape=(length_of_data_4, 1)).tocsr()

    v_n = e3.dot(c3) + e4.dot(c4)

    return v_n


def get_vd(b_matrix):
    # we take the first column and check from which to which line the values for a and b are
    sec_col = b_matrix[:, 1]
    length_of_a = get_number_of_fits_in_array(sec_col, 1)
    length_of_b = get_number_of_fits_in_array(sec_col, 2)
    len_a_and_b = length_of_a + length_of_b
    rows = np.arange(len_a_and_b)
    columns = np.zeros(len_a_and_b, dtype=int)

    # data for a
    data_a = b_matrix[:length_of_a, 2]

    # for b
    data_b = b_matrix[length_of_a:len_a_and_b, 2]

    # combine a and b to data v_d
    data_vd = np.concatenate((data_a, data_b))

    # create sparse matrix vd
    vd = coo_matrix((data_vd, (rows, columns)), shape=(len_a_and_b, 1)).tocsr()

    return vd


# def get_m_e_old(b_matrix, length_specific_mass, h_l, n):
#     c = get_c(b_matrix, n)
#     zero_dimension = c.shape[1]
#     m = get_m_old(length_specific_mass, h_l, n)
#     length = m.shape[1]
#     # create different zeros to combine
#     zero_1 = coo_matrix(( zero_dimension,         length)).tocsr()
#     zero_2 = coo_matrix((         length, zero_dimension)).tocsr()
#     zero_3 = coo_matrix(( zero_dimension, zero_dimension)).tocsr()
#
#     # stack parts together
#     upper = scipy.sparse.hstack([     m, zero_2])
#     lower = scipy.sparse.hstack([zero_1, zero_3])
#
#     m_e = scipy.sparse.vstack([upper, lower]).tocsr()
#
#     return m_e
#
#
# def get_s_e_old(b_matrix, elast, mom_inertia, h_l, n):
#     # get s and c
#     s = get_s_old(elast, mom_inertia, h_l, n)
#     c = get_c(b_matrix, n)
#     c_transposed = c.transpose()
#     dimension_zero = c.shape[1]
#     zero = coo_matrix((dimension_zero, dimension_zero)).tocsr()
#
#     # stack parts together
#     upper = scipy.sparse.hstack([           s,    c])
#     lower = scipy.sparse.hstack([c_transposed, zero])
#
#     s_e = scipy.sparse.vstack([upper, lower]).tocsr()
#
#     return s_e
#
#
# def get_v_e_old(q_load, h_l, b_matrix, n, t):
#     v_q = get_q_old(q_load, h_l, n)
#     v_n = get_vn(b_matrix, n, t)
#     v_d = get_vd(b_matrix)
#
#     upper = v_q + v_n
#     lower = v_d
#
#     v_e = scipy.sparse.vstack([upper, lower]).tocsr()
#
#     return v_e


def get_stencil(stuetzstellen):
    # create exponents from 0 to n where n is the number of stuetzstellen
    stuetz_length = len(stuetzstellen)
    exponents = np.arange(0, stuetz_length)

    exp_matrix, stuetz_matrix = np.meshgrid(exponents, stuetzstellen)

    # multiply the matrix of stuetzvektoren with the exponent matrix
    vandermonde_matrix = stuetz_matrix ** exp_matrix

    # create vector V^(+1)
    vector_v = (1 / (exponents + 1)).reshape(-1, 1)

    # calculate stencil
    stencil = np.linalg.solve(np.transpose(vandermonde_matrix), vector_v)

    return stencil.flatten()


def get_phi(stuetzstellen, n):
    phi_0 = np.vectorize(lambda x: 1 - 3 * pow(x, 2) + 2 * pow(x, 3))
    phi_1 = np.vectorize(lambda x: x - 2 * pow(x, 2) + pow(x, 3))
    phi_2 = np.vectorize(lambda x: 3 * pow(x, 2) - 2 * pow(x, 3))
    phi_3 = np.vectorize(lambda x: -pow(x, 2) + pow(x, 3))

    row0 = phi_0(stuetzstellen)
    row1 = phi_1(stuetzstellen)
    row2 = phi_2(stuetzstellen)
    row3 = phi_3(stuetzstellen)

    # create 2D and 3D phi_i
    phi_i_2d = np.vstack((row0, row1, row2, row3))
    phi_i_3d = np.tile(phi_i_2d, (n, 1, 1))
    # create 4D phi_i
    phi_i_3d_temp = np.tile(phi_i_2d, (4, 1, 1))
    phi_i_4d = np.tile(phi_i_3d_temp, (n, 1, 1, 1))

    # create 4D phi_j
    phi_j_2d_0 = np.tile(row0, (4, 1))
    phi_j_2d_1 = np.tile(row1, (4, 1))
    phi_j_2d_2 = np.tile(row2, (4, 1))
    phi_j_2d_3 = np.tile(row3, (4, 1))
    phi_j_3d = np.stack([phi_j_2d_0, phi_j_2d_1, phi_j_2d_2, phi_j_2d_3])
    phi_j_4_d = np.tile(phi_j_3d, (n, 1, 1, 1))

    # create phi_j for Aufgabe 20
    row0 = row0.reshape(len(stuetzstellen), 1)
    row1 = row1.reshape(len(stuetzstellen), 1)
    row2 = row2.reshape(len(stuetzstellen), 1)
    row3 = row3.reshape(len(stuetzstellen), 1)
    phi_j_new = np.hstack((row0, row1, row2, row3))
    phi_j_new_final = np.tile(phi_j_new, (n, 1, 1))

    return phi_i_2d, phi_i_3d, phi_i_4d, phi_j_4_d, phi_j_new_final


def get_dd_phi(stuetzstellen, n):
    phi_dd_0 = np.vectorize(lambda x: -6 + 12 * x)
    phi_dd_1 = np.vectorize(lambda x: -4 + 6 * x)
    phi_dd_2 = np.vectorize(lambda x: 6 - 12 * x)
    phi_dd_3 = np.vectorize(lambda x: -2 + 6 * x)

    row0 = phi_dd_0(stuetzstellen)
    row1 = phi_dd_1(stuetzstellen)
    row2 = phi_dd_2(stuetzstellen)
    row3 = phi_dd_3(stuetzstellen)

    # create 2D phi_dd_i
    phi_dd_i_2d = np.vstack((row0, row1, row2, row3))
    # create 4D phi_dd_i
    phi_dd_i_3d_temp = np.tile(phi_dd_i_2d, (4, 1, 1))
    phi_dd_i_4d = np.tile(phi_dd_i_3d_temp, (n, 1, 1, 1))

    # create 4D phi_dd_j
    phi_dd_j_2d_0 = np.tile(row0, (4, 1))
    phi_dd_j_2d_1 = np.tile(row1, (4, 1))
    phi_dd_j_2d_2 = np.tile(row2, (4, 1))
    phi_dd_j_2d_3 = np.tile(row3, (4, 1))
    phi_dd_j_3d = np.stack([phi_dd_j_2d_0, phi_dd_j_2d_1, phi_dd_j_2d_2, phi_dd_j_2d_3])
    phi_dd_j_4_d = np.tile(phi_dd_j_3d, (n, 1, 1, 1))

    phi_dd_i_3d = np.tile(phi_dd_i_2d, (n, 1, 1))

    return phi_dd_i_2d, phi_dd_i_3d, phi_dd_i_4d, phi_dd_j_4_d


def get_h(length, n):
    part_length = length / n
    length_array_2d = [part_length]
    length_array_2d = np.tile(length_array_2d, n)
    length_array_2d = length_array_2d.reshape(n, 1)

    length_array_3d = length_array_2d.reshape(n, 1, 1)

    return length_array_2d, length_array_3d


def get_t_inv(length, n, n_tilde):
    # create quadrature array
    x_k = np.linspace(0, length, n_tilde+1)
    x_k = x_k.reshape(1, n_tilde+1)
    # get h (2D)
    h = get_h(length, n)[0]
    x_l_index = np.arange(n)
    x_l = x_l_index * h[0]
    x_l = x_l[:, np.newaxis]

    # calculate as described in the task
    end_matrix_2d = h @ x_k + x_l

    # here we can simply use reshape since we are only adding dimensions without the need of stacking or similar
    end_matrix_3d = end_matrix_2d.reshape(n, 1, n_tilde+1)
    end_matrix_4d = end_matrix_3d.reshape(n, 1, 1, n_tilde+1)

    return end_matrix_2d, end_matrix_3d, end_matrix_4d


def get_exp(n):
    # get the right matrices and extract a matrix from a lower dimension since they repeat themselves
    matrices = get_indices(n)
    i_matrix = matrices[1][0]
    j_matrix = matrices[2][0]
    i_vector = matrices[6][:, 0]

    # search for matches and make a matrix where we place a one at the index where a match is found
    delta_i1 = np.where(np.array(i_matrix) == 1, 1, 0)
    delta_i3 = np.where(np.array(i_matrix) == 3, 1, 0)
    delta_j1 = np.where(np.array(j_matrix) == 1, 1, 0)
    delta_j3 = np.where(np.array(j_matrix) == 3, 1, 0)

    # add all delta matrices as described in the task
    add_delta = delta_i1 + delta_i3 + delta_j1 + delta_j3
    add_delta_three_d = np.tile(add_delta, (n, 1, 1))

    # here the same for the 2D-Array
    delta_i1_one_d = np.where(np.array(i_vector) == 1, 1, 0)
    delta_i3_one_d = np.where(np.array(i_vector) == 3, 1, 0)

    add_delta_one_d = delta_i1_one_d + delta_i3_one_d
    add_delta_two_d = np.tile(add_delta_one_d, (n, 1))

    return add_delta_two_d, add_delta_three_d


def get_exp_20(n, n_p_e, k):
    new_j = get_indices_20(n, n_p_e, k)[0]
    delta_j1_new = np.where(np.array(new_j) == 1, 1, 0)
    delta_j3_new = np.where(np.array(new_j) == 3, 1, 0)
    delta_new_j = delta_j1_new + delta_j3_new

    return delta_new_j


def get_m_b_bar_3d(lambda_my, stuetzstellen, length, n, n_tilde):
    # create potentiated 3D array
    h_l_3d = get_h(length, n)[1]
    exponent_matrix = get_exp(n)
    exponent_matrix = exponent_matrix[1] + 1
    potentiated_matrix = np.power(h_l_3d, exponent_matrix)

    # create 4D array by multiplying my(t_inv(x_k)) * phi_i(x_k) * phi_j(x_k)
    t_inv = get_t_inv(length, n, n_tilde)[2]
    inv_my = lambda_my(t_inv)
    phi_i = get_phi(stuetzstellen, n)[2]
    phi_j = get_phi(stuetzstellen, n)[3]
    temp_4d_array = inv_my * phi_i * phi_j

    stencil = get_stencil(stuetzstellen)

    # to approximate the integral multiply the temp_4d array with the stencil
    approx_integral = np.tensordot(temp_4d_array, stencil, axes=([3], [0]))

    # for the final m_bar multiply elementwise
    m_bar = potentiated_matrix * approx_integral

    return m_bar


def get_m(lambda_my, stuetzstellen, length, n, n_tilde):
    # get_indices to shape correct 1D array for rows and columns
    # and reshape the 3d arrays so that they are in 1D arrays
    indices = get_indices_in_1_d(n)

    # get the matrix m and convert it into a 1D array (n times)
    m_bar = get_m_b_bar_3d(lambda_my, stuetzstellen, length, n, n_tilde)
    m_bar_one_d = m_bar.flatten()

    # create sparse matrix with shape mxm, m = 4 + (2 * (n-1))
    m = 4 + (2 * (n-1))
    coord_matrix = coo_matrix((m_bar_one_d, (indices[0], indices[1])), shape=(m, m)).tocsr()

    return coord_matrix


def get_m_e(lambda_my, stuetzstellen, b_matrix, length, n, n_tilde):
    c = get_c(b_matrix, n)
    zero_dimension = c.shape[1]
    m = get_m(lambda_my, stuetzstellen, length, n, n_tilde)
    length = m.shape[1]
    # create different zeros to combine
    zero_1 = coo_matrix(( zero_dimension,         length)).tocsr()
    zero_2 = coo_matrix((         length, zero_dimension)).tocsr()
    zero_3 = coo_matrix(( zero_dimension, zero_dimension)).tocsr()

    # stack parts together
    upper = scipy.sparse.hstack([     m, zero_2])
    lower = scipy.sparse.hstack([zero_1, zero_3])

    m_e = scipy.sparse.vstack([upper, lower]).tocsr()

    return m_e


def get_s_bar_3d(lambda_e, lambda_i, stuetzstellen, length, n, n_tilde):
    # create potentiated 3D array
    h_l_3d = get_h(length, n)[1]
    exponent_matrix = get_exp(n)
    exponent_matrix = exponent_matrix[1] - 3
    potentiated_matrix = np.power(h_l_3d, exponent_matrix)

    # create 4D array by multiplying e(t_inv(x_k)) * i(t_inv(x_k)) * phi_dd_i(x_k) * phi_dd_j(x_k)
    t_inv = get_t_inv(length, n, n_tilde)[2]
    inv_e = lambda_e(t_inv)
    inv_i = lambda_i(t_inv)
    phi_dd_i = get_dd_phi(stuetzstellen, n)[2]
    phi_dd_j = get_dd_phi(stuetzstellen, n)[3]
    temp_4d_array = inv_e * inv_i * phi_dd_i * phi_dd_j

    stencil = get_stencil(stuetzstellen)

    # to approximate the integral multiply the temp_4d array with the stencil
    approx_integral = np.tensordot(temp_4d_array, stencil, axes=([3], [0]))

    # for the final s_bar multiply elementwise
    s_bar = potentiated_matrix * approx_integral

    return s_bar


def get_s(lambda_e, lambda_i, stuetzstellen, length, n, n_tilde):
    # get_indices_in_1_d to shape correct 1D array for rows and columns
    # and reshape the 3d arrays so that they are in 1D arrays
    indices = get_indices_in_1_d(n)

    # get the matrix s and convert it into a 1D array (n times)
    s_bar = get_s_bar_3d(lambda_e, lambda_i, stuetzstellen, length, n, n_tilde)
    s_bar_one_d = s_bar.flatten()

    # create sparse matrix with shape mxm, m = 4 + (2 * (n-1))
    m = 4 + (2 * (n - 1))
    coord_matrix = coo_matrix((s_bar_one_d, (indices[0], indices[1])), shape=(m, m)).tocsr()

    return coord_matrix


def get_s_e(b_matrix, lambda_e, lambda_i, stuetzstellen, length, n, n_tilde):
    # get s and c
    s = get_s(lambda_e, lambda_i, stuetzstellen, length, n, n_tilde)
    c = get_c(b_matrix, n)
    c_transposed = c.transpose()
    dimension_zero = c.shape[1]
    zero = coo_matrix((dimension_zero, dimension_zero)).tocsr()

    # stack parts together
    upper = scipy.sparse.hstack([           s,    c])
    lower = scipy.sparse.hstack([c_transposed, zero])

    s_e = scipy.sparse.vstack([upper, lower]).tocsr()

    return s_e


def get_q_bar_3d(lambda_q, stuetzstellen, length, n, n_tilde):
    # create potentiated 2D array
    h_l_2d = get_h(length, n)[0]
    exponent_matrix = get_exp(n)
    exponent_matrix = exponent_matrix[0] + 1
    potentiated_matrix = np.power(h_l_2d, exponent_matrix)

    # create 3D array by multiplying q(t_inv(x_k)) * phi_i(x_k)
    t_inv = get_t_inv(length, n, n_tilde)[1]
    inv_q = lambda_q(t_inv)
    phi_i = get_phi(stuetzstellen, n)[1]
    temp_3d_array = inv_q * phi_i

    stencil = get_stencil(stuetzstellen)

    # to approximate the integral multiply the temp_3d array with the stencil
    approx_integral = np.tensordot(temp_3d_array, stencil, axes=([2], [0]))

    # for the final q_bar multiply elementwise
    q_bar = potentiated_matrix * approx_integral
    q_bar = q_bar.reshape(n, 4, 1)

    return q_bar


def get_q(lambda_q, stuetzstellen, length, n, n_tilde):
    # get_indices_in_1_d for rows and fill columns with zeros because it's a vector
    indices = get_indices_in_1_d(n)
    length_indices = len(indices[2])
    columns = np.zeros(length_indices, dtype=int)

    # get vector q and convert it to 1D data array
    q_bar = get_q_bar_3d(lambda_q, stuetzstellen, length, n, n_tilde)
    q_bar_one_d = q_bar.flatten()

    # create sparse matrix with shape mx1, m = 4 + (2 * (n-1))
    m = 4 + (2 * (n - 1))
    coord_vector = coo_matrix((q_bar_one_d, (indices[2], columns)), shape=(m, 1)).tocsr()

    return coord_vector


def get_v_e(b_matrix, lambda_q, stuetzstellen, length, n, n_tilde, t):
    v_q = get_q(lambda_q, stuetzstellen, length, n, n_tilde)
    # v_q = get_q(q_load, h_l, n)
    v_n = get_vn(b_matrix, n, t)
    v_d = get_vd(b_matrix)

    upper = v_q + v_n
    lower = v_d

    v_e = scipy.sparse.vstack([upper, lower]).tocsr()

    return v_e


def get_plot(x_values, y_values, y_lim1, y_lim2, x_lim1=0, x_lim2=1, x_label='x', y_label='y', title='graph'):
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_lim1, x_lim2)
    plt.ylim(y_lim1, y_lim2)
    plt.title(title)
    plt.show()


def get_plot_log(x_values, y_values, y_lim1, y_lim2, x_lim1=0, x_lim2=1, x_label='x', y_label='y', title='graph'):
    x_values = np.log10(x_values)
    y_values = np.log10(y_values)
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.xlim(x_lim1, x_lim2)
    # plt.ylim(y_lim1, y_lim2)
    plt.show()


def get_plot_new(b_matrix, lambda_e, lambda_i, lambda_q, stuetzstellen, eva_array, length, num_elem, n_p_e, n_tilde):
    k = np.arange(n_p_e+1)

    # create potentiated 3D-h-array
    h_l_3d = get_h(length, num_elem)[1]
    exponent_matrix = get_exp_20(num_elem, n_p_e, k)
    potentiated_matrix = np.power(h_l_3d, exponent_matrix)

    # get phi_j and multiply it elementwise with the potentiated matrix
    phi_j = get_phi(eva_array, num_elem)[4]
    result = potentiated_matrix * phi_j
    values = result.flatten()

    indices = get_indices_20(num_elem, n_p_e, k)
    rows = indices[3]
    columns = indices[4]

    m = 4 + (2*(num_elem-1))
    n = (n_p_e+1) + (n_p_e*(num_elem-1))

    # identify values with same indices and just keep one
    sorted_indices = np.lexsort((columns, rows))
    unique_indices, unique_pos = np.unique((rows[sorted_indices], columns[sorted_indices]), axis=1, return_index=True)

    filtered_values = values[sorted_indices][unique_pos]
    filtered_rows = rows[sorted_indices][unique_pos]
    filtered_columns = columns[sorted_indices][unique_pos]

    # assemble sparse matrix
    a_matrix = coo_matrix((filtered_values, (filtered_columns, filtered_rows)), shape=(n, m)).tocsr()

    # calculate alpha
    s_e = get_s_e(b_matrix, lambda_e, lambda_i, stuetzstellen, length, num_elem, n_tilde)
    v_e = get_v_e(b_matrix, lambda_q, stuetzstellen, length, num_elem, n_tilde, 1)
    alpha_e = spsolve(s_e, v_e)
    alpha_e = alpha_e[:2 * num_elem + 2]

    # multiply the evaluation matrix with the vector alpha to get omega
    w = a_matrix @ alpha_e

    x_axis_new = np.linspace(0, length, ((num_elem * n_p_e) + 1))
    get_plot(x_axis_new, w, y_lim1=min(w)-0.13, y_lim2=max(w), x_label=f'x in m mit n = {num_elem}', y_label='\u03C9 in m',
             title='Aufgabe 20')
