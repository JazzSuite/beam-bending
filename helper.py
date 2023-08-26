import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.ndimage import filters

# from main import elast, length, lenght_specific_mass, mom_of_inertia, num_elem, num_points_elem, ord_of_quad, q_load


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


def get_m_bar(length_specific_mass, h_l, n):
    m_ = np.array([[    156.,         22.*h_l,      54.,        -13.*h_l],
                   [ 22.*h_l,  4.*pow(h_l, 2),  13.*h_l, -3.*pow(h_l, 2)],
                   [     54.,         13.*h_l,     156.,        -22.*h_l],
                   [-13.*h_l, -3.*pow(h_l, 2), -22.*h_l,  4.*pow(h_l, 2)]])
    m_ *= (length_specific_mass * h_l)/420

    # make 3D array with n-times the m_ array
    three_d_m = np.tile(m_, (n, 1, 1))

    return m_, three_d_m


def get_m(length_specific_mass, h_l, n):
    # get_indices to shape correct 1D array for rows and columns
    # and reshape the 3d arrays so that they are in 1D arrays
    indices = get_indices_in_1_d(n)

    # get the matrix m and convert it into a 1D array (n times)
    m_bar = get_m_bar(length_specific_mass, h_l, n)
    m_bar_three_d = m_bar[1]
    m_bar_one_d = m_bar_three_d.flatten()

    # create sparse matrix with shape mxm, m = 4 + (2 * (n-1))
    m = 4 + (2 * (n-1))
    coord_matrix = coo_matrix((m_bar_one_d, (indices[0], indices[1])), shape=(m, m)).tocsr()

    return coord_matrix


def get_s_bar(elast, mom_inertia, h_l, n):
    s_ = np.array([[   12.,         6.*h_l,    -12.,         6.*h_l],
                   [6.*h_l, 4.*pow(h_l, 2), -6.*h_l, 2.*pow(h_l, 2)],
                   [  -12.,        -6.*h_l,     12.,        -6.*h_l],
                   [6.*h_l, 2.*pow(h_l, 2), -6.*h_l, 4.*pow(h_l, 2)]])

    s_ *= (elast * mom_inertia)/pow(h_l, 3)

    # make 3D array with n-times the m_ array
    three_d_s = np.tile(s_, (n, 1, 1))

    return s_,  three_d_s


def get_s(elast, mom_inertia, h_l, n):
    # get_indices_in_1_d to shape correct 1D array for rows and columns
    # and reshape the 3d arrays so that they are in 1D arrays
    indices = get_indices_in_1_d(n)

    # get the matrix s and convert it into a 1D array (n times)
    s_bar = get_s_bar(elast, mom_inertia, h_l, n)
    s_bar_three_d = s_bar[1]
    s_bar_one_d = s_bar_three_d.flatten()

    # create sparse matrix with shape mxm, m = 4 + (2 * (n-1))
    m = 4 + (2 * (n - 1))
    coord_matrix = coo_matrix((s_bar_one_d, (indices[0], indices[1])), shape=(m, m)).tocsr()

    return coord_matrix


def get_q_bar(q_load, h_l, n):
    q_ = np.array([[   6.],
                   [ h_l],
                   [   6.],
                   [-h_l]])

    q_ *= (q_load * h_l)/12.

    # make 3D array with n-times the q_ array
    three_d_q = np.tile(q_, (n, 1, 1))

    return q_, three_d_q


def get_q(q_load, h_l, n):
    # get_indices_in_1_d for rows and fill columns with zeros because it's a vector
    indices = get_indices_in_1_d(n)
    length_indices = len(indices[2])
    columns = np.zeros(length_indices, dtype=int)

    # get vector q and convert it to 1D data array
    q_bar = get_q_bar(q_load, h_l, n)
    q_bar_three_d = q_bar[1]
    q_bar_one_d = q_bar_three_d.flatten()

    # create sparse matrix with shape mx1, m = 4 + (2 * (n-1))
    m = 4 + (2 * (n - 1))
    coord_vector = coo_matrix((q_bar_one_d, (indices[2], columns)), shape=(m, 1)).tocsr()

    return coord_vector


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


def get_m_e(b_matrix, length_specific_mass, h_l, n):
    c = get_c(b_matrix, n)
    zero_dimension = c.shape[1]
    m = get_m(length_specific_mass, h_l, n)
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


def get_s_e(b_matrix, elast, mom_inertia, h_l, n):
    # get s and c
    s = get_s(elast, mom_inertia, h_l, n)
    c = get_c(b_matrix, n)
    c_transposed = c.transpose()
    dimension_zero = c.shape[1]
    zero = coo_matrix((dimension_zero, dimension_zero)).tocsr()

    # stack parts together
    upper = scipy.sparse.hstack([           s,    c])
    lower = scipy.sparse.hstack([c_transposed, zero])

    s_e = scipy.sparse.vstack([upper, lower]).tocsr()

    return s_e


def get_v_e(q_load, h_l, b_matrix, n, t):
    v_q = get_q(q_load, h_l, n)
    v_n = get_vn(b_matrix, n, t)
    v_d = get_vd(b_matrix)

    upper = v_q + v_n
    lower = v_d

    v_e = scipy.sparse.vstack([upper, lower]).tocsr()

    return v_e


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


# stuetzstellen1 = np.linspace(0, 1, 4)
# stencil1 = get_stencil(stuetzstellen1)
# print(stencil1)


def get_phi(stuetzstellen):
    phi_0 = np.vectorize(lambda x: 1 - 3 * pow(x, 2) + 2 * pow(x, 3))
    phi_1 = np.vectorize(lambda x: x - 2 * pow(x, 2) + pow(x, 3))
    phi_2 = np.vectorize(lambda x: 3 * pow(x, 2) - 2 * pow(x, 3))
    phi_3 = np.vectorize(lambda x: -pow(x, 2) + pow(x, 3))

    row0 = phi_0(stuetzstellen)
    row1 = phi_1(stuetzstellen)
    row2 = phi_2(stuetzstellen)
    row3 = phi_3(stuetzstellen)

    phi = np.vstack((row0, row1, row2, row3))

    return phi


def get_dd_phi(stuetzstellen):
    phi_dd_0 = np.vectorize(lambda x: -6 + 12 * x)
    phi_dd_1 = np.vectorize(lambda x: -4 + 6 * x)
    phi_dd_2 = np.vectorize(lambda x: 6 - 12 * x)
    phi_dd_3 = np.vectorize(lambda x: -2 + 6 * x)

    row0 = phi_dd_0(stuetzstellen)
    row1 = phi_dd_1(stuetzstellen)
    row2 = phi_dd_2(stuetzstellen)
    row3 = phi_dd_3(stuetzstellen)

    phi_dd = np.vstack((row0, row1, row2, row3))

    return phi_dd


def get_h(length, n):
    part_length = length / n
    length_array = [part_length]
    length_array = np.tile(length_array, n)

    return length_array


# TODO: inverse
def get_t_inv(length, n, stuetzstellen):
    h = get_h(length, n)
    x_l = np.linspace(0, 1, n+1)


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
    add_delta_one_d = add_delta_one_d[:, np.newaxis]
    add_delta_two_d = np.repeat(add_delta_one_d, n, axis=1)

    return add_delta_three_d, add_delta_two_d


# test1, test2 = get_exp(3)
# print("test1: \n", test1)
# print("test2: \n", test2)

# h_array = get_h(1, 3)
# print("H Array: ", h_array)
#
#
# stuetzstellen2 = np.linspace(0, 1, 8)
# print("stuetz: ", stuetzstellen2)
# phi_abl = get_dd_phi(stuetzstellen2)
# print("phi abl: ", phi_abl)


def get_plot(x_values, y_values, y_lim1, y_lim2, x_lim1=0, x_lim2=1, x_label='x', y_label='y'):
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_lim1, x_lim2)
    plt.ylim(y_lim1, y_lim2)
    plt.show()


def get_plot_log(x_values, y_values, y_lim1, y_lim2, x_lim1=0, x_lim2=1, x_label='x', y_label='y'):
    x_values = np.log10(x_values)
    y_values = np.log10(y_values)
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.xlim(x_lim1, x_lim2)
    # plt.ylim(y_lim1, y_lim2)
    plt.show()


arr = np.linspace(0, 1, 4)
print(arr)
