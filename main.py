import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import filters
import matplotlib.pyplot as plt

import helper

elast                = 1     # Elastitaetsmodul                                     (in N/m^2)
length               = 1     # Laenge des Balkens                                   (in m    )
lenght_specific_mass = 1     # Laengenspezifische Masse                             (in kg/m )
mom_of_inertia       = 1     # Flaechentraegheitsmoment                             (in m^4  )
newmark_beta         = 0.25  # NEWMARK-Koeffizient                                  (in 1    )
newmark_gamma        = 0.5   # NEWMARK-Koeffizient                                  (in 1    )
num_elem             = 3     # Anzahl der Elemente                                  (in 1    )
num_points_elem      = 1     # Anzahl der zusätzlichen Auswertungspunkte je Element (in 1    )
num_t_steps          = 100   # Anzahl der Zeitschritte                              (in 1    )
ord_of_quad          = 7     # Ordnung der Quadratur                                (in 1    )
q_load               = 1     # Streckenlast                                         (in N/m  )
time_increment       = 1   # Zeitschrittweite                                     (in s    )

b_matrix = np.array([[       0, 1, 0],
                     [       0, 2, 0],
                     [num_elem, 3, 0],
                     [num_elem, 4, 0]])

b_test_matrix = np.array([[       0, 1, 0],
                          [       2, 1, 0],
                          [       0, 2, 0],
                          [3       , 3, 4],
                          [1       , 4, 5],
                          [num_elem, 4, 2]])

# Aufgabe 15
new_lenght_specific_mass = lambda x : pow(x, 0)
new_elast                = lambda x : pow(x, 0)
new_mom_of_inertia       = lambda x : pow(x, 0)
new_q_load               = lambda x : pow(x, 0)


# Aufgabe 10
# h_l = length / num_elem
# s_e = helper.get_s_e(b_matrix, elast, mom_of_inertia, h_l, num_elem)
# v_e = helper.get_v_e(q_load, h_l, b_matrix, num_elem, 1)
# alpha_e = spsolve(s_e, v_e)
# # Aufgabe 11
# indices = np.arange(0, (2 * num_elem + 2), 2, dtype=int)
# w = alpha_e[indices]
# x_axis = np.linspace(0, length, (num_elem + 1))

# helper.get_plot(x_axis, w, y_lim1=-0.13, y_lim2=0.13, x_label=f'x in m mit n = {num_elem}', y_label='\u03C9 in m')

# Aufgabe 12 q_load = 0 motion over time
# Anfangswerte initialisieren
h_l = length / num_elem
v_e_temp     = helper.get_v_e(q_load, h_l, b_matrix, num_elem, 1)
s_e     = helper.get_s_e(b_matrix, elast, mom_of_inertia, h_l, num_elem)
m_e     = helper.get_m_e(b_matrix, lenght_specific_mass, h_l, num_elem)
start_a = spsolve(s_e, v_e_temp)
new_q = 0
start_a = csr_matrix(start_a)
start_a = start_a.transpose()
# print("start_a: \n", start_a.toarray())
zero_dimension = start_a.shape[0]
zeros = csr_matrix((zero_dimension, 1))
start_a_v   = zeros
# print("start_a_v: \n", start_a_v.toarray())
start_a_acc = zeros
v_e = helper.get_v_e(new_q, h_l, b_matrix, num_elem, time_increment)

# get static values for Aufgabe 14
v_q = helper.get_q(new_q, h_l, num_elem)
v_n = helper.get_vn(b_matrix, num_elem)
c_matrix = helper.get_c(b_matrix, num_elem)
m_matrix = helper.get_m(lenght_specific_mass, h_l, num_elem)
s_matrix = helper.get_s(elast, mom_of_inertia, h_l, num_elem)
energy_array = []

x_axis = np.linspace(0, length, (num_elem + 1))

for _ in range(0, int(num_t_steps/time_increment)):
    a_temp   = start_a + (start_a_v * time_increment) + ((1/2 - newmark_beta) * start_a_acc * pow(time_increment, 2))
    a_temp_v = start_a_v + (1 - newmark_gamma) * start_a_acc * time_increment
    # print("a_temp: ", a_temp.toarray())
    # print("a_temp_v: ", a_temp_v.toarray())

    left_side  = m_e + (s_e * newmark_beta * pow(time_increment, 2))
    right_side = v_e - (s_e * a_temp)
    # calculation of acceleration for next time step
    a_acc_p_1 = spsolve(left_side, right_side)
    a_acc_p_1 = csr_matrix(a_acc_p_1)
    a_acc_p_1 = a_acc_p_1.transpose()
    # calculation of position for next time step
    a_p_1     = a_temp + (newmark_beta * a_acc_p_1 * pow(time_increment, 2))
    # calculation of velocity for next time step
    a_v_p_1   = a_temp_v + (newmark_gamma * a_acc_p_1 * time_increment)

    w_indices = np.arange(0, 2 * num_elem + 2, 2, dtype=int)
    w = a_p_1[w_indices]

    # calculation of approx energy
    # get position and speed from the first 2*num_elem + 2 entries of vectors a and a'
    alpha_indices = np.arange(0, 2 * num_elem + 2, 1, dtype=int)
    alpha_p = start_a[alpha_indices]
    alpha_p_v = start_a_v[alpha_indices]
    ny = a_temp[(2 * num_elem + 2):]

    # use the given formula to calculate the approx energy
    temp_var_to_transpose = 1 / 2 * s_matrix @ alpha_p - v_q - c_matrix @ ny - v_n
    energy = (1 / 2 * alpha_p_v.transpose() @ m_matrix @ alpha_p_v) + temp_var_to_transpose.transpose() @ alpha_p
    energy_array.append(energy.toarray()[0][0])

    # helper.get_plot(x_axis, w.toarray(), y_lim1=-0.5, y_lim2=0.5)

    # assign the start values to the calculated values of the next step
    start_a     = a_p_1
    start_a_v   = a_v_p_1
    start_a_acc = a_acc_p_1

x_axis_energy = np.linspace(0, num_t_steps, int(num_t_steps/time_increment), dtype=float)
helper.get_plot(x_axis_energy, energy_array, 0, 0.1, x_lim1=0, x_lim2=100)


def w(x):
    return (q_load/(elast * mom_of_inertia)) * ((pow(x, 4)/24) - ((length * pow(x, 3))/6) + ((pow(length, 2) * pow(x, 2))/4))
    # return q / (E * I)                       * (n ** 4 / 24    - l * n ** 3 / 6           + l ** 2 * n ** 2 / 4)


def w_abl(x):
    return (q_load/(elast * mom_of_inertia)) * ((pow(x, 3)/6) - ((length * pow(x, 2))/2) + ((pow(length, 2) * x)/2))


def convergence_analysis(n_elem):
    h_l_ = length / n_elem
    b_matrix_ = np.array([[0     , 1, 0],
                          [0     , 2, 0],
                          [n_elem, 3, 0],
                          [n_elem, 4, 0]])
    s_e_ = helper.get_s_e(b_matrix_, elast, mom_of_inertia, h_l_, n_elem)
    v_e_ = helper.get_v_e(q_load, h_l_, b_matrix_, n_elem, 1)
    alpha_e_ = spsolve(s_e_, v_e_)

    x_for_w = np.linspace(0, n_elem, n_elem+1)
    x_for_w = x_for_w * h_l_
    w1 = list(map(w, x_for_w))
    w2 = list(map(w_abl, x_for_w))

    w_combined = np.column_stack((w1, w2)).reshape(-1)
    w_combined = csr_matrix(w_combined)

    matrix_m = helper.get_m(1, 1, n_elem)

    fitting_alpha = csr_matrix(alpha_e_[:(2 * n_elem + 2)])
    fitting_alpha = csr_matrix(fitting_alpha)
    w_minus_alpha = w_combined - fitting_alpha
    w_minus_alpha = w_minus_alpha
    error_l2 = (csr_matrix.sqrt(w_minus_alpha @ matrix_m @ w_minus_alpha.transpose()) /
                csr_matrix.sqrt(w_combined @ matrix_m @ w_combined.transpose()))

    return error_l2.item(0)


n_array = np.arange(1, 1001)

error_array = list(map(convergence_analysis, n_array))

helper.get_plot(n_array, error_array, 0, 1, 1, 1000)
helper.get_plot_log(n_array, error_array, -35, 2, 1, 1000)

# temp_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print(temp_array[(2 * num_elem + 2):])
