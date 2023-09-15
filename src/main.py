import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import imageio
import os

import helper

elast                = 1     # Elastitaetsmodul                                     (in N/m^2)
length               = 1     # Laenge des Balkens                                   (in m    )
lenght_specific_mass = 1     # Laengenspezifische Masse                             (in kg/m )
mom_of_inertia       = 1     # Flaechentraegheitsmoment                             (in m^4  )
newmark_beta         = 0.25  # NEWMARK-Koeffizient                                  (in 1    )
newmark_gamma        = 0.5   # NEWMARK-Koeffizient                                  (in 1    )
num_elem             = 3     # Anzahl der Elemente                                  (in 1    )
num_points_elem      = 5     # Anzahl der zusätzlichen Auswertungspunkte je Element (in 1    )
num_t_steps          = 100   # Anzahl der Zeitschritte                              (in 1    )
ord_of_quad          = 7     # Ordnung der Quadratur                                (in 1    )
q_load               = 1     # Streckenlast                                         (in N/m  )
time_increment       = 0.1   # Zeitschrittweite                                     (in s    )

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


stuetzstellen = np.linspace(0, 1, ord_of_quad+1)
eva_array = np.linspace(0, 1, num_points_elem+1)
h_l = length / num_elem

# Aufgabe 10
s_e = helper.get_s_e(b_matrix, new_elast, new_mom_of_inertia, stuetzstellen, length, num_elem, ord_of_quad)
v_e = helper.get_v_e(b_matrix, new_q_load, stuetzstellen, length, num_elem, ord_of_quad, 1)
alpha_e = spsolve(s_e, v_e)

# Aufgabe 11
indices = np.arange(0, (2 * num_elem + 2), 2, dtype=int)
w = alpha_e[indices]
x_axis = np.linspace(0, length, (num_elem + 1))

# create plot for Aufgabe 10
helper.get_plot(x_axis, w, y_lim1=min(w)-0.13, y_lim2=max(w), x_label=f'x in m mit n = {num_elem}', y_label='\u03C9 in m',
                title='Aufgabe 11')


def newmark(beta, gamma, eta):
    # Aufgabe 12 q_load = 0 motion over time
    new_q = lambda x : x * 0
    # initialize start values
    v_e_temp = helper.get_v_e(b_matrix, new_q_load, stuetzstellen, length, num_elem, ord_of_quad, 1)
    s_e_newmark = helper.get_s_e(b_matrix, new_elast, new_mom_of_inertia, stuetzstellen, length, num_elem, ord_of_quad)
    m_e_newmark = helper.get_m_e(new_lenght_specific_mass, stuetzstellen, b_matrix, length, num_elem, ord_of_quad)
    start_a = spsolve(s_e_newmark, v_e_temp)
    start_a = csr_matrix(start_a)
    start_a = start_a.transpose()
    zero_dimension = start_a.shape[0]
    zeros = csr_matrix((zero_dimension, 1))
    start_a_v = zeros
    start_a_acc = zeros
    v_e_newmark = helper.get_v_e(b_matrix, new_q, stuetzstellen, length, num_elem, ord_of_quad, time_increment)

    # get static values for Aufgabe 14
    v_q = helper.get_q(new_q, stuetzstellen, length, num_elem, ord_of_quad)
    v_n = helper.get_vn(b_matrix, num_elem)
    c_matrix = helper.get_c(b_matrix, num_elem)
    m_matrix = helper.get_m(new_lenght_specific_mass, stuetzstellen, length, num_elem, ord_of_quad)
    s_matrix = helper.get_s(new_elast, new_mom_of_inertia, stuetzstellen, length, num_elem, ord_of_quad)
    energy_array = []

    filenames = []
    x_axis_newmark = np.linspace(0, length, (num_elem + 1))
    for i in range(0, int(num_t_steps/eta)):
        a_temp   = start_a + (start_a_v * eta) + ((1/2 - beta) * start_a_acc * pow(eta, 2))
        a_temp_v = start_a_v + (1 - gamma) * start_a_acc * eta

        left_side  = m_e_newmark + (s_e_newmark * beta * pow(eta, 2))
        right_side = v_e_newmark - (s_e_newmark * a_temp)
        # calculation of acceleration for next time step
        a_acc_p_1 = spsolve(left_side, right_side)
        a_acc_p_1 = csr_matrix(a_acc_p_1)
        a_acc_p_1 = a_acc_p_1.transpose()
        # calculation of position for next time step
        a_p_1     = a_temp + (beta * a_acc_p_1 * pow(eta, 2))
        # calculation of velocity for next time step
        a_v_p_1   = a_temp_v + (gamma * a_acc_p_1 * eta)

        w_indices = np.arange(0, 2 * num_elem + 2, 2, dtype=int)
        w_newmark = a_p_1[w_indices]

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

        # create a picture of the beam for every time step, safe it for the gif creation
        w_new_arr = w_newmark.toarray()
        plt.plot(x_axis_newmark, w_new_arr)
        plt.xlabel(f'x in m mit n = {num_elem}')
        plt.ylabel('\u03C9 in m')
        plt.xlim(0, length)
        plt.ylim(-0.5, 0.5)
        plt.title('Newmark')
        filename = f"temp_{i}.png"
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()

        # assign the start values to the calculated values of the next step
        start_a     = a_p_1
        start_a_v   = a_v_p_1
        start_a_acc = a_acc_p_1

    x_axis_energy = np.linspace(0, num_t_steps, int(num_t_steps / eta), dtype=float)
    helper.get_plot(x_axis_energy, energy_array, 0, 0.03, x_lim1=-2, x_lim2=100, x_label="t in s",
                    y_label="Energie in J",
                    title=f"Energie für \u03B2 = {beta}, \u03B3 = {gamma}, \u03B7 = {eta}")

    with imageio.get_writer('my_gif.gif', mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)


# Aufgabe 14
# a)
newmark(newmark_beta, newmark_gamma, time_increment)
# b)
beta_b = newmark_beta
gamma_b = newmark_gamma
eta_b = 1
newmark(beta_b, gamma_b, eta_b)
# c)
beta_c = 0.5
gamma_c = 1
eta_c = 0.1
newmark(beta_c, gamma_c, eta_c)
# d)
beta_d = 1
gamma_d = 1
eta_d = 1
newmark(beta_d, gamma_d, eta_d)


def w(x):
    return (new_q_load(x)/(new_elast(x) * new_mom_of_inertia(x))) * ((pow(x, 4)/24) - ((length * pow(x, 3))/6) + ((pow(length, 2) * pow(x, 2))/4))


def w_abl(x):
    return (new_q_load(x)/(new_elast(x) * new_mom_of_inertia(x))) * ((pow(x, 3)/6) - ((length * pow(x, 2))/2) + ((pow(length, 2) * x)/2))


def convergence_analysis(n_elem):
    h_l_ = length / n_elem
    b_matrix_ = np.array([[0     , 1, 0],
                          [0     , 2, 0],
                          [n_elem, 3, 0],
                          [n_elem, 4, 0]])
    s_e_ = helper.get_s_e(b_matrix_, new_elast, new_mom_of_inertia, stuetzstellen, length, n_elem, ord_of_quad)
    v_e_ = helper.get_v_e(b_matrix_, new_q_load, stuetzstellen, length, n_elem, ord_of_quad, 1)
    alpha_e_ = spsolve(s_e_, v_e_)

    x_for_w = np.linspace(0, n_elem, n_elem+1)
    x_for_w = x_for_w * h_l_
    w1 = list(map(w, x_for_w))
    w2 = list(map(w_abl, x_for_w))

    w_combined = np.column_stack((w1, w2)).reshape(-1)
    w_combined = csr_matrix(w_combined)

    matrix_m = helper.get_m(new_lenght_specific_mass, stuetzstellen, length, n_elem, ord_of_quad)

    fitting_alpha = csr_matrix(alpha_e_[:(2 * n_elem + 2)])
    fitting_alpha = csr_matrix(fitting_alpha)
    w_minus_alpha = w_combined - fitting_alpha
    w_minus_alpha = w_minus_alpha
    error_l2 = (csr_matrix.sqrt(w_minus_alpha @ matrix_m @ w_minus_alpha.transpose()) /
                csr_matrix.sqrt(w_combined @ matrix_m @ w_combined.transpose()))

    return error_l2.item(0)


n_array = np.arange(1, 1001)

error_array = list(map(convergence_analysis, n_array))

helper.get_plot(n_array, error_array, -0.2, 1, 1, 1000, x_label='n in 1', y_label='error L_2 in 1', title='Aufgabe 13')
helper.get_plot_log(n_array, error_array, -35, 2, 1, 1000, x_label='log n in 1', y_label='log error L_2 in 1', title='Aufgabe 13')

# Aufgabe 20
helper.get_plot_new(b_matrix, new_elast, new_mom_of_inertia, new_q_load, stuetzstellen, eva_array, length, num_elem, num_points_elem, ord_of_quad)
