#!/usr/bin/env python3
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import fsolve
import sys
import os
import argparse
sys.path.append("smithplot")
from smithplot import SmithAxes
# import warnings
# warnings.filterwarnings('ignore', 'The iteration is not making good progress')

def gamma_fit_function(t, a):
    return np.true_divide(a[0] * t + a[1], a[2] * t + 1)

def abs_gamma_fit_function(t,a):
    return np.absolute(gamma(t, a))

def z_to_Gamma(z):
    return np.true_divide(z-1,z+1)

# Fit using Gamma = (a1 * t + a2) / (a3 * t + 1)
def fit_gamma(ts, gammas, weights):
    shape = np.shape(ts)
    e = np.empty([shape[0],3], dtype = 'complex_')
    e[:,0] = ts
    e[:,1] = np.ones(shape)
    e[:,2] = -ts * gammas
    # P = np.diag(weights)
    print(" Calculating the C matrix...", end="\r")
    C = make_C(e, weights)
    print(" Calculating the g matrix...", end="\r")
    g = make_g(gammas, e, weights)
    print("Calculating the inverse of C...", end="\r")
    C_inv = np.linalg.inv(C)
    print(" Calculating the fit parameters...", end="\r")
    fit_params = np.matmul(C_inv, g)
    print(" Calculating the errors...", end="\r")
    eps = fit_params[0] * ts + fit_params[1] - fit_params[2] * ts * gammas - gammas
    eps_abs_sq = eps * np.conjugate(eps)
    S_sq = np.dot(weights, eps_abs_sq)
    sigma_sq = np.empty([3,1])
    denominator = C[0,0] * C_inv[0,0] + C[1,1] * C_inv[1,1] + C[2,2] * C_inv[2,2]
    for i in range(0, 3):
        sigma_sq[i] =  np.real(C_inv[i,i] * S_sq / denominator)
    return sigma_sq[:,0], fit_params[:,0]

def make_C(e, P):
    shape = np.shape(e)
    C = np.empty([shape[1],shape[1]],dtype = 'complex_')
    for i in range(0, shape[1]):
        for j in range(0, shape[1]):
            projected_e_j = P*e[:,j]
            C[i,j] = np.matmul(np.matrix.conjugate(e[:,i]), projected_e_j)
    return C

def make_g(gammas, e, P):
    shape = np.shape(e)
    g = np.empty([shape[1],1],dtype = 'complex_')
    for i in range(0, shape[1]):
        projected_gammas = P*gammas
        g[i,0] = np.matmul(np.matrix.conjugate(e[:,i]), projected_gammas)
    return g

# Calculates the center and radius of a circle given 3 points in the complex plane
def circle_from_3_points(z1, z2, z3):
    x1 = np.real(z1);
    y1 = np.imag(z1);
    x2 = np.real(z2);
    y2 = np.imag(z2);
    x3 = np.real(z3);
    y3 = np.imag(z3);
    c = (x1-x2)**2 + (y1-y2)**2
    a = (x2-x3)**2 + (y2-y3)**2
    b = (x3-x1)**2 + (y3-y1)**2
    s = 2*(a*b + b*c + c*a) - (a*a + b*b + c*c)
    px = (a*(b+c-a)*x1 + b*(c+a-b)*x2 + c*(a+b-c)*x3) / s
    py = (a*(b+c-a)*y1 + b*(c+a-b)*y2 + c*(a+b-c)*y3) / s
    ar = a**0.5
    br = b**0.5
    cr = c**0.5
    r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5
    return (px + 1j * py), r

def calculate_t_intersection(params, z1, center):
    z = z1-center
    # z = z1
    # alpha = np.arctan2(np.imag(z),np.real(z))
    gamma_int = z1 - 2 *z
    t_int = np.real((gamma_int - params[1])/(params[0]-gamma_int*params[2]))
    return t_int, gamma_int

# Calculates the t where the circle defined by params intersects the line between the center point and the z point
# def calculate_t_intersection(params, z1, center):
#     z = z1-center
#     temp1 = np.imag(z) * np.real(params[2]) + np.real(z) * np.imag(params[2])
#     temp2 = np.real(z) * np.real(params[2]) - np.imag(params[2]) * np.imag(z)
#     a = np.real(params[0]-params[2]*center) * temp1 - np.imag(params[0]-params[2]*center) * temp2
#     b = np.real(params[1]-center) * temp1 - np.imag(params[1]-center) * temp2 + np.imag(z) * np.real(params[0]-params[2]*center) - np.real(z) * np.imag(params[0]-params[2]*center)
#     c = np.real(params[1]-center) * np.imag(z) - np.imag(params[1]-center) * np.real(z)
#     t_int = -c/b
#     gamma_int = gamma_fit_function(t_int, params)
#     return t_int, gamma_int

def tan_equations(v, Gamma_d, center_Gamma, radius_Gamma):
    r, x03, y03 = v
    z1 = x03 + y03 * 1j
    eq1 = np.absolute(z1 - center) - (r - radius_Gamma)
    eq2 = np.absolute(z1) - (1 - r)
    eq3 = np.absolute(Gamma_d - z1) - r
    return eq1, eq2, eq3

def find_tangent_circle(Gamma_d, center, radius):
    x02 = np.real(center)
    y02 = np.imag(center)
    xint = np.real(Gamma_d)
    yint = np.imag(Gamma_d)
    Ax = - (yint - y02 - radius * yint)/(x02 * yint - xint * y02)
    Ay = - (xint - x02 - radius * xint)/(y02 * xint - yint * x02)
    Bx = 0.5*((-radius**2 + x02**2 + y02**2 + 1) * yint - y02 * (1 + xint**2 + yint**2))/(x02 * yint - xint * y02)
    By = 0.5*((-radius**2 + x02**2 + y02**2 + 1) * xint - x02 * (1 + xint**2 + yint**2))/(y02 * xint - yint * x02)

    a = Ax**2 + Ay**2 - 1
    b = 2 * (Ax * Bx + Ay * By + 1)
    c = (Bx**2 + By**2 - 1)
    delta = b**2 - 4 * a * c
    if (np.abs(delta) < 1E10):    # check so we don't have small negative values of delta
        radius_tan = -b/(2 * a)
    else:
        radius_tan = (-b + np.sqrt(b**2 - 4 * a * c))/(2 * a)
    x03 = Ax * radius_tan + Bx;
    y03 = Ay * radius_tan + By;
    center_tan = x03 + 1j * y03
    return center_tan, radius_tan

# def find_tangent_circle(Gamma_d, Gamma_l, center, radius):
#     # alpha = np.arctan2(np.imag(center),np.real(center))
#     # Gamma_s = center + radius  * (np.cos(alpha) + 1j * np.sin(alpha))
#     # Gamma_l = center - radius  * (np.cos(alpha) + 1j * np.sin(alpha))
#     Gamma_s = Gamma_d
#
#     # Gamma_s = gammas[np.argmax(np.abs(gammas))]
#     Phi = (np.real(Gamma_s) * np.real(Gamma_s - Gamma_l) + np.imag(Gamma_s) * np.imag(Gamma_s - Gamma_l)) / (np.absolute(Gamma_s) * np.absolute(Gamma_s - Gamma_l))
#     radius_tan = (1 - np.absolute(Gamma_s)**2)/(1 - np.absolute(Gamma_s) * Phi)/2.0
#     m = (np.real(Gamma_s) - np.real(center))/(np.imag(Gamma_s) - np.imag(center))
#     y03_minus = np.imag(Gamma_s) - radius_tan/np.sqrt(m**2 + 1)
#     x03_minus = np.real(Gamma_s) - m * (np.imag(Gamma_s) - y03_minus)
#
#     y03_plus = np.imag(Gamma_s) + radius_tan/np.sqrt(m**2 + 1)
#     x03_plus = np.real(Gamma_s) - m * (np.imag(Gamma_s) - y03_plus)
#
#     if(np.absolute(x03_minus + y03_minus * 1j) < np.absolute(x03_plus + y03_plus * 1j)):
#         center_tan = x03_minus + 1j * y03_minus
#     else:
#         center_tan = x03_plus + 1j * y03_plus
#     return center_tan, radius_tan

def delta(freq, freq0):
    return freq / freq0 - np.true_divide(freq0,freq)

def S11dB(freqs, freq0, Q0, beta, zero_level):
    deltas = delta(freqs, freq0);
    numerator = beta - 1 - 1j * Q0 * deltas
    numitor = beta + 1 + 1j * Q0 * deltas
    return 10*np.log10(np.absolute(np.true_divide(numerator,numitor))) + zero_level

def gamma_to_S11dB(gammas):
    return 10*np.log10(np.absolute(gammas))

# Plots the Smith chart
def plot_step(gammas, params_Gamma, Gamma_d, Gamma_l, center, radius, center_tan, radius_tan):
    thetas = np.linspace(0, 2*np.pi, num = 1000)
    circle_zs = center + radius * np.exp(1j * thetas)
    line_t = np.linspace(0,1,num = 1000)
    line1_zs = (1 - line_t) * Gamma_l + Gamma_d * line_t
    line2_zs = (1 - line_t) * Gamma_l + 0 * line_t
    circle_tan_zs = center_tan + radius_tan * np.exp(1j * thetas)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1, projection='smith')
    ax.set_aspect('equal', adjustable='box')
    plt.plot(gammas, datatype=SmithAxes.S_PARAMETER,  marker='o',color='k',linewidth=2, linestyle='None',markersize=8, label = 'Data points')
    plt.plot(circle_zs, datatype=SmithAxes.S_PARAMETER,  marker='None',color='red',linewidth=2, linestyle='-', label = 'Fit circle')
    plt.plot([center], datatype=SmithAxes.S_PARAMETER,  marker='o',color='red',linewidth=2, linestyle='None',markersize=10, label="Center of fit circle")
    plt.plot(line1_zs, datatype=SmithAxes.S_PARAMETER,  marker='None',color='k',linewidth=2, linestyle='-')
    plt.plot(line2_zs, datatype=SmithAxes.S_PARAMETER,  marker='None',color='k',linewidth=2, linestyle='-')
    plt.plot(circle_tan_zs, datatype=SmithAxes.S_PARAMETER,  marker='None',color='forestgreen',linewidth=2, linestyle='-.', label = 'Tangent Circle')

    plt.plot([Gamma_d], datatype=SmithAxes.S_PARAMETER,  marker='o',color='forestgreen',linewidth=2, linestyle='None',markersize=10, label="$\Gamma_d$")
    plt.plot([Gamma_l], datatype=SmithAxes.S_PARAMETER,  marker='o',color='darkorange',linewidth=2, linestyle='None',markersize=10, label="$\Gamma_l$")
    ax.legend(bbox_to_anchor=(1.08, 1.08))
    plt.show()
    plt.pause(0.001)

def get_data_from_file(file, header, footer, comments, delimiter):
    data = np.genfromtxt(open(file,'rt').readlines()[::every], comments = comments, delimiter = delimiter, skip_header = header, skip_footer = footer)
    # if footer > 0:
    #     data =  np.loadtxt(open(file,'rt').readlines()[header:-footer:every], comments=comments,delimiter=delimiter)
    # else:
    #     data =  np.loadtxt(open(file,'rt').readlines()[header::every], comments=comments,delimiter=delimiter)
    gammas = data[:, 1] + data[:, 2] * 1j
    freqs = data[:, 0]
    return freqs, gammas


def fit_and_display(file, header=0, footer=0, every=1,comments='!', delimiter=',', n_steps=3, plot_every_step=False, out_folder='', out_format='png'):
    head, tail = os.path.split(file) #find file name

    plt.ion()
    freqs, gammas = get_data_from_file(file, header, footer, comments, delimiter)
    gamma_abs_sq = np.real(gammas)*np.real(gammas) + np.imag(gammas) * np.imag(gammas)

    #First guess for f_L
    S11dBs = gamma_to_S11dB(gammas)
    f_L = freqs[np.argmin(S11dBs)]

    ts = 2 * (freqs - f_L)/f_L
    ps = 1.0 / (1.0 + ts ** 2 * (1 + gamma_abs_sq))

    # alpha = np.radians(50)
    # dist_from_center = 0.98
    # gammas = gamma_fit_function(ts, [-1000 * dist_from_center*(np.cos(alpha) + 1j* np.sin(alpha)), -dist_from_center*(np.cos(alpha) + 1j* np.sin(alpha))+(np.cos(alpha) + 1j* np.sin(alpha))+0.1, 1000j])
    # gamma_abs_sq = np.real(gammas)*np.real(gammas) + np.imag(gammas) * np.imag(gammas)

    for i in range(0,n_steps):
        print(f"Step {i+1}/{n_steps}")
        sigma_sq, params_Gamma = fit_gamma(ts, gammas, ps)
        Gamma_d = params_Gamma[0]/params_Gamma[2]
        Gamma_l = params_Gamma[1]

        # Get the circle described by the fitted Parameters
        center, radius = circle_from_3_points(Gamma_d, Gamma_l, gamma_fit_function(1, params_Gamma))
        t_int, Gamma_int = calculate_t_intersection(params_Gamma, Gamma_d, center)

        # Calculate the new frequency
        f_L = (1 + 0.5 * t_int)*f_L

        # Calculate the new t's
        ts = 2 * (freqs - f_L)/f_L

        # Calculate the new weights
        ps = 1.0 / (ts**2 * sigma_sq[0] + sigma_sq[1] + ts**2 * gamma_abs_sq * sigma_sq[2])

        # Get the circle tangent to both the unit circle and the fitted circle
        center_tan, radius_tan = find_tangent_circle(Gamma_d, center, radius)

        # Calculate and show the quality factors and kappa
        Q_L = np.imag(params_Gamma[2])
        sigma_Q_L = np.sqrt(sigma_sq[2])
        d = np.absolute(Gamma_l - Gamma_d)
        sigma_d_sq = sigma_sq[0]/np.absolute(params_Gamma[2])**2 + sigma_sq[1] + sigma_sq[2] * np.absolute(params_Gamma[0]/params_Gamma[2]**2)**2
        kappa = 1.0 / (2.0 * radius_tan/d - 1)
        sigma_kappa = 2.0 * radius_tan * np.sqrt(sigma_d_sq) / (2.0 * radius_tan - d)**2
        Q_0 = Q_L * (1 + kappa)
        sigma_Q_0 = np.sqrt(sigma_Q_L**2 * (1 + kappa)**2 + Q_L**2 * sigma_kappa ** 2)
        kappa_s = 1.0/radius_tan-1

        print(" Done. The results from this step:\r")
        print(f" Q_L = {Q_L:.6f} +/- {sigma_Q_L:.6f}")
        # print(f" d = {d:.3f} +/- {np.sqrt(sigma_d_sq):.4f}")
        print(f" k = {kappa:.6f} +/- {sigma_kappa:.6f}")
        print(f" Q_0 = {Q_0:.6f} +/- {sigma_Q_0:.6f}")
        print(f" f0 = {f_L*1E-9:.8f} GHz")
        print(f" ks = {kappa_s:.6f}")
        print()
        if (plot_every_step or i + 1 == n_steps):
            plot_step(gammas, params_Gamma, Gamma_d, Gamma_l, center, radius, center_tan, radius_tan-0.00001)
            fig_name = os.path.join(out_folder, f"smithfit_{tail}_step{i+1}.{out_format}")
            if(save_plots):
                plt.savefig(fig_name, dpi = 100)
            if(i + 1 == n_steps):
                plt.show(block=True)

if __name__ == "__main__":
    # file = "data/AutoSave8.csv"
    # header = 19
    # footer = 1
    # every = 100
    # comments = '!'
    # delimiter = ','


    file = "data/45d_1.s1p"
    header = 0
    footer = 0
    every = 1
    comments = '!'
    delimiter = ' '


    n_steps = 3
    plot_every_step = False
    save_plots = False
    out_folder=''
    out_format='png'

    parser = argparse.ArgumentParser()
    # parser.add_argument("fname", nargs = '?', type=str, help="File Name", default = file)
    parser.add_argument("fname", type=str, help="File Name")
    parser.add_argument("-steps", nargs = '?', type=int, help="Number of steps for the fit", default=n_steps)
    parser.add_argument("-every", nargs = '?', metavar='N', type=int, help="Read every N lines from file", default=every)
    parser.add_argument("-delimiter",  type=str, help="The string used to separate values", default=delimiter)
    parser.add_argument("-comments",  type=str, help="The characters or list of characters used to indicate the start of a comment", default=comments)
    parser.add_argument("-header",  type=int, help="The number of lines to skip at the beginning of the file.", default=header)
    parser.add_argument("-footer",  type=int, help="The number of lines to skip at the end of the file.", default=footer)
    parser.add_argument("--save-plots", action='store_true', help="If present, the software saves the plots as image files")
    parser.add_argument("-ofolder",  type=str, help="Output folder", default=out_folder)
    parser.add_argument("-oformat",  type=str, help="Output format (.png, .jpg, .svg, .pdf etc.)", default=out_format)
    parser.add_argument("--plot-steps", action='store_true', help="Plot every fit iteration")

    args = parser.parse_args()
    file = args.fname
    n_steps = args.steps
    every = args.every
    delimiter = args.delimiter
    comments = args.comments
    header = args.header
    footer = args.footer
    out_folder = args.ofolder
    out_format = args.oformat
    plot_every_step=args.plot_steps
    save_plots = args.save_plots
    fit_and_display(file, header=header, footer=footer, every=every,\
    comments=comments, delimiter=delimiter, n_steps=n_steps, plot_every_step=plot_every_step,\
     out_folder=out_folder, out_format=out_format)
