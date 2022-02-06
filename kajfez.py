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
    # print(" Calculating the C matrix...", end="\r")
    C = make_C(e, weights)
    # print(" Calculating the g matrix...", end="\r")
    g = make_g(gammas, e, weights)
    # print("Calculating the inverse of C...", end="\r")
    C_inv = np.linalg.inv(C)
    # print(" Calculating the fit parameters...", end="\r")
    fit_params = np.matmul(C_inv, g)
    # print(" Calculating the errors...", end="\r")
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

def tan_equations(v, Gamma_d, center_Gamma, radius_Gamma):
    r, x03, y03 = v
    z1 = x03 + y03 * 1j
    eq1 = np.absolute(z1 - center) - (r - radius_Gamma)
    eq2 = np.absolute(z1) - (1 - r)
    eq3 = np.absolute(Gamma_d - z1) - r
    return eq1, eq2, eq3

def find_tangent_circle(Gamma_d, Gamma_l, center, radius):
    Gamma_s = Gamma_d
    cosPhi = (np.abs(Gamma_s)**2 + 4*radius ** 2 - np.abs(Gamma_l) ** 2)/(4 * radius * np.abs(Gamma_s))

    radius_tan = (1 - np.absolute(Gamma_s)**2)/(1 - np.absolute(Gamma_s) * cosPhi)/2.0

    m = (np.real(Gamma_s) - np.real(center))/(np.imag(Gamma_s) - np.imag(center))
    y03_minus = np.imag(Gamma_s) - radius_tan/np.sqrt(m**2 + 1)
    x03_minus = np.real(Gamma_s) - m * (np.imag(Gamma_s) - y03_minus)

    y03_plus = np.imag(Gamma_s) + radius_tan/np.sqrt(m**2 + 1)
    x03_plus = np.real(Gamma_s) - m * (np.imag(Gamma_s) - y03_plus)

    if(np.absolute(x03_minus + y03_minus * 1j) < np.absolute(x03_plus + y03_plus * 1j)):
        center_tan = x03_minus + 1j * y03_minus
    else:
        center_tan = x03_plus + 1j * y03_plus
    return center_tan, radius_tan

def delta(freq, freq0):
    return freq / freq0 - np.true_divide(freq0,freq)

def S11dB(freqs, freq0, Q0, beta, zero_level):
    deltas = delta(freqs, freq0);
    numerator = beta - 1 - 1j * Q0 * deltas
    numitor = beta + 1 + 1j * Q0 * deltas
    return 10*np.log10(np.absolute(np.true_divide(numerator,numitor))) + zero_level

def gamma_to_S11dB(gammas):
    return 10*np.log10(np.absolute(gammas))


def fit_kajfez(file, header=0, footer=0, every=1,comments='!', delimiter=',',\
 n_steps=3, plot_every_step=False, save_plots=False, out_folder='', out_format='png', hide_plots = False, quiet = False):
