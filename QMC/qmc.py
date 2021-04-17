from scipy.misc import derivative 
from scipy.optimize import minimize
import numpy as np
import time
from joblib import Parallel, delayed

import matplotlib.pyplot as plt


# todo: point copy to rows + x * eyes(n), then derivative
def gradient(func, point, dx=1e-6):
    func_slices = map(lambda i: (lambda x: func([*point[:i], x + point[i], *point[i + 1:]])), range(0, len(point)))
    partial_derivatives = map(lambda func_slice: derivative(func_slice, 0, dx), func_slices)
    return list(partial_derivatives)

def qmc(hamiltonian_over_psi, psi, initial, steps, delta_t, starting):
    x = np.array(initial)
    average     = 0
    steps_count = 0
    accepts     = 0
    for i in range(0, steps):
        log_psi_gradient = np.array(gradient(psi, x)) / psi(x)
        x_new = x + log_psi_gradient * delta_t + np.random.normal(0, np.sqrt(delta_t), len(x))
        acceptance_flag = np.random.binomial(1, acceptance_rate(psi, x_new, x, delta_t))
        if acceptance_flag > 0:
            x = x_new
            accepts += 1
        if i < starting:
            continue
        integrand = hamiltonian_over_psi(psi, x)
        steps_count += 1
        average = average * (steps_count - 1) / steps_count + (integrand) / steps_count
    return average, accepts / steps

def acceptance_rate(psi, y, x, delta_t):
    log_psi_gradient_x = np.array(gradient(psi, x)) / psi(x)
    log_psi_gradient_y = np.array(gradient(psi, y)) / psi(y)
    def gaussian(_y, _x):
        _log_psi_gradient_x = np.array(gradient(psi, _x)) / psi(_x)
        return np.exp(-1 / (2 * delta_t) * np.dot(_y - _x - _log_psi_gradient_x * delta_t, _y - _x - _log_psi_gradient_x * delta_t))
    gaussian_factor = gaussian(x, y) / gaussian(y, x)
    # gaussian_factor = np.exp(0.5 * np.dot((log_psi_gradient_x + log_psi_gradient_y), 2 * x - 2 * y + delta_t * (log_psi_gradient_x - log_psi_gradient_y)))
    psi_factor = (np.abs(psi(y)) ** 2) / (np.abs(psi(x)) ** 2)
    return min(1, gaussian_factor * psi_factor)

def sho_hamiltonian_over_psi(psi, x, a):
    x = x[0]
    return (a + (a ** 2 - 1) * (x ** 2)) / (2 * a ** 2)

def sho_psi(x, a):
    x = x[0]
    return np.exp(- x ** 2 / (2 * a))

def sho_qmc(a, steps, delta_t, starting):
    hamiltonian_over_psi = lambda psi, x: sho_hamiltonian_over_psi(psi, x, a)
    psi = lambda x: sho_psi(x, a)
    return qmc(hamiltonian_over_psi, psi, [0], steps, delta_t, starting)

def sho_qmc_a(a):
    results = Parallel(n_jobs=6)(delayed(lambda i: sho_qmc(a, 10000, 3, 1000))(i) for i in range(12))
    results = [result[0] for result in results]
    return results

if __name__ == "__main__":
    # print(gradient(f, [1, 2, 3, 4]))
    a_array = []
    e_array = []
    for a in [0.8, 0.9, 1.0, 1.1, 1.2]:
        results = sho_qmc_a(a)
        print(f"a: {a}, avg: {np.mean(results)}, std: {np.std(results)}")
        a_array.append(a)
        e_array.append(np.mean(results))
    
    plt.plot(a_array, e_array)
    plt.show()
        
