import numpy as np
import qmc
from joblib import Parallel, delayed

import matplotlib.pyplot as plt


def helium_hamiltonian_over_psi(psi, x, a):
    r1 = np.array(x[:3])
    r2 = np.array(x[3:6])
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    r12_norm = np.linalg.norm(r1 - r2)
    
    item1 = -4
    item2 = np.dot((r1 / r1_norm - r2 / r2_norm), r1 - r2) * 1 / (r12_norm * (1 + a * r12_norm) ** 2)
    item3 = -1 / (r12_norm * (1 + a * r12_norm) ** 3)
    item4 = -1 / (4 * (1 + a * r12_norm) ** 4)
    item5 = 1 / r12_norm

    return item1 + item2 + item3 + item4 + item5
    
def helium_psi(x, a):
    r1 = np.array(x[:3])
    r2 = np.array(x[3:6])
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    r12_norm = np.linalg.norm(r1 - r2)
    return np.exp(- 2 * r1_norm - 2 * r2_norm + r12_norm / (2 * (1 + a * r12_norm)))

def helium_qmc(a, steps, delta_t, starting):
    hamiltonian_over_psi = lambda psi, x: helium_hamiltonian_over_psi(psi, x, a)
    psi = lambda x: helium_psi(x, a)
    return qmc.qmc(hamiltonian_over_psi, psi, [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], steps, delta_t, starting)

def helium_qmc_a(a):
    results = Parallel(n_jobs=6)(delayed(lambda i: helium_qmc(a, 30000, 0.3, 1000))(i) for i in range(72))
    results = [result[0] for result in results]
    return results

if __name__ == "__main__":
    a_array = []
    e_array = []
    for a in [0.125, 0.15, 0.175, 0.20, 0.25]:
        results = helium_qmc_a(a)
        print(f"a: {a}, avg: {np.mean(results)}, std: {np.std(results)}")
        a_array.append(a)
        e_array.append(np.mean(results))
    
    plt.plot(a_array, e_array)
    plt.show()
