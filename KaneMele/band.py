import numpy as np
import matplotlib.pyplot as plt


def hamiltonian(k, t_n, t_so, row_count, a):
    h0 = hamiltonian_s(k, 0, t_n, t_so, row_count, a)
    h1 = hamiltonian_s(k, 1, t_n, t_so, row_count, a)
    zero_fill = np.zeros(h0.shape)
    return np.block([[h0, zero_fill], [zero_fill, h1]])

# hamiltonian of a fixed spin
def hamiltonian_s(k, s, t_n, t_so, row_count, a):
    iak    = a * k * 1j
    sign_s = (-1) ** s
    it_so  = t_so * 1j

    atom_a     = 0
    atom_b     = 1
    atom_count = 2

    h = np.zeros((row_count, atom_count, row_count, atom_count), dtype=complex)

    h[0,             atom_a, 0,             atom_b] += t_n
    h[0,             atom_a, 0,             atom_b] += t_n * np.exp(-iak)
    h[row_count - 1, atom_a, row_count - 2, atom_a] += it_so * sign_s * np.exp(iak)
    h[row_count - 1, atom_a, row_count - 1, atom_a] += it_so * sign_s * np.exp(-iak)
    h[row_count - 1, atom_b, row_count - 1, atom_b] += it_so * sign_s * np.exp(iak)
    h[row_count - 1, atom_b, row_count - 2, atom_b] += it_so * sign_s
    h[0,             atom_a, 1,             atom_a] += it_so * sign_s
    h[0,             atom_a, 0,             atom_a] += it_so * sign_s * np.exp(-iak)
    h[0,             atom_b, 0,             atom_b] += it_so * sign_s * np.exp(iak)
    h[0,             atom_b, 1,             atom_b] += it_so * sign_s * np.exp(-iak)

    for i in range(1, row_count):
        h[i, atom_a, i,     atom_b] += t_n
        h[i, atom_a, i - 1, atom_b] += t_n
        h[i, atom_a, i,     atom_b] += t_n * np.exp(-iak)

    for i in range(1, row_count - 1):
        h[i, atom_a, i + 1, atom_a] += it_so * sign_s
        h[i, atom_a, i,     atom_a] += it_so * sign_s * np.exp(-iak)
        h[i, atom_a, i - 1, atom_a] += it_so * sign_s * np.exp(iak)
        h[i, atom_b, i - 1, atom_b] += it_so * sign_s
        h[i, atom_b, i,     atom_b] += it_so * sign_s * np.exp(iak)
        h[i, atom_b, i + 1, atom_b] += it_so * sign_s * np.exp(-iak)

    h = h.reshape(row_count * atom_count, row_count * atom_count)
    return h + h.conj().T

def band_structure(h_k, k_min, k_max, points):
    ks = np.linspace(k_min, k_max, points)
    eigenvalues = list(map(lambda k: np.linalg.eigvalsh(h_k(k)), ks))
    return ks, np.array(eigenvalues)

def plot_band_structure(ks, bands):
    bands_count = len(bands[0])
    for i in range(bands_count):
        plt.plot(ks, bands[:, i])
    plt.ylim([-1, 1])
    plt.show()

if __name__ == "__main__":
    a         = 1
    t_n       = 1
    t_so      = 0.04
    s         = 0
    row_count = 32

    h_k       = lambda k: hamiltonian_s(k, s, t_n, t_so, row_count, a)
    # plot_band_structure(*band_structure(h_k, 0, 2 * np.pi, 500))

    h_k_sum   = lambda k: hamiltonian(k, t_n, t_so, row_count, a)
    plot_band_structure(*band_structure(h_k_sum, 0, 2 * np.pi, 500))
