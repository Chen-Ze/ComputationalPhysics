import numpy as np
from functools import reduce
from numpy import linalg as LA


def hamiltonian(spin_dimension, chain_length, coupling_j):
    columns = list(map(lambda encoded_state: hamiltonian_column(encoded_state, spin_dimension, chain_length), range(0, spin_dimension ** chain_length)))
    hamiltonian_matrix = coupling_j * np.column_stack(columns)
    return hamiltonian_matrix

def hamiltonian_column(encoded_state, spin_dimension, chain_length):
    decoded_state       = decode_state(encoded_state, spin_dimension, chain_length)
    decoded_state_shift = np.roll(decoded_state, -1)
    # decoded_state_zip = [ (s0, s1), (s1, s2), ..., (s[n-1], s0) ]
    decoded_state_zip   = list(zip(decoded_state, decoded_state_shift))
    item1_columns = map(lambda index_pair: hamiltonian_item1(*index_pair, decoded_state, spin_dimension, chain_length), enumerate(decoded_state_zip))
    item2_columns = map(lambda index_pair: hamiltonian_item2(*index_pair, decoded_state, spin_dimension, chain_length), enumerate(decoded_state_zip))
    item3_columns = map(lambda index_pair: hamiltonian_item3(*index_pair, decoded_state, spin_dimension, chain_length), enumerate(decoded_state_zip))
    item1_column  = sum(item1_columns)
    item2_column  = sum(item2_columns)
    item3_column  = sum(item3_columns)
    return item1_column + item2_column + item3_column

def hamiltonian_item1(index, pair, decoded_state, spin_dimension, chain_length):
    s  = (spin_dimension - 1) / 2
    m0 = pair[0] - s
    m1 = pair[1] - s
    raise_m0_factor = np.sqrt(s * (s + 1) - m0 * (m0 + 1))
    lower_m1_factor = np.sqrt(s * (s + 1) - m1 * (m1 - 1))
    overall_factor  = raise_m0_factor * lower_m1_factor / 2

    new_state = decoded_state[:]
    new_state[index % len(new_state)]       = min(spin_dimension - 1, pair[0] + 1)
    new_state[(index + 1) % len(new_state)] = max(0, pair[1] - 1)

    column = np.zeros(spin_dimension ** chain_length)
    column[encode_state(new_state, spin_dimension)] = overall_factor
    return column

def hamiltonian_item2(index, pair, decoded_state, spin_dimension, chain_length):
    s  = (spin_dimension - 1) / 2
    m0 = pair[0] - s
    m1 = pair[1] - s
    lower_m0_factor = np.sqrt(s * (s + 1) - m0 * (m0 - 1))
    raise_m1_factor = np.sqrt(s * (s + 1) - m1 * (m1 + 1))
    overall_factor  = lower_m0_factor * raise_m1_factor / 2

    new_state = decoded_state[:]
    new_state[index % len(new_state)]       = max(0, pair[0] - 1)
    new_state[(index + 1) % len(new_state)] = min(spin_dimension - 1, pair[1] + 1)

    column = np.zeros(spin_dimension ** chain_length)
    column[encode_state(new_state, spin_dimension)] = overall_factor
    return column

def hamiltonian_item3(index, pair, decoded_state, spin_dimension, chain_length):
    s  = (spin_dimension - 1) / 2
    m0 = pair[0] - s
    m1 = pair[1] - s
    overall_factor = m0 * m1

    new_state = decoded_state[:]

    column = np.zeros(spin_dimension ** chain_length)
    column[encode_state(new_state, spin_dimension)] = overall_factor
    return column

def decode_number_iterator(num, base):
    if num == 0:
        yield 0
    else:
        yield num % base
        yield from decode_number_iterator(num // base, base)

# state_vector[0] + state_vector[1] * spin_dimension + state_vector[2] * (spin_dimension ** 2) + ...
def encode_state(state_vector, spin_dimension):
    return reduce((lambda higher_value, this_value: higher_value * spin_dimension + this_value), reversed(state_vector))

def decode_state(encoded_state, spin_dimension, chain_length):
    digits = np.array(list(decode_number_iterator(encoded_state, spin_dimension)))
    digits.resize(chain_length, refcheck=False)
    return list(digits)


if __name__ == "__main__":
    h = hamiltonian(2, 12, 1)
    print("Hamiltonian obtained.")
    print(sorted(LA.eigh(h)[0]))
