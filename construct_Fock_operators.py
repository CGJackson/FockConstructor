"""
    A library that contains functions for constructing fermionic creation, annihilation and related operators as
    (sparse) matrices.

    The library assumes a basis ordered such that the occupation number of the nth single particle state (in some single
    particle basis) corresponds to the value of the nth bit in the index of each many-body state (indexes starting from
    0, the vacuum). States are created from the vacuum by applying creation operators in order from the first single
    particle state to the last (that is from least to most significant bit)
"""

import scipy.sparse as spr


def _anticommutation_factor(many_particle_state: int, initial_position: int, final_position: int) -> int:
    """
    Calculate the sign factor needed to move an operator from initial_position to final_position in the sequence
    of creation operators that create many_particle_state from the vacuum, assuming that it anticommutes with all
    creation operators it encounters
    :param many_particle_state:
    :param initial_position:
    :param final_position:
    :return: sign factor
    """
    initial_position, final_position = max(initial_position, final_position), min(initial_position, final_position)
    mask = ~(-1 << (initial_position - final_position))
    intermediate_states = (many_particle_state >> final_position) & mask
    sign_factor = 1
    while intermediate_states:
        if 1 & intermediate_states:
            sign_factor *= -1
        intermediate_states >>= 1
    return sign_factor


def creation(basis_size: int, state_index: int) -> spr.csc_matrix:
    """
    Generates the matrix of the fermionic creation operator for a given single particle state
    :param basis_size: The total number of states in the single particle basis
    :param state_index: The index of the state to be created by the operator
    :return: The matrix of the many-body creation operator (2^basis_size x 2^basis_size sparse matrix)
    """
    many_particle_basis_size = 2**basis_size
    temp_matrix = spr.dok_matrix((many_particle_basis_size, many_particle_basis_size))

    single_particle_state_mask = 1 << (state_index-1)

    for state_to_act_on in range(many_particle_basis_size):
        if ~state_to_act_on & single_particle_state_mask:
            temp_matrix[state_to_act_on | single_particle_state_mask, state_to_act_on] = (
                _anticommutation_factor(state_to_act_on, basis_size, state_index)
            )

    return temp_matrix.tocsc()


def annihilation(basis_size: int, state_index: int) -> spr.csc_matrix:
    """
    Generates the matrix of the fermionic annihilation operator for a given single particle state
    :param basis_size: The total number of states in the single particle basis
    :param state_index: The index of the state to be annihilated by the operator
    :return: The matrix of the many-body annihilation operator (2^basis_size x 2^basis_size sparse matrix)
    """
    return creation(basis_size, state_index).getH()


def transition(basis_size: int, out_state: int, in_state: int) -> spr.csc_matrix:
    """
    Constructs the matrix of the operator creation(basis_size,out_state) * annihilation(basis_size,in_state), which
    transfers a particle from in_state to out_state.
    :param basis_size: The total size of the single particle basis
    :param out_state: The index of the single particle final state
    :param in_state: The index of the single particle initial state
    :return: The matrix of the many particle transition operator (the matrix is size 2^basis_size x 2^basis_size)
    """
    # TODO implement direct construction of matrix to avoid multiplication
    return creation(basis_size, out_state)*annihilation(basis_size, in_state)
