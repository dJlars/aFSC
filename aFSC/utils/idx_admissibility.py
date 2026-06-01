# -*- coding: utf-8 -*-
"""
Disclaimer: This file contains code from the
            gitHub reopository https://github.com/dlouk/DALI3
            created by Dimitrios Loukrezis. 
            It belongs to the publication:
            Dimitrios Loukrezis, Ulrich Römer, and Herbert De Gersem. 
            Assessing the Performance of Leja and Clenshaw-Curtis 
            Collocation for Computational Electromagnetics with 
            Random Input Data. 
            International Journal for Uncertainty Quantification, 
            9, January 2018. 
            doi: 10.1615/Int.J.UncertaintyQuantification.2018025234.

Created on Mon Apr 03 11:30:20 2017

@author: Dimitris Loukrezis

Index forward/backward neighbors and admissibility for monotone sets.
"""

import numpy as np


def admissible_neighbors(index, index_set):
    """Given an index and a monotone index set, find admissible neighboring
    indices"""
    for_neighbors = forward_neighbors(index)
    # find admissible neighbors
    for_truefalse = [is_admissible(fn, index_set) for fn in for_neighbors]
    adm_neighbors = np.array(for_neighbors)[for_truefalse].tolist()
    return adm_neighbors


def is_admissible(index, index_set):
    """Given an index and a monotone index set, check index admissibility"""
    back_neighbors = backward_neighbors(index)
    for ind_b in back_neighbors:
        if ind_b not in index_set:
            return False
    return True


def forward_neighbors(index):
    """Given a multiindex, return its forward neighbors as a list of
    multiindices, e.g. (2,1) --> (3,1), (2,2)"""
    N = len(index)
    for_neighbors = []
    for i in range(N):
        index_tmp = index[:]
        index_tmp[i] = index_tmp[i] + 1
        for_neighbors.append(index_tmp)
    return for_neighbors


def backward_neighbors(index):
    """Given a multiindex, return its backward neighbors as a list of
    multiindices, e.g. (2,2) --> (1,2), (2,1)"""
    N = len(index)
    back_neighbors = []
    for i in range(N):
        index_tmp = index[:]
        if index_tmp[i] > 0:
            index_tmp[i] = index_tmp[i] - 1
            back_neighbors.append(index_tmp)
    return back_neighbors



# added by Lars de Jong

def admissible_neighborhood(index_set: list,
                            active_set: list, 
                            adm_set: list = []):
    """Given a monotone index set, an active set and an admissible set,
    return the admissible neighborhood of the index set.

    Parameters
    ----------
    index_set : list
        A monotone index set.
    active_set : list
        A set of active indices.
    adm_set : list, optional
        A set of admissible indices, by default an empty list.

    Returns
    ----------
    adm_set : list
        The updated admissible set.
    active_set : list
        The updated active set.
    add_adm_set : list
        A list of indices that were added to the admissible set.
    """

    # check which index of adm_set is in index_set
    for idx in adm_set:
        if idx in index_set:
            adm_set.remove(idx)

    add_adm_set = []
    for activeIdx in active_set:
        temp_adm_set = forward_neighbors(activeIdx)

        for tempIdx in temp_adm_set:
            if tempIdx not in index_set and \
                tempIdx not in adm_set and \
                is_admissible(tempIdx, index_set):
                adm_set.append(tempIdx)
                add_adm_set.append(tempIdx)

    # make new index and check if admissible
    for admIdx in adm_set:
        if not is_admissible(admIdx, index_set):
            raise ValueError("Admissible index is not admissible.")

    return adm_set, active_set, add_adm_set


def update_active_set(active_set: list):
    """Update the active set by removing indices that are no longer admissible.
    Parameters
    ----------
    active_set : list
        A set of active indices.

    Returns
    ----------
    active_set : list
        The updated active set.
    """
    
    old_active_set = active_set.copy()
    for activeIdx in old_active_set:
        temp_adm_set = forward_neighbors(activeIdx)

        if all(idx in old_active_set for idx in temp_adm_set):
            active_set.remove(activeIdx)

    return active_set

