r"""
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

Author:        Dimitrios Loukrezis 
Date:          2024-12-20
Description:   functions and classes of the DALI framework

"""

import numpy as np
import chaospy as cp

def new_lj_1d(order, dist=cp.Uniform(-1, 1), old_knots=[]):
    """New Leja interpolation knots and weights in 1d."""
    knots, weights = cp.quadrature.leja(order, dist)
    knots = knots.flatten()
    is_in = np.in1d(knots, old_knots, invert=True)
    new_knots = knots[is_in]
    return new_knots, weights

class Lagrange1d:
    """Univariate Lagrange nodal basis polynomial"""
    def __init__(self, current_knot, knots):
        self.current_knot = current_knot
        self.knots = np.array(knots)
        self.other_knots = np.setdiff1d(knots, current_knot)
        # compute denominator once and re-use
        self.denoms_prod = (self.current_knot - self.other_knots).prod()

    def evaluate(self, non_grid_knots):
        """Evaluate polynomial on specific non-grid knots"""
        non_grid_knots = np.array(non_grid_knots).flatten()
        L = list(map(lambda x: np.prod(x-self.other_knots)/self.denoms_prod,
                 non_grid_knots))
        return L


class Hierarchical1d(Lagrange1d):
    """Univariate Lagrange hierarchical basis polynomial"""
    def __init__(self, knots):
        self.knots = np.array(knots)
        self.current_knot = self.knots[-1]
        self.other_knots = self.knots[:-1]
        # compute denominator once and re-use
        self.denoms_prod = (self.current_knot - self.other_knots).prod()
