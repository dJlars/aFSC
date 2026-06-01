r"""
Author:         Lars de Jong
Date:           2024-12-20
Description:    file with classes that make the programming 
                of the adaptive Fourier stochastic collocation
                a bit more structured and convinient to use.
"""

import numpy as np
import chaospy as cp

from aFSC.utils.daliFile import new_lj_1d, Hierarchical1d


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
            """

class uncertainDimension:
    r"""
    Class that represents an uncertain dimension in the 
    stochastic model. It contains the distribution, the 
    knots and weights of the Leja sequence, the polynomials
    and the current order of the dimension.
    """
    
    def __init__(self,
                 dist: cp.Distribution):
        
        self.dist = dist
        self.knots, self.weights = new_lj_1d(0, self.dist)
        self.polynomials = []
        self.addPolynomial(0)
        self._order = 0

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if value < 0:
            raise ValueError("Order must be a positive integer.")
        for i in range(self.order+1, value+1):
            self.addPolynomial(i)
        self._order = value

    # adapted from Loukretzis' dali
    @staticmethod
    def getLagrange(knots):
        knots = np.array(knots)
        lagrange = Hierarchical1d(knots)
        return lagrange


    def getKnot(self,
                order: int):
        r"""
        Returns the knot and weight for a given order.

        Parameters
        ----------
        order : int
            The order of the knot and weight to be returned.
        
        Returns
        ----------
        knot : np.ndarray
            The knot for the given order.
        """
        if order > self.order:
            if order-self.order > 1:
                raise ValueError("Order must be increased by one.")
            newKnot, newWeights = new_lj_1d(order, self.dist, self.knots)
            self.knots = np.append(self.knots,newKnot)
            self.weights = np.append(self.weights, newWeights)
            self.order = order
        
        return self.knots[order]

    def addPolynomial(self,
                      order):
        r"""
        Creates the hierarchical Lagrange polynomial for a given order.

        Parameters
        ----------
        order : int
            The order of the polynomial to be created.
        """
        self.polynomials.append(self.getLagrange(self.knots[:order+1]))
