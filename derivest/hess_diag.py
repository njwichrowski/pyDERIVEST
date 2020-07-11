# -*- coding: utf-8 -*-

from .derivest import derivest
from .utils import replace_element
import numpy as np

def hess_diag(fun, x, par = None, **kwargs):
    """
    Compute diagonal elements of the Hessian matrix of a scalar function.
    
    Uses derivest to provide estimates of both the second derivatives and the
    corresponding error. This method is more efficient than hessian if only the
    diagonal elements of the Hessian matrix are needed.
    
    Arguments:
        fun : Callable object with signature fun(x, *args) -> float, with x the
              (vector) argument, and args an optional list of parameters.
        
        x : Vector location at which to compute the gradient. If x has more
            than one axis, then it is flattened and fun is assumed to be a
            function of np.prod(x.shape) variables.
        
        par : Optional list of parameters to be passed to fun as fun(x, *par).
              If par is not provided, then fun(x, *[]) is equivalent to fun(x),
              in which case fun may have signature fun(x) -> float.
        
        Additional keyword arguments are passed to derivest.
    
    
    Returns a 3-tuple containing flattened arrays of:
        hess : Estimate of the second partial derivatives of fun at location x.
        
        err : Error estimates of the second partial derivatives.
        
        final_delta : The final overall stepsize chosen for each derivative.
    
    
    Example:
      >>>  import derivest, numpy as np
      >>>  f = lambda x: x[0] + x[1]**2.0 + x[2]**3.0
      >>>  (HD, err, fd) = derivest.hess_diag(f, [1, 2, 3])
      >>>  HD
      Out: array([ 0.,  2., 18.])
      >>>  err
      Out: array([0.00000000e+00, 1.02127764e-11, 5.66829319e-12])
    """
    ##### PROCESS ARGUMENTS AND CHECK FOR VALIDITY #####
    if kwargs.pop("deriv_order", 2) != 2:
        raise ValueError("hess_diag() can only perform "
                         "second-order differentiation.")
    if kwargs.pop("vectorized", False):
        raise ValueError("hess_diag() is incompatible with "
                         "vectorized evaluation.")
    kwargs["deriv_order"] = 2 # Force second-order differentiation
    kwargs["vectorized"] = False # and non-vectorized evaluation.
    if isinstance(x, list):
        x = np.array(x)
    if par is None:
        par = []
    
    ##### MAKE ARRAYS TO HOLD VALUES OF INTEREST #####
    x = x.flatten().astype(np.float64)
    hess = np.zeros(x.size)
    err = np.zeros(x.size)
    final_delta = np.zeros(x.size)
    
    ##### COMPUTE EACH PARTIAL DERIVATIVE #####
    for i in range(x.size):
        func = lambda xi: fun(replace_element(x, i, xi), *par)
        (hess[i], err[i], final_delta[i]) = derivest(func, x[i], **kwargs)
    return (hess, err, final_delta)
