# -*- coding: utf-8 -*-

from .gradest import gradest
from .hess_diag import hess_diag
from .utils import replace_elements, romb_extrap
import numpy as np

def hessian(fun, x, par = None, **kwargs):
    """
    Estimate the Hessian matrix (array of second partial derivatives) of a
    scalar function of n variables.
    
    This function is not a tool for frequent use on an expensive-to-evaluate
    functions, especially in a large number of dimensions. Its computation will
    use roughly O(6*n**2) function evaluations.
    
    
    Arguments:
        fun : Callable object with signature fun(x, *args) -> float, where x is
              the (vector) argument, and args is an optional list of parameters.
        
        x : Vector location at which to compute the gradient. If x has more than
            one axis, then it is flattened and fun is assumed to be a function
            of np.prod(x.shape) variables.
        
        par : Optional list of parameters to be passed to fun as fun(x, *par).
              If par is not provided, then fun(x, *[]) is equivalent to fun(x),
              in which case fun may have signature fun(x) -> float.
        
        Additional keyword arguments are passed to hess_diag. If romberg_terms
        is not provided as a keyword argument, hessian() uses a default value of
        3, rather than the ordinary default value of 2.
    
    
    Returns a 2-tuple containing square arrays of:
        hess : Estimate of the Hessian of fun at location x.
        
        err : Error estimates.
    
    
    Example:
      >>>  import derivest, numpy as np
      >>>  def f(xyz):
      ...      (x, y, z) = tuple(xyz)
      ...      return x*y + y**2.0*z + x*z**3.0
      >>>  X = np.array([-1, -1, -1])
      >>>  (hess, err) = derivest.hessian(f, X)
      Out: array([[ 0.,  1.,  3.],
                  [ 1., -2., -2.],
                  [ 3., -2.,  6.]])
      >>>  np.max(err)
      Out: 1.0921319030883257e-12
    """
    ##### PROCESS ARGUMENTS AND CHECK FOR VALIDITY #####
    if kwargs.pop("deriv_order", 2) != 2: raise ValueError("hessian() can only perform second-order differentiation.")
    if kwargs.pop("vectorized", False): raise ValueError("hessian() is incompatible with vectorized evaluation.")
    kwargs["deriv_order"] = 2 # Force second-order differentiation
    kwargs["vectorized"] = False # and non-vectorized evaluation.
    kwargs["romberg_terms"] = kwargs.pop("romberg_terms", 3) # Set default value if not provided.
    kwargs["step_ratio"] = float(kwargs.pop("step_ratio", 2.0000001))
    terms = kwargs["romberg_terms"] + 1
    if isinstance(x, list): x = np.array(x)
    if par is None: par = []
    
    ##### GET THE DIAGONAL OF THE HESSIAN #####
    (hess, err, _) = hess_diag(fun, x, par, **kwargs)
    
    # If 1-by-1 system, return floats; otherwise, form eventual Hessian matrix:
    if x.size == 1: return (float(hess), float(err))
    hess = np.diag(hess)
    err = np.diag(err)
    
    ##### FORM THE REMAINDER OF THE HESSIAN #####
    # Compute the gradient vector. This is done only to choose intelligent
    # step sizes for the mixed partials derivatives:
    step_size = gradest(fun, x, par)[2]
    
    # Get estimates of the lower triangle of the hessian matrix:
    dfac = kwargs["step_ratio"]**-np.arange(terms)
    zero = np.zeros(x.shape) # Generate a zero-matrix for use with swap_elements.
    for i in range(1, x.size):
        for j in range(i):
            dij = np.zeros(terms)
            for k in range(terms):
                dij[k] = fun(x + replace_elements(zero, (i,j), ( dfac[k]*step_size[i],  dfac[k]*step_size[j]))) - \
                         fun(x + replace_elements(zero, (i,j), ( dfac[k]*step_size[i], -dfac[k]*step_size[j]))) - \
                         fun(x + replace_elements(zero, (i,j), (-dfac[k]*step_size[i],  dfac[k]*step_size[j]))) + \
                         fun(x + replace_elements(zero, (i,j), (-dfac[k]*step_size[i], -dfac[k]*step_size[j])))
            dij /= 4.0*step_size[i]*step_size[j]*dfac**2.0
            
            # Romberg extrapolation step:
            (hess[i,j], err[i,j]) = romb_extrap(kwargs["step_ratio"], dij, [2, 4]);
            (hess[j,i], err[j,i]) = (hess[i,j], err[i,j]) # Use symmetry to avoid further computation.
    return (hess, err)
