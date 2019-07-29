# -*- coding: utf-8 -*-

from .derivest import derivest
import numpy as np

def directional_diff(fun, x, d, par = None, normalize = True, **kwargs):
    """
    Estimate the directional derivative of a function of n variables.
    
    Uses the derivest method to provide both a directional derivative
    and an error estimate.
    
    
    Arguments:
        fun : Callable object with signature fun(x, *args) -> float, where x is
              the (vector) argument, and args is an optional list of parameters.
        
        x : Vector location at which to differentiate fun. If x has more than
            one axis, then fun is assumed to be a function of np.prod(x.shape)
            variables, but it is not flattened.
        
        d : Vector (d.shape == x.shape) defining the line along which to take
            the derivative. The vector will automatically be normalized.
        
        par : Optional list of parameters to be passed to fun as fun(x, *par).
              If par is not provided, then fun(x, *[]) is equivalent to fun(x),
              in which case fun may have signature fun(x) -> float.
        
        normalize : Boolean specifying whether to normalize d.  (Default: True)
        
        Additional keyword arguments are passed to derivest.
    
    
    Returns a 3-tuple, containing:
        dd : A scalar estimate of the first derivative of fun at location x,
             in the specified direction d.
        
        err : Error estimate of the directional derivative.
        
        final_delta : Vector of final step sizes for each partial derivative.
    
    
    Example:
      >>>  import derivest, numpy as np
      >>>  def f(v, *args):
      ...      '''f(x, y; z) = x**2.0 + y**z'''
      ...      return v[0]**2.0 + v[1]**args[0]
      >>>  v = np.array([-1, 0]) # Evaluate at (x, y) = (-1, 0)
      >>>  d = np.array([ 1, 1]) # in the direction [1, 1].
      >>>  p = [1.0] # Set parameter value z = 1.
      >>>  derivest.directional_diff(f, v, d, p)
      Out: (-0.7071, 1.2478e-13, 0.0002)
    """
    ##### PROCESS ARGUMENTS AND CHECK FOR VALIDITY #####
    if kwargs.pop("deriv_order", 1) != 1: raise ValueError("directional_diff() can only perform first-order differentiation.")
    if kwargs.pop("vectorized", False): raise ValueError("directional_diff() is incompatible with vectorized evaluation.")
    kwargs["deriv_order"] = 1 # Force first-order differentiation
    kwargs["vectorized"] = False # and non-vectorized evaluation.
    if isinstance(x, list): x = np.array(x) # Avoid AttributeError from x.shape if given a list.
    if isinstance(d, list): d = np.array(d)
    d = d.astype(np.float64) # Avoid TypeError from trying to /= on an array of integers.
    if par is None: par = [] # Avoid TypeError from attempting *None when par not provided.
    if x.shape != d.shape: raise ValueError("Shapes must match. Got x -> %s and d -> %s" % (x.shape, d.shape))
    if np.allclose(d, np.zeros_like(d)): raise ValueError("Direction vector is numerically zero.")
    if normalize: d /= np.sqrt(np.sum(d**2.0)) # Normalize direction.
    
    ##### COMPUTE DIRECTIONAL DERIVATIVE #####
    func = lambda t: fun(x + t*d, *par)
    return derivest(func, 0.0, **kwargs)
