# -*- coding: utf-8 -*-

from .derivest import derivest
from .utils import replace_element
import numpy as np

def gradest(fun, x, par = None, **kwargs):
    """
    Estimate the gradient vector of an analytical function of n variables.
    
    Uses the derivest method to provide both an estimate of each component
    of the gradient vector and error estimates for each.
    
    
    Arguments:
        fun : Callable object with signature fun(x, *args) -> float, where x is
              the (vector) argument, and args is an optional list of parameters.
        
        x : Vector location at which to compute the gradient. If x has more than
            one axis, then fun is assumed to be a function of np.prod(x.shape)
            variables, but it is not flattened.
        
        par : Optional list of parameters to be passed to fun as fun(x, *par).
              If par is not provided, then fun(x, *[]) is equivalent to fun(x),
              in which case fun may have signature fun(x) -> float.
        
        Additional keyword arguments are passed to derivest.
    
    
    Returns a 3-tuple containing arrays, matching x in shape, of:
        der : Estimate of the partial derivatives of fun at location x.
        
        err : Error estimates of the partial derivatives.
        
        final_delta : The final overall stepsize chosen for each derivative.
    
    
    Example:
      >>>  import derivest, numpy as np
      >>>  def rosenbrock(x, *args):
      ...      if not args: args = [100.0]
      ...      return (x[0] - 1.0)**2.0 + args[0]*(x[1] - x[0]**2.0)**2.0
      >>>  (der, err, delta) = derivest.gradest(rosenbrock, [1.0, 1.0])
      >>>  print(np.c_[der, err, delta].T)
      Out: [[1.96469224e-24 0.00000000e+00]
            [7.24175738e-23 0.00000000e+00]
            [7.62939453e-06 1.52587891e-05]]
      >>>  derivest.gradest(rosenbrock, [0.5, 1.0])[0]
      Out: array([-151.,  150.])
    """
    ##### PROCESS ARGUMENTS AND CHECK FOR VALIDITY #####
    if kwargs.pop("deriv_order", 1) != 1: raise ValueError("gradest() can only perform first-order differentiation.")
    if kwargs.pop("vectorized", False): raise ValueError("gradest() is incompatible with vectorized evaluation.")
    if kwargs.pop("method_order", 2) != 2: raise ValueError("gradest() can only use second-order derivative estimation.")
    kwargs["deriv_order"] = 1 # Force first-order differentiation,
    kwargs["vectorized"] = False # non-vectorized evaluation, and
    kwargs["method_order"] = 2 # second-order computation.
    if isinstance(x, list): x = np.array(x)
    if par is None: par = []
    
    ##### MAKE ARRAYS TO HOLD VALUES OF INTEREST #####
    sx = x.shape # Record the size of x for reshaping later.
    nx = x.size  # The number of derivatives to be taken.
    x = x.flatten()
    grad = np.zeros(nx)
    err = np.zeros(nx)
    final_delta = np.zeros(nx)
    
    ##### COMPUTE EACH PARTIAL DERIVATIVE #####
    for i in range(nx):
        func = lambda xi: fun(replace_element(x, i, xi), *par)
        (grad[i], err[i], final_delta[i]) = derivest(func, x[i], **kwargs)
    return (np.reshape(grad, sx), np.reshape(err, sx), np.reshape(final_delta, sx))
