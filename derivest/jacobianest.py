# -*- coding: utf-8 -*-

from . import utils

import numpy as np

# Default values for keyword arguments:
__kwargs = {"max_step": 1.0,
            "step_ratio": 2.0000001}

# Valid values for keyword arguments:
__acceptable_values = {"fixed_step": lambda x: x is None or x > 0,
                       "max_step": lambda x: x > 0,
                       "step_ratio": lambda x: x > 1}

def jacobianest(fun, x, par = None, **kwargs):
    """
    Estimate the Jacobian matrix of a vector-valued function of n variables.
    
    
    Arguments:
        fun : Callable object with signature fun(x, *par) -> np.ndarray, with x
              the argument and args an optional list of parameters.
        
        x : Location at which to compute the Jacobian. If x has more than one
            axis, then it is flattened and fun is assumed to be a function of
            np.prod(x.shape) variables.
        
        par : Optional list of parameters to be passed to fun as fun(x, *par).
              If par is not provided, then fun(x, *[]) is equivalent to fun(x),
              in which case fun may have signature fun(x) -> np.ndarray.
        
        
    Keyword Arguments:
        max_step : Maximum distance from the point x at which fun is to be
                   evaluated. Must be a positive value.          (Defualt: 1.0)
            
        step_ratio : Ratio between successive step sizes used in the propor-
                     tionally cascaded series of function evaluations. Must be
                     a value that exceeds one.             (Default: 2.0000001)
    
    
    Returns a 2-tuple containing:
        jac : Estimate of the Jacobian of fun at location x.
        
        err : Error estimates.
    
    
    Example:
      >>> import derivest, numpy as np
      >>>  def F(v, *args):
      ...      return np.array([v[0] + v[1]*v[2] + v[2]**3.0,
                                np.cos(v[0]**2.0 + v[1])])
      >>>  (jac, err) = derivest.jacobianest(F, np.array([1, 1, 1]))
      >>>  jac
      Out: array([[ 1.        ,  1.        ,  4.        ],
                  [-1.81859485, -0.90929743,  0.        ]])
      >>>  -np.sin(2.0)
      Out: -0.9092974268256817
    """
    ##### DETERMINE PARAMETER VALUES #####
    p = __kwargs.copy() # Start with default values.
    for key in kwargs:
        if key not in p:
            raise TypeError("jacobianest() got an unrecognized "
                            "keyword argument: %s" % key)
        p[key] = kwargs[key] # Replace with specified values.
        
        # Check for parameter value validity, either by membership in a given
        # list or by evaluation of a function as True:
        if isinstance(__acceptable_values[key], list):
            if p[key] not in __acceptable_values[key]:
                raise ValueError("Argument '%s' received an invalid value: %s"
                                 % (key, p[key]))
        elif not __acceptable_values[key](p[key]):
            raise ValueError("Argument '%s' received an invalid value: %s"
                             % (key, p[key]))
    if par is None:
        par = []
    
    # Evaluate at center point, check for easy case:
    try: x = x.flatten()
    except AttributeError: x = np.array(x).flatten()
    fx = fun(x, *par).flatten()
    if fx.size == 0: # Empty begets empty:
        return (np.zeros((0, x.size)), np.zeros((0, x.size)))
    (m, n) = (fx.size, x.size)
    
    ##### SET STEP SIZES TO USE AND COMPUTE APPROXIMATIONS #####
    delta = p["max_step"]*p["step_ratio"]**-np.arange(26)
    num_delta = len(delta)
    fdel = np.zeros((m, num_delta))
    jac = np.zeros((m, n))
    err = np.zeros((m, n))
    for i in range(n):
        # Evaluate at each step, centered around xi.
        # Use difference for 2nd-order estimate:
        for j in range(num_delta):
            fdel[:,j] = (fun(x + utils.evec(n, i, delta[j])) -
                             fun(x - utils.evec(n, i, delta[j]))).flatten()
        
        # The error term obtained here has a second order component, but also
        # some fourth and sixth order terms in it. Use Romberg exrapolation to
        # improve the estimates to sixth order, and provide the error estimate.
        der_est = 0.5*fdel/delta
        for j in range(m): # Loop; rombextrap w/ trimming too complicated.
            (der_romb, errors) = utils.romb_extrap(p["step_ratio"],
                                                   der_est[j,:], [2, 4])
            
            # Trim three estimates from each end of the scale:
            num_est = der_romb.size
            trim = np.array([0, 1, 2, num_est - 3, num_est - 2, num_est - 1])
            idx = np.delete(np.argsort(der_romb), trim)
            der_romb = der_romb[idx] # Trimmed, sorted array.
            errors = errors[idx]
            
            # Pick the estimate with the lowest predicted error:
            idx = np.argmin(errors) # Index of smallest (non-trimmed) error.
            err[j, i] = errors[idx]
            jac[j, i] = der_romb[idx]
    return (jac, err)
