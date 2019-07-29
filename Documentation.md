# Documentation for pyDERIVEST
## ``derivest``

def derivest(fun, x, par = None, **kwargs):
    """
    Estimate the derivative of a function of one variable.
    
    
    Arguments:
        fun : Callable object with signature fun(x, *args) -> float, where x is
              the (scalar) argument, and args is an optional list of parameters.
              
              For increased speed, a vectorized function may be provided, with
              signature fun(x, *args) -> np.ndarry, where the returned array
              must have the same shape as the array of argument values.
        
        x : Scalar or numpy array containing point(s) at which the derivative
            is requested. If an array is provided, it will not be flattened.
        
        par : Optional list of parameters to be passed to fun as fun(x, *par).
              If par is not provided, then fun(x, *[]) is equivalent to fun(x),
              in which case fun may have signature fun(x) -> float.
        
        
    Keyword Arguments:
        deriv_order : Order of the derivative to be estimated. Must be an
                      integer in [1, 2, 3, 4].                     (Default: 1)
        
        fixed_step : Allows the specification of a fixed step size, rather than
                     using an adaptive approach. Computation times will be sig-
                     nificantly shorter, but results may be less accurate.
                     If specified, fixed_step must be a positive value and will
                     define the maximum distance from x at which fun is to be
                     evaluated.       (Default: None --> use adaptive approach)
        
        max_step : Maximum distance from the point x at which fun is to be
                   evaluated. Must be a positive value.          (Defualt: 1.0)
        
        method_order : Order of the finite difference method used for estim-
                       ation. Must be an integer in [1, 2, 3, 4]; further, if
                       style == "central" then only even values are permitted.
                       Higher-order methods are generally more accurate but tend
                       to be more susceptible to numerical problems. First-order
                       methods are usually not recommended.        (Default: 4)
        
        romberg_terms : Number of terms to use in Romberg extrapolation. Must be
                        an integer in [0, 1, 2, 3]. Specifying a value of zero
                        disables Romberg extrapolation.            (Default: 2)
        
        step_ratio : Ratio between successive step sizes used in the proportion-
                     ally cascaded series of function evaluations. Must be a
                     value that exceeds one.               (Default: 2.0000001)
        
        style : Type of finite difference method used. Must be a string in
                ["backward", "central", "forward"].        (Default: "central")
        
        vectorized : Boolean specifying whether fun can be evaluated at multiple
                     points from a single call. Doing so minimizes the overhead
                     of a loop and additional function calls.   (Default: True)
    
    
    Returns a 3-tuple containing:
        der : An estimate of the specified derivative of fun at location x.
        
        err : 95% uncertainty estimate of the error in the computed derivative.
        
        final_delta : The final overall stepsize chosen by derivest.
    
    
    Example:
      >>>  import derivest, numpy as np
      >>>  (der, err, delta) = derivest(np.exp, 1.0)
      >>>  print(der, "|", np.exp(1.0) - der)
      Out: 2.7182818284590344 | 1.0658141036401503e-14
        
