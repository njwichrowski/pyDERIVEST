# Documentation for pyDERIVEST
## ``derivest``
``derivest(fun, x, par = None, **kwargs)``

Estimate the derivative of a function of one variable.

### Arguments
**fun**: *callable* ``fun(x, *args) -> float``
> The function to be numerically differentiated; ``x`` is the (scalar) argument, and ``args`` is an optional list of parameter values. For increased speed, a vectorized function ``fun(x, \*args) -> np.ndarry`` may be provided, in which case the returned array must have the same shape as the array of argument values.

**x** : *float* or *ndarray*
> Point(s) at which the derivative is requested. If an array is provided, it will not be flattened.
        
**par** : *iterable*, optional
> List of parameter values to be passed to ``fun`` as ``fun(x, \*par)``. If ``par`` is not provided, then ``fun(x, \*\[\])`` is used, which is equivalent to ``fun(x)``.

### Keyword Arguments:
**deriv_order**: *int*, Default: 1
> Order of the derivative to be estimated. Derivatives up to order four ("fourth derivative") are available.
        
**fixed_step** : *float* or *None*, Default: ``None``
> If specified, the fixed step size to use for computation; otherwise, uses an adaptive approach. Computation times will be significantly shorter with a fixed step size, but results may be less accurate.
        
**max_step**: *float*, Default: 1.0
> Maximum distance from the point ``x`` at which ``fun`` is to be evaluated. Must be a positive value. (Defualt: 1.0)
        
**method_order** : *int*, Default: 4
> Order of the finite difference method used for estimation. The maximum available order is four, but only even values are permitted if ``style == "central"`` (see below). Higher-order methods are generally more accurate but tend to be more susceptible to numerical problems. First-order methods are usually not recommended. (Default: 4)
        
**romberg_terms** : *int*, Default: 2
> Number of terms to use in Romberg extrapolation, up to a maximum of three. Specifying a value of zero disables Romberg extrapolation.
        
**step_ratio** : *float*, Default: 2.0000001
> Ratio (must exceed unity) between successive step sizes used in the proportionally-cascaded series of function evaluations.
        
**style** : *str*, Default: "central"
> Type of finite difference method used, from among: "backward", "central", "forward".
        
**vectorized** : *bool*, Default: True
> Specifier whether ``fun`` can be evaluated at multiple points from a single call. Doing so minimizes the overhead of a loop and additional function calls.

### Returns
**der** : *float* or *ndarray*
> An estimate of the specified derivative of ``fun`` at location(s) ``x``.
        
**err** : *float* or *ndarray*
> 95% uncertainty estimate of the error in the computed derivative.
        
**final_delta** : *float* The final overall stepsize chosen by derivest.

### Example
    >>>  import derivest, numpy as np
    >>>  (der, err, delta) = derivest.derivest(np.exp, 1.0)
    >>>  print(der, "|", np.exp(1.0) - der)
    Out: 2.7182818284590344 | 1.0658141036401503e-14
        
## ``directional_diff``
``directional_diff(fun, x, d, par = None, normalize = True, **kwargs)``

Estimate the directional derivative of a function of n variables.
    
Uses the derivest method to provide both a directional derivative and an error estimate.

### Arguments
**fun**: *callable* ``fun(x, *args) -> float``
> The function to be numerically differentiated; ``x`` is the (vector) argument, and ``args`` is an optional list of parameter values.

**x** : *ndarray*
> Vector location at which to differentiate fun. If x has more than one axis, then fun is assumed to be a function of ``np.prod(x.shape)`` variables, as if it were flattened, but its shape is maintained.
        
**d** : *ndarray*, ``d.shape == x.shape``
> Vector defining the line along which to take the derivative. By default, the vector will automatically be normalized.
        
**par** : *iterable*, optional
> List of parameter values to be passed to ``fun`` as ``fun(x, \*par)``. If ``par`` is not provided, then ``fun(x, \*\[\])`` is used, which is equivalent to ``fun(x)``.
        
**normalize** : *bool*, optional
> Boolean specifying whether to normalize the direction vector to unit length.
        
Additional keyword arguments are passed to the internal call to ``derivest``.

### Returns
**dd** : *float*
> A scalar estimate of the first derivative of fun at location ``x``, in the specified direction ``d``.
        
**err** : *float*
> Error estimate of the directional derivative.
        
**final_delta** : *float*
> Vector of final step sizes for each partial derivative.
