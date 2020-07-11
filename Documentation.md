# Documentation for pyDERIVEST
## ``derivest``
``derivest.derivest(fun, x, par = None, **kwargs)``

Estimate the derivative of a function of one variable.

### Arguments
**fun**: *callable* ``fun(x, *args) -> float``
> The function to be numerically differentiated; ``x`` is the (scalar) argument, and ``args`` is an optional list of parameter values. For increased speed, a vectorized function ``fun(x, *args) -> np.ndarry`` may be provided, in which case the returned array must have the same shape as the array of argument values.

**x** : *float* or *ndarray*
> Point(s) at which the derivative is requested. If an array is provided, it will not be flattened.
        
**par** : *iterable*, optional
> List of parameter values to be passed to ``fun``, as ``fun(x, *par)``. If ``par`` is not provided, then ``fun(x, *[])`` is used, which is equivalent to calling the function with only the argument: ``fun(x)``.

### Keyword Arguments:
**deriv_order**: *int*, Default: ``1``
> Order of the derivative to be estimated. Derivatives up to order four ("fourth derivative") are available.
        
**fixed_step** : *float* or ``None``, Default: ``None``
> If specified, the fixed step size to use for computation; otherwise, uses an adaptive approach. Computation times will be significantly shorter with a fixed step size, but results may be less accurate.
        
**max_step**: *float*, Default: ``1.0``
> Maximum distance from the point ``x`` at which fun is to be evaluated. Must be a positive value.
        
**method_order** : *int*, Default: ``4``
> Order of the finite difference method used for estimation. The maximum available order is four, but only even values are permitted if ``style == "central"`` (see below). Higher-order methods are generally more accurate but tend to be more susceptible to numerical problems. First-order methods are usually not recommended.
        
**romberg_terms** : *int*, Default: ``2``
> Number of terms to use in Romberg extrapolation, up to a maximum of three. Specifying a value of zero disables Romberg extrapolation.
        
**step_ratio** : *float*, Default: ``2.0000001``
> Ratio (must exceed unity) between successive step sizes used in the cascaded series of function evaluations.
        
**style** : ``{"backward", "central", "forward"}``, Default: ``"central"``
> Type of finite difference method used.
        
**vectorized** : *bool*, Default: ``True``
> Specifier whether fun can be evaluated at multiple points from a single call. Doing so minimizes the overhead of a loop and additional function calls.

### Returns
**der** : *float* or *ndarray*
> An estimate of the specified derivative of ``fun`` at location(s) ``x``.
        
**err** : *float* or *ndarray*
> 95% uncertainty estimate of the error in the computed derivative.
        
**final_delta** : *float*
> The final overall stepsize chosen by derivest.

### Example
    >>>  import numpy as np
    >>>  from derivest import derivest
    >>>  (der, err, delta) = derivest(np.exp, 1.0)
    >>>  print(der, "|", np.exp(1.0) - der)
    Out: 2.7182818284590438 | 1.3322676295501878e-15
        
## ``directional_diff``
``derivest.directional_diff(fun, x, d, par = None, normalize = True, **kwargs)``

Estimate the directional derivative of a function of n variables.
    
Uses the ``derivest`` method to provide both a directional derivative and an error estimate.

### Arguments
**fun**: *callable* ``fun(x, *args) -> float``
> The function to be numerically differentiated; ``x`` is the (vector) argument, and ``args`` is an optional list of parameter values.

**x** : *ndarray*
> Vector location at which to differentiate fun. If ``x`` has more than one axis, then ``fun`` is assumed to be a function of ``x.size`` variables, as if it were flattened, but its shape is maintained.
        
**d** : *ndarray*, ``d.shape == x.shape``
> Vector defining the line along which to take the derivative. By default, the vector will automatically be normalized.
        
**par** : *iterable*, optional
> List of parameter values to be passed to fun as ``fun(x, *par)``. If par is not provided, then ``fun(x, *[])`` is used, which is equivalent to calling the function with only the argument: ``fun(x)``.
        
**normalize** : *bool*, optional
> Boolean specifying whether to normalize the direction vector to unit length.
        
Additional keyword arguments are passed to the internal call to ``derivest``, subject to compatibility with the task of computing a directional derivative.

### Returns
**dd** : *float*
> A scalar estimate of the first derivative of ``fun`` at location ``x``, in the specified direction ``d``.
        
**err** : *float*
> Error estimate of the directional derivative.
        
**final_delta** : *float*
> Vector of final step sizes for each partial derivative.

### Example
    >>>  import derivest, numpy as np
    >>>  def f(v, *args):
    ...      '''f(x, y; z) = x**2.0 + y**z'''
    ...      return v[0]**2.0 + v[1]**args[0]
    >>>  v = np.array([-1, 0]) # Evaluate at (x, y) = (-1, 0)
    >>>  d = np.array([ 1, 1]) # in the direction [1, 1].
    >>>  p = [1.0] # Set parameter value z = 1.
    >>>  (der, err, delta) = derivest.directional_diff(f, v, d, p)
    >>>  print(der, "|", err)
    Out: -0.7071067811865482 | 5.410189000248267e-15

## ``gradest``
``derivest.gradest(fun, x, par = None, **kwargs)``

Estimate the gradient vector of an analytical function of ``n`` variables.
    
Uses the ``derivest`` method to provide both an estimate of each component of the gradient vector and error estimates for each.

### Arguments
**fun**: *callable* ``fun(x, *args) -> float``
> The function to be numerically differentiated; ``x`` is the (vector) argument, and ``args`` is an optional list of parameter values.

**x** : *ndarray*
> Vector location at which to compute the gradient. If ``x`` has more than one axis, then ``fun`` is assumed to be a function of ``x.size`` variables, as if it were flattened, but its shape is maintained.
        
**par** : *iterable*, optional
> List of parameter values to be passed to ``fun`` as ``fun(x, *par)``. If par is not provided, then ``fun(x, *[])`` is used, which is equivalent to calling the function with only the argument: ``fun(x)``.
        
Additional keyword arguments are passed to the internal call to ``derivest``, subject to compatibility with the task of computing a gradient.

### Returns
**der** : *float*
> Estimate of the partial derivatives of ``fun`` at location ``x``.
        
**err** : *float*
> Error estimates of the partial derivatives.
        
**final_delta** : *float*
> The final overall stepsize chosen for each derivative.

### Example
    >>>  import derivest, numpy as np
    >>>  def rosenbrock(x, *args):
    ...      if not args: args = [100.0]
    ...      return (x[0] - 1.0)**2.0 + args[0]*(x[1] - x[0]**2.0)**2.0
    >>>  (der, err, delta) = derivest.gradest(rosenbrock, [1.0, 1.0])
    >>>  np.c_[der, err, delta].T
    Out: [[5.0073e-16 1.2336e-16]
          [4.7019e-15 1.1640e-15]
          [1.9073e-06 1.9073e-06]]
    >>>  derivest.gradest(rosenbrock, [0.5, 1.0])[0]
    Out: array([-151.,  150.])

## ``hess_diag``
``derivest.hess_diag(fun, x, par = None, **kwargs)``

Compute diagonal elements of the Hessian matrix of a scalar function.
    
Uses the ``derivest`` method to provide estimates of both the second derivatives and the corresponding error. This method is more efficient than ``hessian`` if only the diagonal elements of the Hessian matrix are needed.

### Arguments
**fun**: *callable* ``fun(x, *args) -> float``
> The function to be numerically differentiated; ``x`` is the (vector) argument, and ``args`` is an optional list of parameter values.

**x** : *ndarray*
> Vector location at which to compute the Hessian's diagonal. If ``x`` has more than one axis, then it is flattened and fun is assumed to be a function of ``x.size`` variables
        
**par** : *iterable*, optional
> List of parameter values to be passed to ``fun`` as ``fun(x, *par)``. If ``par`` is not provided, then ``fun(x, *[])`` is used, which is equivalent to calling the function with only the argument: ``fun(x)``.
        
Additional keyword arguments are passed to the internal call to ``derivest``, subject to compatibility with the task of computing diagonal elements of a Hessian matrix.

### Returns
**hess** : *float*
> Estimate of the partial derivatives of ``fun`` at location ``x``.
        
**err** : *float*
> Error estimates of the partial derivatives.
        
**final_delta** : *float*
> The final overall stepsize chosen for each derivative.

### Example
    >>>  from derivest import hess_diag
    >>>  f = lambda x: x[0] + x[1]**2.0 + x[2]**3.0
    >>>  (HD, err, fd) = hess_diag(f, [1, 2, 3])
    >>>  HD
    Out: array([ 0.,  2., 18.])
    >>>  err
    Out: array([0.0000e+00, 1.0213e-11, 5.6683e-12])

## ``hessian``
``derivest.hessian(fun, x, par = None, **kwargs)``

Estimate the Hessian matrix (array of second partial derivatives) of a scalar function of ``n`` variables.
    
This function is not a tool for frequent use on an expensive-to-evaluate functions, especially in a large number of dimensions. Its computation will use roughly ``O(6*n**2)`` function evaluations.

### Arguments
**fun**: *callable* ``fun(x, *args) -> float``
> The function to be numerically differentiated; ``x`` is the (vector) argument, and ``args`` is an optional list of parameter values.

**x** : *ndarray*
> Vector location at which to compute the Hessian's diagonal. If ``x`` has more than one axis, then it is flattened and fun is assumed to be a function of ``x.size`` variables
        
**par** : *iterable*, optional
> List of parameter values to be passed to ``fun`` as ``fun(x, *par)``. If ``par`` is not provided, then ``fun(x, *[])`` is used, which is equivalent to calling the function with only the argument: ``fun(x)``.
        
Additional keyword arguments are passed to the internal call to ``derivest``, subject to compatibility with the task of computing a Hessian matrix. If ``romberg_terms`` is not provided as a keyword argument, ``hessian`` uses a default value of 3, rather than the ordinary default value of 2.

### Returns
**hess** : *float*
> Estimate of the partial derivatives of ``fun`` at location ``x``.
        
**err** : *float*
> Error estimates of the partial derivatives.

### Example
    >>>  import derivest, numpy as np
    >>>  def f(xyz):
    ...      (x, y, z) = tuple(xyz)
    ...      return x*y + y**2.0*z + x*z**3.0
    >>>  X = np.array([-1, -1, -1])
    >>>  (hess, err) = derivest.hessian(f, X)
    >>>  hess
    Out: array([[ 0.,  1.,  3.],
                [ 1., -2., -2.],
                [ 3., -2.,  6.]])
    >>>  np.max(err)
    Out: 1.0927457963115134e-12

## ``jacobian``
``derivest.jacobian(fun, x, par = None, **kwargs)``

Estimate the Jacobian matrix of a vector-valued function of ``n`` variables.

### Arguments
**fun**: *callable* ``fun(x, *args) -> np.ndarray``
> The function to be numerically differentiated; ``x`` is the (vector) argument, and ``args`` is an optional list of parameter values.

**x** : *ndarray*
> Location at which to compute the Jacobian. If ``x`` has more than one axis, then it is flattened and ``fun`` is assumed to be a function of ``x.size`` variables.
        
**par** : *iterable*, optional
> List of parameter values to be passed to ``fun`` as ``fun(x, *par)``. If ``par`` is not provided, then ``fun(x, *[])`` is used, which is equivalent to calling the function with only the argument: ``fun(x)``.
        
### Keyword Arguments:
**max_step**: *float*, Default: ``1.0``
> Maximum distance from the point ``x`` at which fun is to be evaluated. Must be a positive value.
        
**step_ratio** : *float*, Default: ``2.0000001``
> Ratio (must exceed unity) between successive step sizes used in the cascaded series of function evaluations.

### Returns
**jac** : *float*
> Estimate of the Jacobian of ``fun`` at location ``x``.
        
**err** : *float*
> Error estimates of the partial derivatives.

### Example
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