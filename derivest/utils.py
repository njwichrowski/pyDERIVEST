# -*- coding: utf-8 -*-
"""Auxiliary methods for the derivest module.

"""

__all__ = ["build_kwargs", "diag_tile", "evec", "fdamat", "loop_eval", "outlier_bounds", "replace_element", "replace_elements", "romb_extrap"]

from scipy.special import factorial
from scipy.stats import iqr
import numpy as np
import warnings

def build_kwargs(COL = {}, CON = {}, DIS = {}, INT = {}):
    """
    Construct a dictionary of keyword arguments and values, for use in ensemble.
    
    
    Arguments: All four parameters must be dictionaries, with keys corresponding
    to the desired keyword arguments for the returned dictionary. Values for
    each key are determined on the basis of which parameter contains them:
        
        COL : Collections of items; choose uniformly at random from a finite
              number of values, contained in an iterable.
        
        CON : Constant values; use a provided, fixed value.
        
        DIS : Distributions objects; use an object's rvs() method, in the manner
              of scipy.stats rv_ distributions, to generate a value.
        
        INT : Intervals; choose a value uniformly on an interval defined by
              lower and upper bounds.
    
    
    Returns a dictionary d which can be unpacked within a call to one of the
    main methods to specify keyword argument values.
    
    
    Example:
      >>>  import derivest
      >>>  from scipy.stats import lognorm
      >>>  COL = {"method_order": [2, 4]}
      >>>  CON = {"deriv_order": 2}
      >>>  DIS = {"max_step": lognorm(2.0)}
      >>>  INT = {"step_ratio": (1.1, 3.0)}
      >>>  kw = derivest.build_kwargs(COL, CON, DIS, INT)
      >>>  print(kw)
      Out: {'method_order': 4, 'deriv_order': 2, 'max_step': 0.1480, 'step_ratio': 1.5413}
      >>>  print(derivest.build_kwargs(COL, CON, DIS, INT))
      Out: {'method_order': 2, 'deriv_order': 2, 'max_step': 1.9204, 'step_ratio': 1.6793}
      >>>  derivest.derivest(lambda x: x**2.0, 1.0, **kw)
      Out: 1.999999999999703
    """
    d = {}
    for key in COL: # Collections of items.
        x = COL[key]
        d[key] = x[np.random.randint(len(x))]
    for key in CON: # Constant values.
        d[key] = CON[key]
    for key in DIS: # Distributions.
        d[key] = DIS[key].rvs()
    for key in INT: # Numeric intervals.
        (lo, hi) = tuple(INT[key])
        d[key] = np.random.uniform(low = lo, high = hi)
    return d

def diag_tile(vec, size, flatten = False, modulo = False):
    """
    Create an array by diagonally tiling a vector.
    
    
    Arguments:
        vec : The vector to be tiled. Must be one-dimensional unless
              'flatten = True' is specified.
              
        size : A tuple specifying the shape of the new array.
        
        flatten : If True, a multi-dimensional array is flattened before
                  being tiled; otherwise, raises a ValueError. (Default: False)
        
        modulo : If True, repeats vec as necessary to accomodate an otherwise-
                 too-short vector; otherwise, raises a ValueError whenever
                 vec.size <= sum(size) - len(size).            (Default: False)
    
    
    Returns an array M satisfying:
        M.shape == size    
        M[i,j, ...] == vec[i+j+...] for all valid indices
    """
    # Check input shape, and flatten if allowed:
    if len(vec.shape) > 1:
        if flatten: vec = vec.flatten()
        else: raise ValueError("Specify 'flatten = True' to flatten multi-dimensional arrays.")
    
    # Build the tiled array:
    arrs = np.meshgrid(*[np.arange(x) for x in size], indexing = "ij") # List of arrays corresponding to each index.
    idx = np.sum(np.array(arrs), axis = 0) # Compute sum of arrays contained in arrs.
    if modulo: return vec[idx % vec.size] # Build array by viewing into original vector.
    elif vec.size <= sum(size) - len(size): raise ValueError("Provided vector is too short (%d) for shape %s." % (vec.size, size))
    else: return vec[idx]

def evec(dim, entry, value = 1.0, shape = None):
    """
    Form a standard basis vector, consisting of all zeros except at one entry.
    
    
    Arguments:
        dim : Number of entries the vector should have.
        
        entry : Index of the non-zero entry.
        
        value : Value of the non-zero entry. (Default: 1.0)
        
        shape : Tuple specifying the desired shape for the vector. If specified,
                entry refers to the flattened index.     (Default: None --> 1D)
    """
    e = np.zeros(dim)
    e[entry] = value
    if shape is not None: e = np.reshape(e, shape)
    return e

def fdamat(sr, parity, num_terms):
    """
    Compute matrix for Finite Difference Approximation derivation.
    
    
    Arguments:
        sr : Ratio between successive steps.
        
        parity : An integer in {0, 1, 2}:
                    0 --> one-sided, all terms included except zeroth order
                    1 --> only odd terms included
                    2 --> only even terms included
        
        num_terms : Number of terms included.
    
    
    Returns a two dimensional array that defines a linear system, the solution
    of which specifies the finite difference approximation coefficients used
    in the derivest method.
    """
    sr_inv = 1.0/sr
    vec = np.arange(num_terms)
    (j, i) = np.meshgrid(vec, vec)
    if parity == 0:
        c = 1.0/factorial(vec + 1)
        return c[j]*sr_inv**(i*(j + 1))
    elif parity == 1:
        c = 1.0/factorial(2*vec + 1)
        return c[j]*sr_inv**(i*(2*j + 1))
    elif parity == 2:
        c = 1.0/factorial(2*(vec + 1))
        return c[j]*sr_inv**(i*(2*j + 2))
    else: raise ValueError("fdamat() got an invalid parity: %s" % parity)

def loop_eval(fun, x, par = None):
    """
    Evaluate a scalar function of one variable at multiple points via for-loop.
    
    
    Arguments:
        fun : Callable object with signature fun(x, *args) -> float, where x is
              the (vector) argument, and args is an optional list of parameters.
        
        x : Array of scalar values at which to evaluate fun.
        
        par : Optional list of parameters to be passed to fun as fun(x, *par).
              If par is not provided, then fun(x, *[]) is equivalent to fun(x),
              in which case fun may have signature fun(x) -> float.
    """
    x_flat = x.flatten()
    fx = np.zeros(x.size)
    if par is None: par = [] # Avoid TypeError from attempting *None when par not provided.
    for i in range(x.size): fx[i] = fun(x_flat[i], *par)
    return np.reshape(fx, x.shape)

def outlier_bounds(sample, scale = 1.5, **kwargs):
    """
    For a set or sets of one-dimensional data, compute, via the interquartile
    range approach, values beyond which data points may be considered outliers.
    
    
    Arguments:
        sample : An array of values for which the outlier bounds are desired.
                 May be treated as multiple sets via "axis" keyword argument.
        
        scale : Multiple of the interquartile range used to define the bounds:
                    lower = Q1 - scale*IQR
                    upper = Q3 + scale*IQR
        
        Additional keyword arguments (except "keepdims") are passed to
        scipy.stats.iqr and, if recognized thereby, to numpy.percentile as well.
        If "rng" is provided for computing the IQR, it is also used for the base
        points, in place of the proper quartiles (Q1 and Q3).
    
    
    Returns a numpy.ndarray with the computed bounds. If no axis was specified,
    then the results correspond to the whole sample, the returned array has
    ob.shape == (2,), and the interval of non-outliers is ob[0] <= x <= ob[1].
    If an axis was specified, then the first axis of the returned array corre-
    sponds to lower and upper bounds, and the remaining axes are left in order.
    """
    if "axis" in kwargs and kwargs["axis"] is not None:
        try:
            ax = int(kwargs["axis"]) % len(sample.shape)
            kwargs["axis"] = ax
        except TypeError: raise ValueError("Could not coerce '%s' into an int." % kwargs["axis"])
        except ValueError: raise ValueError("Could not convert '%s' to an int." % kwargs["axis"])
        keepdims = True
    else: keepdims = False
    kwargs["keepdims"] = keepdims
    iqr_ = iqr(sample, **kwargs) # Compute interquartile range.
    qile_kw = {key:kwargs[key] for key in kwargs if key in ["axis", "interpolation", "keepdims"]}
    r = np.array([-scale, scale])
    
    # Compute the quartiles themselves (or surrogate):
    if "rng" in kwargs: Q = kwargs["rng"]
    else: Q = (25.0, 75.0)
    qile = np.percentile(sample, Q, **qile_kw).T
    if keepdims:
        iqr_ = iqr_.T # Reverse order of axes.
        ob = (qile + r*iqr_[..., np.newaxis]).T # Compute outlier bounds.
        return np.squeeze(ob, axis = ax + 1) # Remove length-one axis.
    else: return (qile + r*iqr_[np.newaxis, ...]).flatten()

def replace_element(vec, idx, val, dtype = np.float64):
    """
    Generate a copy of a vector, with a single entry value altered.
    
    
    Arguments:
        vec : Vector to be modified.
        
        idx : The index at which to alter vec.
        
        val : The new value for vec[idx].
    
        dtype : Data type to which the copied array should be converted, or None
                to leave in the original format.       (Default: numpy.float64)
    """
    if dtype is None: vec = np.array(vec) # Convert list to array or copy array.
    else: vec = np.array(vec, dtype = dtype) # Change dtype if requested.
    vec[idx] = val
    return vec

def replace_elements(vec, idxs, vals, dtype = np.float64):
    """
    Generate a copy of a vector, with specified entry values altered.
    
    
    Arguments:
        vec : Vector to be modified.
        
        idxs : Iterable of indices at which to alter vec.
        
        vals : Iterable of new values for corresponding entries.
    
        dtype : Data type to which the copied array should be converted, or None
                to leave in the original format.       (Default: numpy.float64)
    """
    if len(idxs) != len(vals): raise ValueError("Lists of indices, values must have the same length.")
    if dtype is None: vec = np.array(vec) # Convert list to array or copy array.
    else: vec = np.array(vec, dtype = dtype) # Change dtype if requested.
    for (i, v) in zip(idxs, vals): vec[i] = v
    return vec

def romb_extrap(sr, der_init, expon, compute_amp = False):
    """
    Perform Romberg extrapolation for estimates formed within derivest.
    
    
    Arguments:
        sr : Decrease ratio between successive steps.
        
        der_init : Initial derivative estimates.
        
        expon : List of orders corresponding to the higher-order terms to be
                cancelled via Romberg step. The accepted parameter values of
                derivest will use a list of, at most, three values. A warning
                is issued if a longer list is received.
        
        compute_amp : Boolean specifying whether to also compute the noise
                      amplification factor.                    (Default: False)
    
    
    Returns a 2-tuple or 3-tuple, containing:
        der_romb : Derivative estimates.
        
        err_est : Error estimates.
        
        amp : Computed noise amplification factor (only if compute_amp == True).
    """
    # Guarantee that expon is a one-dimensional array of floats:
    if isinstance(expon, list): expon = np.array(expon).flatten()
    elif not isinstance(expon, np.ndarray): expon = np.array([float(expon)])
    else: expon = expon.flatten()
    num_expon = expon.size
    
    # Construct the Romberg matrix:
    sr_inv = 1.0/sr
    rmat = np.ones((num_expon + 2, num_expon + 1))
    if num_expon > 3: warnings.warn("Ordinary use of derivest() should not need more than three terms to be cancelled.", RuntimeWarning)
    elif num_expon > 0:
        for i in range(1, num_expon + 2):
            rmat[i, np.arange(1, num_expon + 1)] = sr_inv**(i*expon)# Compute QR factorization for extrapolation and uncertainty estimates:
    (Q, R) = np.linalg.qr(rmat)
    
    # Extrapolate to a zero step-size:
    rhs = diag_tile(der_init, (num_expon + 2, max(1, der_init.size - num_expon - 2)), flatten = True)
    coeffs = np.linalg.lstsq(R, Q.T @ rhs, rcond = None)[0] # Compute Romberg coefficients by solving linear systems.
    der_romb = coeffs[0,:] # Extract derivative estimates.
    
    # Approximate the uncertainty:
    s = np.sqrt(np.sum((rhs - rmat @ coeffs)**2.0, axis = 0))
    R_inv = np.linalg.lstsq(R, np.eye(num_expon + 1), rcond = None)[0]
    cov = np.sum(R_inv**2.0, axis = 1)
    err_est = 12.7062047361747*np.sqrt(cov[0])*s
    
    if compute_amp: return (der_romb, err_est, np.linalg.cond(R))
    else: return (der_romb, err_est)
