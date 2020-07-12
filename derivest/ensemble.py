# -*- coding: utf-8 -*-

from scipy.stats import lognorm
import numpy as np

from .derivest import derivest
from .directional_diff import directional_diff
from .gradest import gradest
from .hess_diag import hess_diag
from .hessian import hessian
from .jacobianest import jacobianest
from .utils import build_kwargs, outlier_bounds

# Dictionary for methods included in the original MATLAB package:
__base_methods = {"derivest": derivest,
                  "directional_diff": directional_diff,
                  "gradest": gradest,
                  "hess_diag": hess_diag,
                  "hessian": hessian,
                  "jacobianest": jacobianest}

def ensemble(method, fun, x, par = None, N = 10, weights = "uniform",
             eps = 1e-15, method_kwargs = None, ob_kwargs = None, **kwargs):
    """
    Compute multiple approximations of a derivative and aggregate the results.
    
    Checks the results for outliers, removing any that are found, and uses the
    remaining estimates to form a single approximation to the derivative.
    
    
    Arguments:
        method : String specifying the name of the method to use (e.g.,
                 "derivest", etc.) or the method itself (derivest.derivest).
        
        fun : Callable object that evaluates the function of interest.
        
        x : Array containing the point at which to differentiate.
        
        par : Optional list of parameters to be passed to fun as fun(x, *par).
              If par is not provided, then fun(x, *[]) is equivalent to fun(x),
              in which case fun may have signature fun(x) -> float.
        
        N : The number of evaluations to use to form the aggregated estimate.
            Must be an integer at least equal to 3.               (Default: 10)
        
        weights : String, from options described below, indicating how to com-
                  bine the N estimates into one result.    (Default: "uniform")
                  
                  "uniform"    : Unweighted average of all non-outliers.
                  "error"      : Average of non-outliers, weighted by the reci-
                                 procal of the error estimates on an entry-wise
                                 basis (relevant for non-scalar outputs).
                  "mean_error" : Average of non-outliers, weighted by the reci-
                                 procal of the mean error estimates for each of
                                 the N evaluations.
                  "max_error"  : Average of non-outliers, weighted by the reci-
                                 procal of the maximum error estimates for each
                                 of the N evaluations.
                  "best"       : Selects the estimate with the smallest
                                 approximated error.
                  "best_mean"  : Selects the estimate with the smallest mean
                                 (over the result entries) approximated error.
                  "best_max    : Selects the estimate with the smallest maximum
                                 (over the result entries) approximated error.
        
        eps : Positive float to serve as a precision buffer. When computing
              weighted averages of non-outlier results, any error estimates
              less than eps will be replaced with eps. A ValueError is raised
              if eps is smaller than machine precision (1.0 + eps <= 1.0). In
              most cases, machine precision is ~2.23e-16.      (Default: 1e-15)
        
        method_kwargs: Dictionary in which the keys are accepted keyword argu-
                       ments for the method to be used, and the values specify
                       how to determine the keyword values for each of the N
                       times the method is called. If the value for a key ...
                       
                       [a] has an rvs() method for generating random variates
                       [b] is a list
                       [c] is a tuple
                       [d] fails to satisfy any of [a], [b], or [c],
                       
                       ... then the value of the corresponding keyword argument
                       is determined for each method call by (resp.) ...
                       
                       [a] calling rvs() to obtain a single (scalar) value
                       [b] picking uniformly at random from the list's elements
                       [c] sampling uniformly at random from the interval
                           extending between the 0 and 1 elements of the tuple
                       [d] always using the provided value.
                       
                       Note that it is a computational waste to use only [d]
                       (constant-valued arguments) in method_kwargs, since do-
                       ing so will merely compute an approximate derivative N
                       times at the exact same conditions.
                       
                       By default, method_kwargs varies the 'max_step' para-
                       meter according to a log-normal distribution.
        
        ob_kwargs : Dictionary of key-value pairs to pass to outlier_bounds
                    when determining which estimates constitute outliers.
        
        If using directional_diff, then d and normalize are taken from the
        additional keyword arguments (**kwargs). If using derivest, deriv_order
        may be specified in **kwargs as well.
    
    
    Returns a 2-tuple containing:
        deriv : Estimate of the requested derivative of fun at location x.
        
        error : Corresponding error estimate(s).
    """
    args = {"method": method, "fun": fun, "par": par, "N": N,
            "weights": weights, "eps": eps, "method_kwargs": method_kwargs,
            "ob_kwargs": ob_kwargs, "kwargs": kwargs}
    weight_types = ["uniform", "error", "mean_error", "max_error",
                    "best", "best_mean", "best_max"]
    if kwargs.pop("details", False):
        details = {"arguments": args}
    else:
        details = False
    
    # Check for valid arguments and perform set-up:
    if method in __base_methods:
        method = __base_methods[method] # Convert to actual function.
    elif method not in __base_methods.values():
        if isinstance(method, str):
            raise ValueError("Unrecognized method name: '%s'" % method)
        else:
            raise ValueError("Unrecognized method: %s" % method)
    if method_kwargs is None:
        method_kwargs = {"max_step": lognorm(2.0)}
    if method == directional_diff:
        ddiff = True
        if "d" in kwargs:
            d = kwargs.pop("d")
        elif "d" in method_kwargs:
            d = method_kwargs.pop("d")
        else:
            raise ValueError("Need direction 'd' for directional_diff().")
            
        if "normalize" in kwargs:
            normalize = kwargs.pop("normalize")
        else:
            normalize = method_kwargs.pop("d", True)
    else:
        ddiff = False
        
    if N < 3:
        raise ValueError("Need at least three estimates to form an ensemble.")
    if weights not in weight_types:
        raise ValueError("Invalid weighting method: '%s'" % weights)
    if (1.0 + eps) <= 1.0:
        raise ValueError("Precision buffer value is not "
                         "sufficiently positive.")
    if "deriv_order" in kwargs:
        if method == derivest:
            method_kwargs["deriv_order"] = kwargs.pop("deriv_order")
        else:
            raise ValueError("Can only specify deriv_order for derivest().")
    if par is None: par = []
    
    # Determine which of the method_kwargs is each type of specification:
    (collections, constants, distributions, intervals) = ({}, {}, {}, {})
    for key in method_kwargs: # Sort method arguments by type.
        val = method_kwargs[key]
        try: # Do we have a scipy.stats frozen distribution?
            val.rvs()
            distributions[key] = val
        except AttributeError: # If not,
            if isinstance(val, list):
                collections[key] = val # Uniform choice over finite collection.
            elif isinstance(val, tuple):
                intervals[key] = val # Numeric interval, uniform distribution.
            else:
                constants[key] = val # Constant value.
    
    # Compute a first approximation and determine shape of results:
    kw = build_kwargs(collections, constants, distributions, intervals)
    if details:
        iter_kw = [kw]
    if ddiff:
        output = directional_diff(fun, x, d, par, normalize, **kw)
    else:
        output = method(fun, x, par, **kw)
        
    try:
        values_ = np.zeros(list(output[0].shape) + [N])
    except AttributeError:
        values_ = np.zeros(N) # Use 1D array if float returned.
    
    values_[..., 0] = output[0]
    try:
        errors_ = np.zeros(list(output[1].shape) + [N])
    except AttributeError:
        errors_ = np.zeros(N) # Use 1D array if float returned.
    errors_[..., 0] = output[1]
    all_but_last = tuple(range(len(values_.shape) - 1))
    
    # Compute the other N - 1 approximations for analysis:
    for i in range(1, N):
        kw = build_kwargs(collections, constants, distributions, intervals)
        if details:
            iter_kw.append(kw)
        if ddiff:
            output = directional_diff(fun, x, d, par, normalize, **kw)
        else:
            output = method(fun, x, par, **kw)
        values_[..., i] = output[0]
        errors_[..., i] = output[1]
    
    # Check whether any values are outliers:
    if ob_kwargs is None:
        ob_kwargs = {"axis": -1}
    else:
        ob_kwargs["axis"] = -1 # Force looking through last axis (estimates).
    ob = outlier_bounds(values_, **ob_kwargs)
    good_lo = (ob[0, ...].T <= values_.T).T.all(axis = all_but_last)
    good_hi = (ob[1, ...].T >= values_.T).T.all(axis = all_but_last)
    good = np.logical_and(good_lo, good_hi)
    
    # If it appears all points are outliers, try again with epsilon buffer:
    if not good.any():
        good_lo = (ob[0, ...].T <= values_.T + eps).T.all(axis = all_but_last)
        good_hi = (ob[1, ...].T >= values_.T - eps).T.all(axis = all_but_last)
        good = np.logical_and(good_lo, good_hi)
        if not good.any():
            raise RuntimeError("All results are numerically outliers.")
    
    # Trim the outliers from consideration:
    values = values_[..., good]
    errors = errors_[..., good]
    
    # Form ensemble prediction via specified weighting method (non-outliers):
    if weights == "uniform": # Weight all non-outliers equally.
        deriv = np.mean(values, axis = -1)
        error = np.mean(errors, axis = -1)
    elif weights == "error": # Weight by reciprocal error estimate, entry-wise.
        W = 1.0/np.maximum(errors, eps) # Avoid dividing by zero.
        deriv = np.average(values, axis = -1, weights = W)
        error = np.average(errors, axis = -1, weights = W)
    elif weights == "mean_error": # Weight by reciprocal of avg error estimate.
        W = 1.0/np.maximum(np.mean(errors, axis = all_but_last), eps)
        deriv = np.average(values, axis = -1, weights = W)
        error = np.average(errors, axis = -1, weights = W)
    elif weights == "max_error": # Weight by reciprocal of max error estimate.
        W = 1.0/np.maximum(np.max(errors, axis = all_but_last), eps)
        deriv = np.average(values, axis = -1, weights = W)
        error = np.average(errors, axis = -1, weights = W)
    elif weights == "best": # Choose estimate with smallest error estimate.
        idx = np.argmin(errors)
        deriv = values[..., idx]
    elif weights == "best_mean": # Estimate with smallest mean error entry.
        mean_err = np.mean(errors, axis = all_but_last)
        idx = np.argmin(mean_err)
        deriv = values[..., idx]
        error = values[..., idx]
    elif weights == "best_max": # Estimate with smallest maximum error entry.
        max_err = np.max(errors, axis = all_but_last)
        idx = np.argmin(max_err)
        deriv = values[..., idx]
        error = values[..., idx]
    else: raise ValueError("Invalid weighting method: '%s'" % weights)
    
    # Convert scalar arrays to floats and return results:
    if deriv.shape == (): deriv = float(deriv)
    if error.shape == (): error = float(error)
    if details:
        details["iter_kw"] = iter_kw
        details["values_"] = values_
        details["errors_"] = errors_
        details["good_lo"] = good_lo
        details["good_hi"] = good_hi
        return (deriv, error, details)
    else: return (deriv, error)
