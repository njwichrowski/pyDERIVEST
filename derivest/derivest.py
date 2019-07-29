# -*- coding: utf-8 -*-

from . import utils

import numpy as np
import warnings

# Default values for keyword arguments:
__kwargs = {"deriv_order": 1,
            "fixed_step": None,
            "max_step": 1.0,
            "method_order": 4,
            "romberg_terms": 2,
            "step_ratio": 2.0000001,
            "style": "central",
            "vectorized": True}

# Valid values for keyword arguments:
__acceptable_values = {"deriv_order": [1, 2, 3, 4],
                       "fixed_step": lambda x: x is None or x > 0,
                       "max_step": lambda x: x > 0,
                       "method_order": [1, 2, 3, 4],
                       "romberg_terms": [0, 1, 2, 3],
                       "step_ratio": lambda x: x > 1,
                       "style": ["central", "forward", "backward"],
                       "vectorized": [True, False]}

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
        
    """
    ##### DETERMINE PARAMETER VALUES #####
    p = __kwargs.copy() # Start with default values.
    for key in kwargs:
        if key not in p: raise TypeError("derivest() got an unrecognized keyword argument: %s" % key)
        p[key] = kwargs[key] # Replace with specified values.
        
        # Check for parameter value validity, either by membership in a given
        # list or by evaluation of a function as True:
        if isinstance(__acceptable_values[key], list):
            if p[key] not in __acceptable_values[key]:
                raise ValueError("Argument '%s' received an invalid value: %s" % (key, p[key]))
        elif not __acceptable_values[key](p[key]):
            raise ValueError("Argument '%s' received an invalid value: %s" % (key, p[key]))
    
    # Perform one additional check for a combination of values:
    if p["style"] == "central" and p["method_order"] % 2 == 1:
        raise ValueError("Cannot perform a 'central' method of odd order.")
    elif p["style"] != "central":
        warnings.warn("Using style = '%s' is highly unstable and not recommended." % p["central"], RuntimeWarning)
    p["step_ratio"] = float(p["step_ratio"]) # Avoid "ValueError: Integers to negative integer powers are not allowed."
    if par is None: par = [] # Avoid TypeError from attempting *None when par not provided.
    
    ##### SET THE STEPS TO USE #####
    if isinstance(x, list): x = np.array(x)
    try:
        N = x.size # At how many points must we compute the derivative?
        number = False
    except AttributeError: # A single number was provided.
        x = np.array([x])
        N = 1
        number = True
    if p["fixed_step"] is None: # Use a basic sequence of steps, relative to a unit step.
        delta = p["max_step"]*p["step_ratio"]**-np.arange(26)
        num_delta = len(delta)
    else: # Fixed, user supplied absolute sequence of steps.
        num_delta = int(3 + np.ceil(p["deriv_order"]/2) + p["method_order"] + p["romberg_terms"])
        if p["style"] == "central": num_delta -= 2
        delta = p["fixed_step"]*p["step_ratio"]**-np.arange(num_delta)
    
    ##### GENERATE FINITE DIFFERENCE APPROXIMATION RULE IN ADVANCE #####
    fda_rule = None
    # The rule is for a nominal unit step size, and will
    # be scaled later to reflect the local step size.
    if p["style"] == "central":
        # For central rules, we will reduce the load by an
        # even or odd transformation, as appropriate.
        if p["method_order"] == 2:
            if p["deriv_order"] == 1:
                fda_rule = np.array([1.0]) # The odd transformation did all the work
            elif p["deriv_order"] == 2:
                fda_rule = np.array([2.0]) # The even transformation did all the work
            elif p["deriv_order"] == 3: # Odd transformation did most of the work, but need to remove linear term.
                fda_rule = np.linalg.lstsq(utils.fdamat(p["step_ratio"], 1, 2).T, np.array([[0], [1]]), rcond = None)[0].T
            elif p["deriv_order"] == 4: # Even transformation did most of the work, but need to remove quadratic term.
                fda_rule = np.linalg.lstsq(utils.fdamat(p["step_ratio"], 2, 2).T, np.array([[0], [1]]), rcond = None)[0].T
        else: # Implies method_order == 4, since style == "central".
            if p["deriv_order"] == 1: # Odd transformation did most of the work, but need to remove cubic term.
                fda_rule = np.linalg.lstsq(utils.fdamat(p["step_ratio"], 1, 2).T, np.array([[1], [0]]), rcond = None)[0].T
            elif p["deriv_order"] == 2:# Even transformation did most of the work, but need to remove quartic term.
                fda_rule = np.linalg.lstsq(utils.fdamat(p["step_ratio"], 2, 2).T, np.array([[1], [0]]), rcond = None)[0].T
            elif p["deriv_order"] == 3: # Odd transformation did much of the work, but need to remove linear, quintic terms.
                fda_rule = np.linalg.lstsq(utils.fdamat(p["step_ratio"], 1, 3).T, np.array([[0], [1], [0]]), rcond = None)[0].T
            elif p["deriv_order"] == 4: # Even transformation did much of the work, but need to remove quadratic, sextic terms.
                fda_rule = np.linalg.lstsq(utils.fdamat(p["step_ratio"], 2, 3).T, np.array([[0], [1], [0]]), rcond = None)[0].T
    else: # The "forward" and "backeard" cases are identical, except at the end.
        if p["method_order"] == 1: # No odd/even transformations, but we already removed the constant term.
            if p["deriv_order"] == 1: fda_rule = np.array([1.0]) # An easy one.
            else: # [2, 3, 4]
                v = np.zeros((p["deriv_order"], 1))
                v[p["deriv_order"], 0] = 1.0
                fda_rule = np.linalg.lstsq(utils.fdamat(p["step_ratio"], 0, p["deriv_order"]).T, v, rcond = None)[0].T
        else: # p["method_order"] methods drop off the lower order terms, plus terms directly above p["deriv_order"]
            nt = p["deriv_order"] + p["method_order"] - 1
            v = np.zeros((nt, 1))
            v[p["deriv_order"], 0] = 1.0
            fda_rule = np.linalg.lstsq(utils.fdamat(p["step_ratio"], 0, nt).T, v, rcond = None)[0].T
        if p["style"] == "backward": fda_rule = -fda_rule # Correct sign for "backward" rule.
    if fda_rule is None: raise RuntimeError("Could not generate a finite difference approximation rule.")
    num_fda = fda_rule.size
    
    ##### APPROXIMATE THE DERIVATIVE AT EACH POINT #####
    x_shape = x.shape # Store the original shape of x, flatten it for easy
    x = x.flatten().astype(np.float64) # indexing, and convert to 64-bit float.
    
    # Evaluate fun at the point x itself, if required:
    if p["deriv_order"] % 2 == 0 or p["style"] != "central":
        if p["vectorized"] or N == 1:
            fx = fun(x, *par)
            if fx.shape == (): fx = np.reshape(fx, (1,)) # Convert numpy scalar to length-one vector.
            if x.shape != fx.shape:
                if N > 1: raise RuntimeError("fun() returned an array with an unexpected shape. Try specifying 'vectorized = False'.")
                else: raise RuntimeError("fun() returned an array when given a scalar argument.")
        else: fx = utils.loop_eval(fun, x, par) # Not vectorized, so must loop.
    else: fx = np.array([]) # Don't need fun(x) if central difference and odd derivative order.
    
    ##### LOOP OVER ELEMENTS OF x TO REDUCE TO A SCALAR PROBLEM ####
    der = np.zeros(N)
    err = np.zeros(N)         # Start with flattened arrays.
    final_delta = np.zeros(N) # Reshape before returning.
    for i in range(N):
        # Below, fev is the set of all the function evaluations we
        # will generate. For a central rule, it will have the even
        # or odd transformation built in.
        try:
            if p["style"] == "central": # We must evaluate symmetrically around xi.
                if p["vectorized"]:
                    f_plus  = fun(x[i] + delta, *par)
                    f_minus = fun(x[i] - delta, *par)
                else:
                    f_plus  = np.zeros(delta.shape)
                    f_minus = np.zeros(delta.shape)
                    for j in range(num_delta):
                        f_plus[j]  = fun(x[i] + delta[j], *par)
                        f_minus[j] = fun(x[i] - delta[j], *par)
                if p["deriv_order"] in [1, 3]: fev = (f_plus - f_minus)/2.0 # Odd transformation.
                else: fev = (f_plus + f_minus)/2.0 - fx[i] # Even transformation.
            elif p["style"] == "forward": # Drop off the constant only.
                if p["vectorized"]: fev = fun(x + delta) - fx[i]
                else:
                    fev = np.zeros(delta.shape)
                    for j in range(num_delta): fev[j] = fun(x + delta[j]) - fx[i]
            else: # Backward rule; drop off the constant only.
                if p["vectorized"]: fev = fx[i] - fun(x - delta) # Swapped order from "fun(x - h*delta) - fx[i]" to fix error.
                else:
                    fev = np.zeros(delta.shape)
                    for j in range(num_delta): fev[j] = fx[i] - fun(x - delta[j]) # Swapped order from "fun(x - h*delta[j]) - fx[i]".
        except ValueError as exc:
            raise RuntimeError("Dimension mismatch in derivest():\n'%s'\nTry using 'vectorize = False'." % exc) from exc
        
        # Check the size of f_del to ensure it was properly vectorized:
        if fev.size != num_delta: raise RuntimeError("fun() did not return the correct size result; it must be vectorized.")
        
        ##### APPLY FD RULE AT EACH DELTA, SCALING AS APPROPRIATE #####
        num_estim = num_delta + 1 - num_fda - p["romberg_terms"] # Number of estimates to use.
        
        # Form initial derivative estimates from chosen FD method:
        der_init = utils.diag_tile(fev, (num_estim, num_fda)) @ fda_rule.T
        der_init = der_init.flatten()/(delta[:num_estim])**p["deriv_order"]
        
        # Each approximation that results is an approximation of order
        # par.DerivativeOrder to the desired derivative. Additional (higher-
        # order, even or odd) terms in the Taylor series also remain. Use a
        # multi-term Romberg extrapolation to improve these estimates:
        if p["style"] == "central": romb_expon = p["method_order"] + 2*np.arange(p["romberg_terms"])
        else: romb_expon = p["method_order"] + np.arange(p["romberg_terms"])
        (der_romb, errors) = utils.romb_extrap(p["step_ratio"], der_init, romb_expon)
        
        ##### CHOOSE WHICH RESULT TO RETURN #####
        if p["fixed_step"] is None:
            # Trim off the estimates at each end of the scale:
            num_est = der_romb.size
            if p["deriv_order"] == 1: trim = np.array([0, 1, num_est - 2, num_est - 1])
            else: trim = np.r_[np.arange(2*(p["deriv_order"] - 1)), num_est + np.arange(2*(1 - p["deriv_order"]), 0)]
            idx = np.delete(np.argsort(der_romb), trim) # Get indices of non-extremal values.
            der_romb = der_romb[idx] # View through idx for trimmed, sorted array.
            errors = errors[idx]
            trim_delta = delta[idx]
            idx = np.argmin(errors) # Find index of smallest (non-trimmed) error.
            err[i] = errors[idx]
            final_delta[i] = trim_delta[idx]
            der[i] = der_romb[idx] # Use the corresponding derivative estimate.
        else:
            idx = np.argmin(errors) # Find index of smallest error.
            err[i] = errors[idx]
            final_delta[i] = delta[idx]
            der[i] = der_romb[idx] # Use the corresponding derivative estimate.
    der = np.reshape(der, x_shape)
    err = np.reshape(err, x_shape)
    final_delta = np.reshape(final_delta, x_shape)
    if number: return (float(der), float(err), float(final_delta))
    else: return (der, err, final_delta)
