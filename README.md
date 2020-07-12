# pyDERIVEST
This module is an expanded port of John D'Errico's MATLAB package, available at the [MathWorks File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/13490-adaptive-robust-numerical-differentiation). The ``derivest`` suite provides methods for numerical differentiation of (real- or vector-valued) analytically-defined functions of one of several variables; it cannot be used with collections of input-output pairs. For additional information, see [the documentation](https://github.com/njwichrowski/pyDERIVEST/blob/master/Documentation.md).

The main routine is ``derivest.derivest``, which implements a finite difference scheme and Romberg extrapolation to estimate derivatives up to fourth order of scalar-valued functions of a single scalar. Additional wrapper routines are provided for directional derivatives, gradients, Hessians, and Jacobians.

## Significant Additions and Changes
1. The ``max_step`` parameter now functions on the basis of absolute step size, rather than a multiple of the magnitude of the location of differentiation. This helps prevent certain round-off errors from occuring near the origin, but the value of ``max_step`` may need to be increased when differentiating at points where the function changes rapidly.
2. In view of numerical idiosyncrasies observed when using the original MATLAB version, a new method (``ensemble``) has been added that allows aggregated evaluation of the "base" methods in order to yield more accurate estimates than a single computation.

At the moment, there is a bug in the routine used for the uncentered (``"style" in ["forward", "backward"]``) finite difference schemes.