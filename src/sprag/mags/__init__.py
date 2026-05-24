"""MAGS: manifold-guided steering for residual stream correction."""
from .calibrate import collect_residuals, fit_mags, MAGSParams
from .intervene import mags_hook
