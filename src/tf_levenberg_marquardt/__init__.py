from importlib.metadata import metadata

import tf_levenberg_marquardt.damping as damping
import tf_levenberg_marquardt.loss as loss
import tf_levenberg_marquardt.training as training
import tf_levenberg_marquardt.model as model

__all__ = ['damping', 'loss', 'training', 'model']

# Dynamically load metadata from pyproject.toml
meta = metadata('tf-levenberg-marquardt')

__version__ = meta['Version']
__description__ = meta['Summary']
