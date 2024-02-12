# Built-in imports

# from api import logging

# Third-party imports

# Darkest imports
from darkest import qtc
from darkest import tfk

import darkest.Core as da




class DNN(tfk.Model):
  """Keras Model subclass wrapper with convinience methods.
  """
  fimodel = ...
  fimodelcp = ...


  @classmethod
  def cb_checkpoint(cls, path:str=None, best:bool=True):
    """Call back for saving the best model during training.
    """
    if not path: path = cls.fimodelcp.absoluteFilePath()

    da.iofile.mkdir(qtc.QFileInfo(path))

    return tfk.callbacks.ModelCheckpoint(
      filepath=path,
      monitor="val_accuracy",
      mode="max",
      save_weights_only=False,
      save_best_only=best,
      verbose=1,
    )


  def call(self, inputs):
    """Custom call method."""
    return self.model(inputs)

