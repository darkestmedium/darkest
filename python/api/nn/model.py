# Built-in imports

# from api import logging

# Third-party imports

# Darkest imports
from api import qtc
from api import tfk

import api.Darkest as da




@tfk.saving.register_keras_serializable()
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
      monitor="accuracy",
      mode="max",
      save_best_only=best,
      verbose=1,
    )


  def call(self, inputs):
    """Custom call method."""
    return self.model(inputs)

