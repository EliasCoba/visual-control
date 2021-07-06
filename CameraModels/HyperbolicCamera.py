__author__ = 'aslan'

import numpy as np
from CameraModels.Camera import GenericCamera


class HyperbolicCamera(GenericCamera):

    def __init__(self, d, p):

        """

        :param p:
        """

        GenericCamera.__init__(self)

        temp     = np.sqrt((d*d) + 4.0*(p*p))
        self.xi  = d / temp
        self.psi = (d + 2.0*p) / temp