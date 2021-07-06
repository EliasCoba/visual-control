__author__ = 'aslan'


from CameraModels.Camera import GenericCamera


class ParabolicCamera(GenericCamera):

    def __init__(self, p):

        """

        :param p:
        """
        #super(ParabolicCamera, self).__init__()
        GenericCamera.__init__(self)
        self.xi  = 1.0
        self.psi = 1.0 + (2.0*p)








