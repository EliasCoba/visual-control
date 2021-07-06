__author__ = 'aslan'

import numpy as np
from Utils import Rotations


class GenericCamera:
    #Properties(read / write)::
        # pp intrinsic: principal point(2 x1)
        # rho intrinsic: pixel dimensions(2 x1) in metres
        # f intrinsic: focal length
        # k intrinsic: radial distortion vector
        # p intrinsic: tangential distortion parameters
        # distortion intrinsic: camera distortion[k1 k2 k3 p1 p2]
        # R: camera pose in world coordinates
        # t: camera translation in world coordinates
        # T extrinsic: camera pose as homogeneous transformation

    def __init__(self):

        self.name = None

        # Image properties
        self.height = None
        self.width  = None

        # Intrinsic camera parameters

        self.K   = None
        self.rho = None  # pixel dimensions 1x2
        self.pp  = None  # principal point 1x2

        self.f = None  # Focal length

        # Extrinsic camera parameter

        self.R = None
        self.t = None
        self.T = None  # camera pose

        # For catadioptric cameras

        self.xi  = None
        self.psi = None

        self.E   = None

        # Projection matrix K*E
        self.P = None

        # For noise
        self.noise          = None  # pixel noise 1x2
        self.sigma_noise    = None

        self.setDefaultCamera()

    def setImageSize(self, _size):
        """
        Set the image size
        :param _size:
        :raise "Bad vector shape":
        """
        if _size.shape == (1, 2):
            self.width  = _size[0, 0]
            self.height = _size[0, 1]

        elif _size.shape == (2, 1):
            self.width  = _size[0, 0]
            self.height = _size[1, 0]
        else:
            raise Exception ("Bad image size shape")

    def setPrincipalPoint(self, _pp=None):

        """
        Set the principal point.
        If _pp is None and the image with and height exist set the pp to image center.
        If _pp is None and the image size == 0 is set to 0
        :param _pp: principal point 1x2 or 2x1
        """
        if _pp is None:
            if self.width is not None and self.height is not None:
                self.pp = np.array([[self.width / 2.0, self.height / 2.0]])
            else:
                self.pp = np.array([[0, 0]])

        elif _pp.shape == (1, 2):
            self.pp = _pp

        elif _pp.shape == (2, 1):
            self.pp = np.transpose(_pp)

        else:
            raise Exception( "Bad principal point dimensions")

    def setPixelsSize(self, value):
        """

        :param value: Pixel size vector
        """
        if value is None:
            self.rho = np.array([10.0e-6, 10.0e-6])

        elif value.shape == (1, 2):
            self.rho = value

        elif value.shape == (2, 1):
            self.rho = np.transpose(value)

        else:
            raise "Bad pixel size vector dimensions"

    def setIntrinsicMatrixParameters(self, _K):

        """
        Set the intrinsic parameters matrix
        :param _K: Intrinsic parameters matrix
        :raise "Bad matrix dimensions on intrinsic parameters":
        """
        if _K is None:
            self.setIntrinsicParametersDefault()
            self.K = np.array([
                [self.f / self.rho[0,0], 0.0, self.pp[0,0]],
                [0.0, self.f / self.rho[0,1], self.pp[0,1]],
                [0.0, 0.0, 1.0]
            ])

        elif _K.shape == (3, 3):
            self.K = _K

            self.width  = _K[0,2]*2.0
            self.height = _K[1,2]*2.0

            self.setPrincipalPoint()

        else:
            raise Exception ("Bad matrix dimensions on intrinsic parameters")

    def setExtrinsicMatrixParameters(self, _T=None):

        """
        Set the extrinsic parameters matrix
        :param _T: Extrinsic parameters matrix in homogeneous
        :raise "Bad matrix dimensions in extrinsic parameters":
        """
        if _T is None:
            self.T = np.eye(4, 4)


        elif _T.shape == (4, 4):
            self.T = _T
            self.R = _T[0:3, 0:3]
            self.t = np.array([
                [
                    _T[0,3],
                    _T[1,3],
                    _T[2,3]
                ]

            ])
        else:
            raise  Exception ("Bad matrix dimensions in extrinsic parameters")

    def setPosition(self, rot_x, rot_y, rot_z, x_pos, y_pos, z_pos):
        """

        :param rot_x: Rotation over axis X
        :param rot_y: Rotation over axis Y
        :param rot_z: Rotation over axis Z
        :param x_pos: Translation over axis X
        :param y_pos: Translation over axis Y
        :param z_pos: Translation over axis Z
        """

        self.R = np.dot(Rotations.rotox(rot_x), np.dot(Rotations.rotoy(rot_y), Rotations.rotoz(rot_z)))
        self.t = np.array([
            [
                x_pos,
                y_pos,
                z_pos
            ]

        ])


    def setEMatrix(self):

        self.E = np.array([

            [self.psi - self.xi,        0.0,                0.0],
            [0.0,                       (self.xi-self.psi), 0.0],
            [0.0,                       0.0,                1.0]
        ])

    def setIntrinsicParametersDefault(self):

        """

        """
        self.f      = 0.002 #0.002
        self.rho    = np.array([ [10e-6, 10e-6]]) # square pixels 10um side

        # 1 Mpix image plane
        self.width  = 640
        self.height = 480

        self.setPrincipalPoint(None)


    def setDefaultCamera(self):
        """

        """
        self.setIntrinsicParametersDefault()
        self.setIntrinsicMatrixParameters(None)


    ##################################################################################################
    def setIntrinsicParameters(self, focal_length, rho, width, height):
        """

        :param focal_length: Camera focal length
        :param rho: Pixel size
        :param width:  Image width
        :param height: Image height
        :return:
        """
        self.width  = width
        self.height = height

        self.f      = focal_length
        self.rho    = rho
        self.pp     = np.array([[self.width / 2.0, self.height / 2.0]])

        self.K = np.array([
            [self.f / self.rho[0,0], 0.0, self.pp[0,0]],
            [0.0, self.f / self.rho[0,1], self.pp[0,1]],
            [0.0, 0.0, 1.0]
        ])

        return self.K

    def RotatePlanarAxisX(self):

        """


        :return:
        """
        return self.R.T

    def setNoise(self,_sigma):

        if _sigma is None:
            self.noise       = False
            self.sigma_noise = 0.0
        else:
            self.noise = True
            self.sigma_noise = _sigma

    ##################################################################################################
    def projection(self, points):

        """
        This function computes the perspective projection of a set of feature points U expressed in the world-frame.
        :param points: "matrix of points in 3d scene"(can be also in homogeneous notation)
        [x1 x2 x3 .... xn]
        [y1 y2 y3 .... yn]
        [z1 z2 z3 .... zn]
        :return:
        """

        if points.shape[0] == 3:
            Uo = np.vstack((points, np.ones((1, points.shape[1]))))
        else:
            Uo = points

        self.setEMatrix()

        self.P = np.dot(self.K, self.E)

        H       = np.eye(4, 4)

        # Make the camera the origin with the inverse transformation
        Rw2i    = self.RotatePlanarAxisX()
        tw2i    = -np.dot(Rw2i, self.t.T)

        H[0:3, 0:3]  = Rw2i
        H[0:3, 3]    = tw2i.T

        # World points to image frame
        points_p = np.dot(H, Uo)

        points_p = points_p[0:3, :]

        rows, cols = points_p.shape

        result = np.zeros([rows, cols])

        for i in xrange(0, cols):

            norm   = np.linalg.norm(points_p[:, i])
            if norm != 0.0:
                x      = points_p[:, i]/norm
            else:
                raise  Exception('Division by zero: possible wrong world setting')

            x[2]  += self.xi

            #PASO 1: COMPLETAR EL MODELO DE PROYECCION
            if x[2] != 0.0:
                x /= x[2]
            result[:, i] = points_p[:, i]
            
            #print points_p
            result[:, i] = np.dot(self.P, x)

        if self.noise is True:
            noise_matrix = np.random.normal(0,self.sigma_noise,size=result.shape)

            result += noise_matrix


        return result[0:2, :], points_p, H


    ##################################################################################################
    def getImageSize(self):

        """
        Get image size
        :return:
        """
        return np.array([[self.width, self.height]])


    ##################################################################################################
    @property
    def name(self):
        return self.name

    @name.setter
    def name(self, value):
        if value is not None:
            self.name = value.title()
        else:
            self.name = "Basic Camera"

    @name.deleter
    def name(self):
        del self.name

    ##################################################################################################
    @property
    def K(self):
        return self.K

    @K.setter
    def K(self, value):

        if value is None:
            self.K = np.array([
                [self.f / self.rho[0], 0.0, self.pp[0]],
                [0.0, self.f / self.rho[1], self.pp[1]],
                [0.0, 0.0, 1.0]
            ])
        else:
            self.setIntrinsicMatrix(value)

    @K.deleter
    def K(self):
        del self.K

    ##################################################################################################

    @property
    def T(self):
        return self.T

    @T.setter
    def T(self, value):
            self.setExtrinsicMatrixParameters(value)

    @T.deleter
    def T(self):
        del self.T

    ##################################################################################################
    @property
    def rho (self):
        return self.rho

    @rho.setter
    def rho (self, value):
        self.setPixelsSize(value)

    @rho.deleter
    def rho (self):
        del self.rho


    ##################################################################################################
    @property
    def P(self):
        return self.P

    @P.setter
    def P(self, value):
        if value is None:
            self.P = np.dot(self.K, self.E)
        else:
            if value.shape == (3,3):
                self.P = value
            else:
                raise  Exception("Bad P matrix dimensions")


    ##################################################################################################
    @property
    def f (self):
        return self.f

    @f.setter
    def f (self, value):
        if value is None:
            self.f = 8.0e-3
        else:
            if value > 0.0:
                self.f = value
            else:
                raise  Exception("Bad focal length")


##################################################################################################
    @property
    def pp (self):
        return self.rho

    @pp.setter
    def pp (self, value):
        self.setPrincipalPoint(value)
