__author__ = 'aslan'

import numpy as np
from CameraModels.Camera import GenericCamera
from Utils import Rotations
from Utils import Arrow3D


class PlanarCamera(GenericCamera):

    def __init__(self):

        """


        """
        GenericCamera.__init__(self)

        self.xi  = 0.0
        self.psi = 1.0

    def setEMatrix(self):

        self.E =  np.eye(3, 3)

    def RotatePlanarAxisX(self):

        """


        :return:
        """
        temp    = np.dot(Rotations.rotox(-np.pi/2.0), self.R)
        return temp.T

    def DrawCamera(self, ax, color='cyan', scale=1.0):

        #CAmera points: to be expressed in the camera frame;
        CAMup=scale*np.array([
            [-1,-1,  1, 1, 1.5,-1.5,-1, 1 ],
            [ 1, 1,  1, 1, 1.5, 1.5, 1, 1 ],
            [ 2,-2, -2, 2,   3,   3, 2, 2 ],
        ])

        Ri2w    = np.dot(Rotations.rotox(-np.pi/2.0), self.R)
        trasl   = self.t.reshape(3, -1)

        CAMupTRASF = Ri2w.dot(CAMup) + trasl;


        CAMdwn=scale*np.array([
            [-1,-1,  1, 1, 1.5,-1.5,-1, 1  ],
            [ -1,-1, -1,-1,-1.5,-1.5,-1,-1 ],
            [  2,-2, -2, 2,   3,   3, 2, 2 ]
        ])

        CAMdwnTRASF     = Ri2w.dot( CAMdwn ) + trasl
        CAMupTRASFm     = CAMupTRASF
        CAMdwnTRASFm    = CAMdwnTRASF



        ax.plot(CAMupTRASFm[0,:],CAMupTRASFm[1,:],CAMupTRASFm[2,:],c=color)
        ax.plot(CAMdwnTRASFm[0,:],CAMdwnTRASFm[1,:],CAMdwnTRASFm[2,:],c=color)
        ax.plot([CAMupTRASFm[0,0],CAMdwnTRASFm[0,0]],[CAMupTRASFm[1,0],CAMdwnTRASFm[1,0]],[CAMupTRASFm[2,0],CAMdwnTRASFm[2,0]],c=color)
        ax.plot([CAMupTRASFm[0,1],CAMdwnTRASFm[0,1]],[CAMupTRASFm[1,1],CAMdwnTRASFm[1,1]],[CAMupTRASFm[2,1],CAMdwnTRASFm[2,1]],c=color)
        ax.plot([CAMupTRASFm[0,2],CAMdwnTRASFm[0,2]],[CAMupTRASFm[1,2],CAMdwnTRASFm[1,2]],[CAMupTRASFm[2,2],CAMdwnTRASFm[2,2]],c=color)
        ax.plot([CAMupTRASFm[0,3],CAMdwnTRASFm[0,3]],[CAMupTRASFm[1,3],CAMdwnTRASFm[1,3]],[CAMupTRASFm[2,3],CAMdwnTRASFm[2,3]],c=color)
        ax.plot([CAMupTRASFm[0,4],CAMdwnTRASFm[0,4]],[CAMupTRASFm[1,4],CAMdwnTRASFm[1,4]],[CAMupTRASFm[2,4],CAMdwnTRASFm[2,4]],c=color)
        ax.plot([CAMupTRASFm[0,5],CAMdwnTRASFm[0,5]],[CAMupTRASFm[1,5],CAMdwnTRASFm[1,5]],[CAMupTRASFm[2,5],CAMdwnTRASFm[2,5]],c=color)

    def DrawFrame(self, ax, c='red', scale=2.0, _scale_x=None, _scale_y=None, _scale_z=None):

        """

        :param ax:
        :param H:
        :param c:
        :param scale:
        """
        if ax is None:
            ax = plt.gca()

        scale_x = _scale_x
        scale_y = _scale_y
        scale_z = _scale_z

        if scale_x is None:
            scale_x = scale

        if scale_y is None:
            scale_y = scale

        if scale_z is None:
            scale_z = scale

        R = np.dot(Rotations.rotox(-np.pi/2.0), self.R)
        t = self.t.reshape(3, -1)

        # Camera reference frame
        Oc = scale   * np.array([[0.,0,0]]).T
        Xc = scale_x * np.array([[1.,0,0]]).T
        Yc = scale_y * np.array([[0.,1,0]]).T
        Zc = scale_z * np.array([[0.,0,1]]).T

        Ri2w    = R;
        Oc1     = Ri2w.dot(Oc) + t
        Xc1     = Ri2w.dot(Xc) + t
        Yc1     = Ri2w.dot(Yc) + t
        Zc1     = Ri2w.dot(Zc) + t


        a1 = Arrow3D.Arrow3D([Oc1[0,0],Xc1[0,0]],[Oc1[1,0],Xc1[1,0]],[Oc1[2,0],Xc1[2,0]], mutation_scale=20, lw=1, arrowstyle="-|>", color=c)
        a2 = Arrow3D.Arrow3D([Oc1[0,0],Yc1[0,0]],[Oc1[1,0],Yc1[1,0]],[Oc1[2,0],Yc1[2,0]], mutation_scale=20, lw=1, arrowstyle="-|>", color=c)
        a3 = Arrow3D.Arrow3D([Oc1[0,0],Zc1[0,0]],[Oc1[1,0],Zc1[1,0]],[Oc1[2,0],Zc1[2,0]], mutation_scale=20, lw=1, arrowstyle="-|>", color=c)

        ax.add_artist(a1)
        ax.add_artist(a2)
        ax.add_artist(a3)

        ax.text(Xc1[0,0], Xc1[1,0], Xc1[2,0], (r'$X_{cam}$'))
        ax.text(Yc1[0,0], Yc1[1,0], Yc1[2,0], (r'$Y_{cam}$'))
        ax.text(Zc1[0,0], Zc1[1,0], Zc1[2,0], (r'$Z_{cam}$'))
