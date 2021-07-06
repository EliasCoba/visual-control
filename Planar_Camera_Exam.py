import numpy as np
import matplotlib.pyplot as plt

from CameraModels.PlanarCamera import PlanarCamera


if __name__ == "__main__":

    nPoints = 4
    x, y, z = np.mgrid[0:nPoints / 2, 0:1, 0:nPoints / 2]

    xx = x.flatten()
    yy = y.flatten()
    zz = z.flatten()

    w_points = np.vstack([xx, yy, zz])

    # Set the camera
    camera = PlanarCamera()

    # Extrinsic camera matrix
    T = np.eye(3, 4)

    # Intrinsic camera parameters
    K = np.array([
        [2.8145167816086894e+02, 0.,                     1.6718968649606481e+02],
        [0.,                     2.8032922369948636e+02, 1.2826276507252587e+02],
        [0.,                     0.,                     1.]
    ])

    camera.setIntrinsicMatrixParameters(K)
    focal_length = K[0,0]

    # Camera position
    cam_x = 2.0
    cam_y = -9.0
    cam_z = 0.0

    cam_rot_x = 0.0
    cam_rot_y = np.deg2rad(-12.5288077092)
    cam_rot_z = 0.0


    camera.setPrincipalPoint()
    camera.setPosition(cam_rot_x,cam_rot_y,cam_rot_z,cam_x,cam_y,cam_z)

    uv, wf_2_if, Ext = camera.projection(w_points)

    # Use only points inside image
    w_index = uv[0,:]<camera.width
    h_index = uv[1,:]<camera.height

    index = w_index & w_index

    uu = uv[0,:]
    uu = uu[index]

    vv = uv[1,:]
    vv = vv[index]

    uv = np.vstack([uu,vv])

    z_world_points = wf_2_if[2,:]

    # Plot camera
    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    ax.plot(xx, yy, zz, 'o')

    camera.DrawCamera(ax, scale=0.5,color='black')
    camera.DrawFrame(ax, scale=3.0)
    limit_x = 5
    limit_y = 10
    ax.set_xlim3d((-limit_x, limit_x))
    ax.set_ylim3d((-limit_y, 4))
    ax.set_zlim3d((-5, 7))
    ax.set_xlabel("$w_x$")
    ax.set_ylabel("$w_y$")
    ax.set_zlabel("$w_z$")
    ax.grid()
    ax.set_title('World setting')

    # Draw the image
    fig_image = plt.figure()
    ax2 = fig_image.add_subplot(111)
    ax2.plot(uv[0,:],uv[1,:],'o')
    ax2.set_xlim([0,camera.width])
    ax2.set_ylim([0,camera.height])
    ax2.set_xlabel("u(pixels)")
    ax2.set_ylabel("v(pixels)")
    ax2.set_title('Image')



    plt.show()