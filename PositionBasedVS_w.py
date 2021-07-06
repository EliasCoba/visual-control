# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 00:03:12 2016

@author: robotics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from CameraModels.PlanarCamera import PlanarCamera
from homography import (homogToRt,H_from_points,Rodrigues)
   

if __name__ == "__main__":

    #Create the world points
    nPoints = 4
    xx       = np.array([-0.5,-0.5,0.5,0.5])
    yy       = np.array([-0.5,0.5,0.5,-0.5])
    zz       = np.array([1.0,1.0,1.0,1.0])
    
    w_points = np.vstack([xx, yy, zz])

    # Set the camera target position
    angle_fixed = 90.

    # Target position
    target_x        = 0.0
    target_y        = 0.0
    target_z        = 0.0
    target_pitch    = np.deg2rad(angle_fixed) # Degrees to radians 'y'
    target_roll     = np.deg2rad(0.0) # Degrees to radians   'z'
    target_yaw      = np.deg2rad(0.0) # Degrees to radians    'x'

    camera1 = PlanarCamera() # Set the target camera
    camera1.setPosition(target_pitch, target_roll, target_yaw, target_x, target_y, target_z)

    p1, wf_2_if_1, Ext_1 = camera1.projection(w_points) # Project the points for camera 1

    # Initial position
    init_x_pos     = 0.8
    init_y_pos     = 0.6
    init_z_pos     = -1.0

    init_pitch   = np.deg2rad(angle_fixed)
    init_yaw     = np.deg2rad(0.0)
    init_roll    = np.deg2rad(0.0)

    camera2 = PlanarCamera() # Set the init camera
    camera2.setPosition(init_pitch, init_yaw, init_roll, init_x_pos, init_y_pos, init_z_pos)
    p2, wf_2_if_2, Ext_2 = camera2.projection(w_points)
    
    #exit
    # Timing parameters
    dt = 0.01   # Time Delta, seconds.
    t0 = 0      # Start time of the simulation
    t1 = 2.0    # End time of  the simulation

    # Initial controls
    v       = np.array([[0],[0],[0]]) # speed in m/s
    omega   = np.array([[0],[0],[0]]) # angular velocity in rad/s
    U       = np.vstack((v,omega))

    # Variables initialization
    steps               = 500                           # Quantity of simulation steps
    UArray              = np.zeros((6,steps))           # Matrix to save controls history
    tArray              = np.zeros(steps)               # Matrix to save the time steps
    pixelCoordsArray    = np.zeros((2*nPoints,steps))   # List to save points positions on the image
    averageErrorArray   = np.zeros(steps)               # Matrix to save error points positions
    positionArray       = np.zeros((3,steps))           # Matrix to save  camera positions

    I               = np.eye(3, 3)
    lambdav         = 2.
    lambdaw         = 6.
    Gain            = np.zeros((6,1))
    Gain[0]         = lambdav
    Gain[1]         = lambdav
    Gain[2]         = lambdav
    Gain[3]         = lambdaw
    Gain[4]         = lambdaw
    Gain[5]         = lambdaw

    t       = t0
    K1      = camera1.K
    K2      = camera2.K
    K1_inv  = np.linalg.inv(K1)
    K2_inv  = np.linalg.inv(K2)
    #camera2.setNoise(0.0001)

    x_pos = init_x_pos
    y_pos = init_y_pos
    z_pos = init_z_pos

    pitch   = init_pitch
    yaw     = init_yaw
    roll    = init_roll

    p20 = []
    
    j = 0
    error_e = 1000
    while( j<steps ):
        # ===================== Calculate new translation and rotation values using Euler's method====================
        x_pos     +=  dt * U[0, 0] 
        y_pos     -=  dt * U[1, 0] # Note the velocities change due the camera framework
        z_pos     +=  dt * U[2, 0]
        
        pitch     +=  dt * U[3, 0]
        yaw       +=  dt * U[4, 0]
        roll      +=  dt * U[5, 0]

        camera2.setPosition(pitch, yaw, roll, x_pos, y_pos, z_pos)
        p2, wf_2_if_2, Ext_2 = camera2.projection(w_points)
                      
        # =================================== Homography =======================================
        #PASO 1.COMPLETAR EL ALGORITMO DE ESTIMACION DE LA MATRIZ HOMOGRAFIA
        H = H_from_points(p1,p2)
        # =================================== Euclidian Homography =======================================
#        He = K2_inv.dot(H.dot(K1));
        #PASO 2.CALCULAR LA MATRIZ HOMOGRAFIA EUCLIDIANA
        He = H
        R,tr,normal = homogToRt(He)

        u = Rodrigues(R)
        # ==================================== CONTROL COMPUTATION =======================================
#        ev       = R.T.dot(tr)
#        ew       = u
#        e        = np.vstack((ev,ew))
#        U        = -Gain*e.reshape((6,1))  
        #PASO 3.INCLUIR LAS ECUACIONES DEL ERROR Y DEL CONTROLADOR
        U        = np.zeros((6,1))                     
        U[1, 0]  = -U[1, 0] # Inverse control due the the camera framework
        #error_e  = np.linalg.norm(e)
 
        #Avoiding numerical error
        U[np.abs(U) < 1.0e-9] = 0.0

        # Copy data for plot
        UArray[:, j]    = U[:, 0]
        tArray[j]       = t

        pixelCoordsArray[:,j] = p2.reshape((2*nPoints,1), order='F')[:,0]

        positionArray[0, j] = x_pos
        positionArray[1, j] = y_pos
        positionArray[2, j] = z_pos

        # =================================== Average feature error ======================================
        pixel_error             = p2-p1
        averageErrorArray[j]    = np.mean(np.linalg.norm(pixel_error.reshape((2*nPoints,1), order='F')))

        t += dt
        j += 1

    # ======================================  Draw cameras ========================================

    fig = plt.figure(figsize=(15,10))
    fig.suptitle('World setting')

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax = fig.gca(projection='3d')
    ax.plot(xx, yy, zz, 'o')

    ax.plot(positionArray[0,0:j],positionArray[1,0:j],positionArray[2,0:j]) # Plot camera trajectory


    axis_scale      = 0.5
    camera_scale    = 0.09
    camera1.DrawCamera(ax, scale=camera_scale, color='red')
    camera1.DrawFrame(ax, scale=axis_scale, c='black')

    camera2.setPosition(pitch, yaw, roll, x_pos, y_pos, z_pos)
    camera2.DrawCamera(ax, scale=camera_scale, color='brown')
    camera2.DrawFrame(ax, scale=axis_scale, c='black')
    
    camera2.setPosition(init_pitch, init_yaw, init_roll, init_x_pos, init_y_pos, init_z_pos)
    camera2.DrawCamera(ax, scale=camera_scale, color='blue')
    camera2.DrawFrame(ax, scale=axis_scale, c='black')
    limit_x = 1.0
    limit_y = 1.0
    limit_z = 1.0

    ax.set_xlabel("$w_x$")
    ax.set_ylabel("$w_y$")
    ax.set_zlabel("$w_z$")
    ax.grid(True)
    ax.set_title('World setting')

    # ======================================  Plot the pixels ==========================================
    ax = fig.add_subplot(2, 2, 2)

    p20 = pixelCoordsArray[:,0].reshape((2,nPoints), order='F')
    ax.plot(p1[0, :],  p1[1, :], 'o', color='red')
    ax.plot(p20[0, :], p20[1, :], 'o', color='blue')

    ax.set_ylim(0, camera1.height)
    ax.set_xlim(0, camera1.width)
    ax.grid(True)

    ax.legend([mpatches.Patch(color='red'),
               mpatches.Patch(color='blue')],
              ['Desired', 'Init'], loc=2)

    ax.plot(pixelCoordsArray[0,0:j], pixelCoordsArray[1,0:j])
    ax.plot(pixelCoordsArray[2,0:j], pixelCoordsArray[3,0:j])
    ax.plot(pixelCoordsArray[4,0:j], pixelCoordsArray[5,0:j])
    ax.plot(pixelCoordsArray[6,0:j], pixelCoordsArray[7,0:j])

    # ======================================  Plot the controls ========================================
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(tArray[0:j], UArray[0, 0:j], label='$V_x$')
    ax.plot(tArray[0:j], UArray[1, 0:j], label='$V_y$')
    ax.plot(tArray[0:j], UArray[2, 0:j], label='$V_z$')
    ax.plot(tArray[0:j], UArray[3, 0:j], label='$\omega_x$')
    ax.plot(tArray[0:j], UArray[4, 0:j], label='$\omega_y$')
    ax.plot(tArray[0:j], UArray[5, 0:j], label='$\omega_z$')
    ax.grid(True)
    ax.legend(loc=0)

    # ======================================  Plot the pixels position ===================================
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(tArray[0:j], averageErrorArray[0:j], label='Average error')
    ax.grid(True)
    ax.legend(loc=0)
    plt.show()

