import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from CameraModels.PlanarCamera import PlanarCamera
from Utils.ImageJacobian import ImageJacobian

if __name__ == "__main__":

    #Create the world points
    nPoints = 4
    x, y, z = np.mgrid[-0.5:1, 2:3, -0.5:1]

    xx = x.flatten()
    yy = y.flatten()
    zz = z.flatten()

    w_points = np.vstack([xx, yy, zz])

    # Target position
    target_x        = 0
    target_y        = 0
    target_z        = 0
    target_pitch    = np.deg2rad(0.0) # Degrees to radians
    target_roll     = np.deg2rad(0.0) # Degrees to radians
    target_yaw      = np.deg2rad(0.0) # Degrees to radians

    camera1 = PlanarCamera() # Set the target camera
    camera1.setPosition(target_pitch, target_roll, target_yaw, target_x, target_y, target_z)
    K1      = camera1.K
    K1_inv  = np.linalg.inv(K1)

    #Cambia matriz de PInt   
    K3 = np.array([[ 300.,    0.,  320.],
                   [   0.,  300.,  240.],
                   [   0.,    0.,    1.]])
    camera1.setIntrinsicMatrixParameters(K1)

    #camera1.setIntrinsicParameters(0.003, np.array([ [10e-6, 10e-6]]), 640, 480)

    #PASO 1.COMPLETAR EL MODELO DE PROYECCION DE LA CAMARA
    p1, wf_2_if_1, Ext_1 = camera1.projection(w_points) # Project the points for camera 1

    # Initial position
    init_x_pos     = 0
    init_y_pos     = -1
    init_z_pos     = 0

    init_pitch   = np.deg2rad(0.0)
    init_yaw     = np.deg2rad(0.0)
    init_roll    = np.deg2rad(0.0)

    camera2 = PlanarCamera() # Set the init camera
    camera2.setPosition(init_pitch, init_yaw, init_roll, init_x_pos, init_y_pos, init_z_pos)
    K2      = camera2.K
    K2_inv  = np.linalg.inv(K2)

    #PASO 2.CAMBIAR LOS PARAMETROS INTRINSECOS DE AMBAS CAMARAS Y OBSERVAR EL EFECTO EN LAS IMAGENES
    #PASO 3.CAMBIAR A DIFERENTES CONDICIONES INICIALES EN SOLO UN GDL DE LA CAMARA 2 Y OBSERVAR EL EFECTO EN LAS IMAGENES

    camera2.setIntrinsicMatrixParameters(K2)
    
    #camera2.setIntrinsicParameters(0.003, np.array([ [10e-6, 10e-6]]), 640, 480)
    
    #PASO 11.INCLUIR RUIDO DE IMAGEN EN EL CONTROL COMPLETO    
    camera2.setNoise(0.0) #0.5
        
    # Timing parameters
    dt = 0.01   # Time Delta, seconds.
    t0 = 0      # Start time of the simulation
    t1 = 3.0    # End time of  the simulation
    t       = t0
    
    # Initial controls
    v       = np.array([[0],[0],[0]]) # speed in m/s
    omega   = np.array([[0],[0],[0]]) # angular velocity in rad/s
    U       = np.vstack((v,omega))

    # Variables initialization
    steps               = int((t1 - t0)/dt + 1)         # Quantity of simulation steps
    UArray              = np.zeros((6,steps))           # Matrix to save controls history
    tArray              = np.zeros(steps)               # Matrix to save the time steps
    pixelCoordsArray    = np.zeros((2*nPoints,steps))   # List to save points positions on the image
    averageErrorArray   = np.zeros(steps)               # Matrix to save error points positions
    positionArray       = np.zeros((3,steps))           # Matrix to save  camera positions

    I               = np.eye(3, 3)
    #PASO 9.PROBAR EL CONTROL COMPLETO PARA DIFERENTES VALORES DE GANANCIA UNICA Y DIFERENTE PARA POS Y ORI
    lambda_value    = 0
    Gain            = lambda_value * np.eye(6, 6)

    #PASO 10.PROBAR EL CONTROL COMPLETO PARA DIFERENTES VALORES DE PROFUNDIDAD ESTIMADA    
    z_fixed     = 1.0 # Fixed depth value not null
    p_aux       = np.vstack((p1, np.ones((1, nPoints))))  # Temporally array for normalizing the image points
    p1n         = K1_inv.dot(p_aux)
    vecDesired  = p1n[0:2,:].reshape((2*nPoints,1),order='F')
    depth       = z_fixed*np.ones(nPoints)

    #PASO 4.ESCRIBIR LAS ECUACIONES DE JACOBIANO DE IMAGEN
    Lo          = ImageJacobian(p1n, z_fixed*np.ones(nPoints), 1.0)

    #PASO 5.DEFINIR EL CALCULO DE LA PSEUDOINVERSA
    Lo_inv = np.ones((6,8))  # Pseudo inverse

    x_pos = init_x_pos
    y_pos = init_y_pos
    z_pos = init_z_pos

    pitch   = init_pitch
    yaw     = init_yaw
    roll    = init_roll

    p20 = []

    for j in xrange(0, steps):

        # ===================== Calculate new translation and rotation values using Euler's method====================
        #PASO 6.DEFINIR EL MODELO DE MOVIMIENTO DE LA CAMARA
        x_pos += 0
        y_pos += 0
        z_pos += 0

        pitch   += 0
        yaw     += 0
        roll    += 0        

        camera2.setPosition(pitch, yaw, roll, x_pos, y_pos, z_pos)

        p2, wf_2_if_2, Ext_2 = camera2.projection(w_points)


        p_aux       = np.vstack((p2, np.ones((1, nPoints))))  # Temporally array for normalizing the image points
        p2n         = K2_inv.dot(p_aux)
        vecCurrent  = p2n[0:2,:].reshape((2*nPoints,1), order='F')

        # ==================================== CONTROL COMPUTATION =======================================
        #PASO 7.DEFINIR EL ERROR Y EL CONTROLADOR
        e = np.zeros((6, 1))
        U = np.zeros((6, 1))
        
        U[1, 0] = -U[1, 0] # Inverse control due the the camera framework

        #Avoiding numerical error
        U[np.abs(U) < 1.0e-9] = 0.0

        # Copy data for plot
        UArray[:, j]    = U[:, 0]
        tArray[j]       = t

        pixelCoordsArray[:,j] = p2.reshape((2*nPoints,1), order='F')[:,0]

        positionArray[0, j] = x_pos
        positionArray[1, j] = y_pos
        positionArray[2, j] = z_pos

        #  ========= Average feature error =====================
        #PASO 8.DEFINIR EL CALCULO DEL ERROR PROMEDIO
        pixel_error = np.zeros((2,4))
        averageErrorArray[j]=0

        t += dt

    # ======================================  Draw cameras ========================================

    fig = plt.figure(figsize=(15,10))
    fig.suptitle('World setting')

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax = fig.gca(projection='3d')
    ax.plot(xx, yy, zz, 'o')

    ax.plot(positionArray[0,:],positionArray[1,:],positionArray[2,:]) # Plot camera trajectory


    axis_scale      = 0.5
    camera_scale    = 0.09
    camera1.DrawCamera(ax, scale=camera_scale, color='red')
    camera1.DrawFrame(ax, scale=axis_scale, c='black')

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

    ax.plot(pixelCoordsArray[0,:], pixelCoordsArray[1,:])
    ax.plot(pixelCoordsArray[2,:], pixelCoordsArray[3,:])
    ax.plot(pixelCoordsArray[4,:], pixelCoordsArray[5,:])
    ax.plot(pixelCoordsArray[6,:], pixelCoordsArray[7,:])

    # ======================================  Plot the controls ========================================
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(tArray, UArray[0, :], label='$V_x$')
    ax.plot(tArray, UArray[1, :], label='$V_y$')
    ax.plot(tArray, UArray[2, :], label='$V_z$')
    ax.plot(tArray, UArray[3, :], label='$\omega_x$')
    ax.plot(tArray, UArray[4, :], label='$\omega_y$')
    ax.plot(tArray, UArray[5, :], label='$\omega_z$')
    ax.grid(True)
    ax.legend(loc=0)

    # ======================================  Plot the pixels position ===================================
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(tArray, averageErrorArray, label='Average error')
    ax.grid(True)
    ax.legend(loc=0)
    plt.show()
