import numpy as np

def ImageJacobian(image_points, z_world_points, focal_length):

    """

    :param image_points:
    :param z_world_points:
    :param camera:
    :return:
    """
    f = focal_length

    points_counter = image_points.shape[1]

    L = np.zeros((2 * points_counter, 6))

    for i in xrange(0, points_counter):
        u = image_points[0, i]
        v = image_points[1, i]

        z = z_world_points[i]

        #PASO 4: INCLUIR LAS ECUACIONES DEL JACOBIANO DE IMAGEN
        L[2 * i, :]     = np.zeros((1,6))
        L[2 * i + 1, :] = np.zeros((1,6))

    return L