# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:47:41 2016

@author: robotics
"""
import numpy as np
import math

def unitize(x,y):
    l = x/(math.sqrt(x**2+y**2)) 
    m = y/(math.sqrt(x**2+y**2))
    return l,m
    

def homogToRt(H):
    U, S, V = np.linalg.svd(H, full_matrices=True)
    s1      = S[0]/S[1]
    s3      = S[2]/S[1]
    zeta    = s1-s3
    a1      = math.sqrt(1-s3**2)
    b1      = math.sqrt(s1**2-1)
    a,b     = unitize(a1,b1)
    c,d     = unitize( 1+s1*s3, a1*b1 )
    e,f     = unitize( -b/s1, -a/s3 )
    v1      = V[0,:] # V es la transpuesta de la que regresa matlab
    #v1      = V[:,0]
    v3      = V[2,:]
    #v3      = V[:,2]
    n1      = b*v1-a*v3
    n2      = b*v1+a*v3
    R1      = U.dot(np.array([[c,0,d], [0,1,0], [-d,0,c]]).dot(V) )
    R2      = U.dot(np.array([[c,0,-d], [0,1,0], [d,0,c]]).dot(V) )
    t1      = e*v1+f*v3
    t2      = e*v1-f*v3
    if (n1[2]<0): 
        t1 = -t1
        n1 = -n1
    if (n2[2]<0): 
        t2 = -t2
        n2 = -n2
    if (n1[2]>n2[2]):
        R = R1.T
        t = zeta*t1
        n = n1
        return R,t,n
    else:
        R = R2.T
        t = zeta*t2
        n = n2
        return R,t,n
        
def H_from_points(fp,tp):
    """ Find homography H, such that fp is mapped to tp
        using the linear DLT method. Points are conditioned
        automatically. """
    
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
        
    #     condition points (important for numerical reasons)
        #     --from points--
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = np.vstack([fp[0,:],fp[1,:],[1,1,1,1]])
    fp = np.dot(C1,fp)
    
    # --to points--
    m = np.mean(tp[:2], axis=1)
    maxstd = max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = np.vstack([tp[0,:],tp[1,:],[1,1,1,1]])
    tp = np.dot(C2,tp)
    
    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2*nbr_correspondences,9))
    for i in range(nbr_correspondences):        
#        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
#                    tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
#        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
#                    tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]
    #PASO 1.INCLUIR ECUACIONES PARA LA ESTIMACION DE HOMOGRAFIA        
        A[2*i] = 1
        A[2*i+1] = 1
    
    U,S,V = np.linalg.svd(A)
    H = V[8].reshape((3,3))
    
    # decondition
    H = np.dot(np.linalg.inv(C2),np.dot(H,C1))
    
    # normalize and return
    return H / H[2,2]
    
def Rodrigues(R):
    
    if( (0.5*(np.trace(R)-1))  > 1.):
        theta = np.arccos(1.0)
    else:
        theta = np.arccos(0.5*(np.trace(R)-1))
    if(theta==0.):
        return np.array([0.,0.,0.]).T
    else:
        coef = (0.5*theta)/(np.sin(theta) )
        u = coef*np.array([ [R[2,1]-R[1,2] ],[ -R[2,0]+R[0,2] ], [ R[1,0]-R[0,1] ]  ]).T
        return u
    
def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[2]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point