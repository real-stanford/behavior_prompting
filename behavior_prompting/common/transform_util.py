"""Transform utilities for converting between different pose representations."""

import numpy as np
from transforms3d import affines, quaternions, axangles
from scipy.spatial.transform import Rotation

def pose_7d_to_4x4(mat):
    # Bx[pos, quat] -> Bx4x4
    return np.array([
        affines.compose(
            T=pose_7d[:3],
            R=quaternions.quat2mat(
                pose_7d[3:]
            ),
            Z=np.ones(3),
        )
        for pose_7d in mat
    ])

def pose_6d_to_4x4(mat):
    # Bx[pos, axis angle] -> Bx4x4
    return np.array([
        affines.compose(
            T=pose6d[:3],
            R=axangles.axangle2mat(
                axis=pose6d[3:]/np.linalg.norm(pose6d[3:]),
                angle=np.linalg.norm(pose6d[3:]),
            ),
            Z=np.ones(3)
        )
        for pose6d in mat
    ])

def pose_4x4_to_6d(mat):
    # Bx4x4 -> Bx[pos, axis angle]
    return np.concatenate((mat[:,:3,3], Rotation.from_matrix(mat[:, :3, :3]).as_rotvec()), axis=1)

def pose_4x4_to_quat_xyzw(mat):
    # Bx4x4 -> Bx[pos, quat]
    return np.concatenate((mat[:,:3,3], Rotation.from_matrix(mat[:, :3, :3]).as_quat()), axis=1)

def pos_quat_xyzw_to_4x4(mat):
    # Bx[pos, quat_xyzw] -> Bx4x4
    return np.array([
        affines.compose(
            T=pose_7d[:3],
            R=Rotation.from_quat(
                pose_7d[3:]
            ).as_matrix(),
            Z=np.ones(3),
        )
        for pose_7d in mat
    ])
