#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

import numpy as np
import json
from pyquaternion import Quaternion


def get_K_multiplier():
    K_multiplier = np.zeros((3, 3))
    K_multiplier[0][1] = K_multiplier[1][2] = -1
    K_multiplier[2][0] = 1
    return K_multiplier


def get_projection_matrix(camera):
    K_matrix = np.zeros((3, 3))
    K_matrix[0][0] = camera.fx
    K_matrix[0][2] = camera.u0
    K_matrix[1][1] = camera.fy
    K_matrix[1][2] = camera.v0
    K_matrix[2][2] = 1
    return K_matrix


def apply_transformation_points(points, transformation_matrix):
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.matmul(transformation_matrix, points.T).T
    return points


class Camera(object):
    def __init__(self, fx, fy, u0, v0, sensor_T_ISO_8855):
        self.fx = fx
        self.fy = fy
        self.u0 = u0
        self.v0 = v0
        self.sensor_T_ISO_8855 = sensor_T_ISO_8855


class Box3DImageTransform(object):
    def __init__(self, size, quaternion, center, camera):
        self._camera = camera
        self._rotation_matrix = np.array(Quaternion(quaternion).rotation_matrix)
        self._size = np.array(size)
        self._center = np.array(center)

        self._box_points_2d = np.zeros((8, 2))
        self._box_points_3d_vehicle = np.zeros((8, 3))
        self._box_points_3d_cam = np.zeros((8, 3))

        self.bottom_arrow_2d = np.zeros((2, 2))
        self._bottom_arrow_3d_vehicle = np.zeros((2, 3))
        self._bottom_arrow_3d_cam = np.zeros((2, 3))

        self._box_left_side_cropped_2d = []
        self._box_right_side_cropped_2d = []
        self._box_front_side_cropped_2d = []
        self._box_back_side_cropped_2d = []
        self._box_top_side_cropped_2d = []
        self._box_bottom_side_cropped_2d = []

        self.update()

    def _get_side_visibility(self, face_center, face_normal):
        return np.dot(face_normal, face_center) < 0

    def get_all_side_visibilities(self):
        K_multiplier = get_K_multiplier()
        rotation_matrix_cam = np.matmul(
            np.matmul(K_multiplier, self._rotation_matrix), K_multiplier.T
        )

        box_vector_x = rotation_matrix_cam[:, 0]
        box_vector_y = rotation_matrix_cam[:, 1]
        box_vector_z = rotation_matrix_cam[:, 2]

        front_visible = self._get_side_visibility(
            (self._box_points_3d_cam[3] + self._box_points_3d_cam[6]) / 2, box_vector_z
        )
        back_visible = self._get_side_visibility(
            (self._box_points_3d_cam[0] + self._box_points_3d_cam[5]) / 2, -box_vector_z
        )
        top_visible = self._get_side_visibility(
            (self._box_points_3d_cam[7] + self._box_points_3d_cam[5]) / 2, -box_vector_y
        )
        bottom_visible = self._get_side_visibility(
            (self._box_points_3d_cam[0] + self._box_points_3d_cam[2]) / 2, box_vector_y
        )
        left_visible = self._get_side_visibility(
            (self._box_points_3d_cam[0] + self._box_points_3d_cam[7]) / 2, -box_vector_x
        )
        right_visible = self._get_side_visibility(
            (self._box_points_3d_cam[1] + self._box_points_3d_cam[6]) / 2, box_vector_x
        )

        return [
            front_visible,
            back_visible,
            top_visible,
            bottom_visible,
            left_visible,
            right_visible,
        ]

    def get_all_side_polygons_2d(self):
        front_side = self._box_front_side_cropped_2d
        back_side = self._box_back_side_cropped_2d
        top_side = self._box_top_side_cropped_2d
        bottom_side = self._box_bottom_side_cropped_2d
        left_side = self._box_left_side_cropped_2d
        right_side = self._box_right_side_cropped_2d

        return [front_side, back_side, top_side, bottom_side, left_side, right_side]

    def get_full_box_2d(self):
        corner_points_2d = np.array(self.get_all_side_polygons_2d()).reshape(-1, 2)
        return [
            np.amin(corner_points_2d[:, 0]),
            np.amin(corner_points_2d[:, 1]),
            np.amax(corner_points_2d[:, 0]),
            np.amax(corner_points_2d[:, 1]),
        ]

    def _crop_side_polygon_and_project(self, side_point_indices=[], side_points=[]):
        K_matrix = get_projection_matrix(self._camera)
        camera_plane_z = 0.01

        side_points_3d_cam = [self._box_points_3d_cam[i] for i in side_point_indices]
        side_points_3d_cam += side_points

        cropped_polygon_3d = []
        for i, point in enumerate(side_points_3d_cam):
            if point[2] > camera_plane_z:  # 1 cm
                cropped_polygon_3d.append(point)
            else:
                next_index = (i + 1) % len(side_points_3d_cam)
                prev_index = i - 1

                if side_points_3d_cam[prev_index][2] > camera_plane_z:
                    delta_0 = point - side_points_3d_cam[prev_index]
                    k_0 = (camera_plane_z - point[2]) / delta_0[2]
                    point_0 = point + k_0 * delta_0
                    cropped_polygon_3d.append(point_0)

                if side_points_3d_cam[next_index][2] > camera_plane_z:
                    delta_1 = point - side_points_3d_cam[next_index]
                    k_1 = (camera_plane_z - point[2]) / delta_1[2]
                    point_1 = point + k_1 * delta_1
                    cropped_polygon_3d.append(point_1)

        if len(cropped_polygon_3d) == 0:
            cropped_polygon_2d = []
        else:
            cropped_polygon_2d = np.matmul(K_matrix, np.array(cropped_polygon_3d).T)
            cropped_polygon_2d = cropped_polygon_2d[:2, :] / cropped_polygon_2d[-1, :]
            cropped_polygon_2d = cropped_polygon_2d.T.tolist()
            cropped_polygon_2d.append(cropped_polygon_2d[0])

        return cropped_polygon_2d

    def update(self):
        self._update_box_points_3d()
        self._update_box_sides_cropped()
        self._update_box_points_2d()

    def _update_box_sides_cropped(self):
        self._box_left_side_cropped_2d = self._crop_side_polygon_and_project(
            [3, 0, 4, 7]
        )
        self._box_right_side_cropped_2d = self._crop_side_polygon_and_project(
            [1, 5, 6, 2]
        )
        self._box_front_side_cropped_2d = self._crop_side_polygon_and_project(
            [3, 2, 6, 7]
        )
        self._box_back_side_cropped_2d = self._crop_side_polygon_and_project(
            [0, 1, 5, 4]
        )
        self._box_top_side_cropped_2d = self._crop_side_polygon_and_project(
            [4, 5, 6, 7]
        )
        self._box_bottom_side_cropped_2d = self._crop_side_polygon_and_project(
            [0, 1, 2, 3]
        )
        self.bottom_arrow_2d = self._crop_side_polygon_and_project(
            side_points=[self._bottom_arrow_3d_cam[x] for x in range(2)]
        )


    def _update_box_points_3d(self):
        center_vectors = np.zeros((8, 3))
        # Bottom Face
        center_vectors[0] = np.array(
            [-self._size[0] / 2, self._size[1] / 2, -self._size[2] / 2]
            # Back Left Bottom
        )
        center_vectors[1] = np.array(
            [-self._size[0] / 2, -self._size[1] / 2, -self._size[2] / 2]
            # Back Right Bottom
        )
        center_vectors[2] = np.array(
            [self._size[0] / 2, -self._size[1] / 2, -self._size[2] / 2]
            # Front Right Bottom
        )
        center_vectors[3] = np.array(
            [self._size[0] / 2, self._size[1] / 2, -self._size[2] / 2]
            # Front Left Bottom
        )

        # Top Face
        center_vectors[4] = np.array(
            [-self._size[0] / 2, self._size[1] / 2, self._size[2] / 2]
            # Back Left Top
        )
        center_vectors[5] = np.array(
            [-self._size[0] / 2, -self._size[1] / 2, self._size[2] / 2]
            # Back Right Top
        )
        center_vectors[6] = np.array(
            [self._size[0] / 2, -self._size[1] / 2, self._size[2] / 2]
            # Front Right Top
        )
        center_vectors[7] = np.array(
            [self._size[0] / 2, self._size[1] / 2, self._size[2] / 2]
            # Front Left Top
        )

        # Rotate the vectors
        box_points_3d = np.matmul(self._rotation_matrix, center_vectors.T).T
        # Translate to box position in 3d space
        box_points_3d += self._center

        self._box_points_3d_vehicle = box_points_3d

        self._bottom_arrow_3d_vehicle = np.array(
            [
                (0.5 * (self._box_points_3d_vehicle[3] + self._box_points_3d_vehicle[2])),
                (0.5 * (self._box_points_3d_vehicle[3] + self._box_points_3d_vehicle[1])),
            ]
        )
        bottom_arrow_3d_cam = apply_transformation_points(
            self._bottom_arrow_3d_vehicle, self._camera.sensor_T_ISO_8855
        )

        # Points in ISO8855 system with origin at the sensor
        box_points_3d_cam = apply_transformation_points(
            self._box_points_3d_vehicle, self._camera.sensor_T_ISO_8855
        )
        K_multiplier = get_K_multiplier()
        self._box_points_3d_cam = np.matmul(K_multiplier, box_points_3d_cam.T).T
        self._bottom_arrow_3d_cam = np.matmul(K_multiplier, bottom_arrow_3d_cam.T).T

    def _update_box_points_2d(self):
        K_matrix = get_projection_matrix(self._camera)
        box_points_2d = np.matmul(K_matrix, self._box_points_3d_cam.T)
        box_points_2d = box_points_2d[:2, :] / box_points_2d[-1, :]
        self._box_points_2d = box_points_2d.T
