# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class Camera:
    """Camera class that encapsulates all camera settings and logic."""

    def __init__(self, fov=45.0, near=0.01, far=1000.0, width=1280, height=720, pos=None, up_axis="Z"):
        """
        Initialize camera with given parameters.

        Args:
            fov (float): Field of view in degrees
            near (float): Near clipping plane
            far (float): Far clipping plane
            width (int): Screen width
            height (int): Screen height
            pos (tuple): Initial camera position (if None, uses appropriate default for up_axis)
            up_axis (str): Up axis ("X", "Y", or "Z")
        """
        from pyglet.math import Vec3 as PyVec3

        self.fov = fov
        self.near = near
        self.far = far
        self.width = width
        self.height = height

        # Handle up axis properly first
        if isinstance(up_axis, int):
            self.up_axis = up_axis
        else:
            self.up_axis = "XYZ".index(up_axis.upper())

        # Set appropriate defaults based on up_axis
        if pos is None:
            if self.up_axis == 0:  # X up
                pos = (2.0, 0.0, 10.0)  # 2 units up in X, 10 units back in Z
            elif self.up_axis == 2:  # Z up
                pos = (10.0, 0.0, 2.0)  # 2 units up in Z, 10 units back in Y
            else:  # Y up (default)
                pos = (0.0, 2.0, 10.0)  # 2 units up in Y, 10 units back in Z

        # Camera position
        self.pos = PyVec3(*pos)

        # Camera orientation - this is what users can modify
        self.pitch = 0.0
        self.yaw = -180.0

        # Arcball orbit target (initialized to origin)
        self.target = PyVec3(0.0, 0.0, 0.0)

    @property
    def orbit_distance(self) -> float:
        """Distance from camera to orbit target."""
        d = self.pos - self.target
        return float(np.sqrt(d.x * d.x + d.y * d.y + d.z * d.z))

    def _world_up_vec(self) -> np.ndarray:
        """Return the world-up unit vector as a numpy array."""
        up = np.zeros(3, dtype=np.float64)
        up[self.up_axis] = 1.0
        return up

    def rotate_around_target(self, dyaw: float, dpitch: float):
        """Orbit the camera around ``target`` by the given yaw/pitch deltas [deg].

        Uses the actual camera-to-target arm vector (Rodrigues' rotation) so
        the orbit is correct even when yaw/pitch are stale after a pin.
        Updates ``pos``, ``yaw``, and ``pitch``.
        """
        from pyglet.math import Vec3 as PyVec3

        arm = np.array(
            [self.pos.x - self.target.x, self.pos.y - self.target.y, self.pos.z - self.target.z],
            dtype=np.float64,
        )
        dist = np.linalg.norm(arm)
        if dist < 1e-8:
            return

        world_up = self._world_up_vec()

        # Yaw: rotate arm around world up axis
        yaw_rad = np.radians(-dyaw)
        arm = self._rodrigues(arm, world_up, yaw_rad)

        # Pitch: rotate arm around the camera's right vector
        arm_n = arm / np.linalg.norm(arm)
        right = np.cross(arm_n, world_up)
        rn = np.linalg.norm(right)
        if rn > 1e-8:
            right /= rn
            pitch_rad = np.radians(dpitch)
            new_arm = self._rodrigues(arm, right, pitch_rad)
            # Clamp: reject if too close to straight up/down (gimbal lock)
            new_arm_n = new_arm / np.linalg.norm(new_arm)
            if abs(np.dot(new_arm_n, world_up)) < 0.99:
                arm = new_arm

        # Maintain original distance
        arm = arm / np.linalg.norm(arm) * dist

        self.pos = PyVec3(
            self.target.x + arm[0],
            self.target.y + arm[1],
            self.target.z + arm[2],
        )
        self.sync_yaw_pitch_from_target()

    def dolly(self, delta: float):
        """Move the camera toward (positive) or away from (negative) the target.

        Uses the actual arm vector so it works correctly after a pin.
        Clamps minimum distance to 0.01.
        """
        from pyglet.math import Vec3 as PyVec3

        arm = np.array(
            [self.pos.x - self.target.x, self.pos.y - self.target.y, self.pos.z - self.target.z],
            dtype=np.float64,
        )
        dist = np.linalg.norm(arm)
        if dist < 1e-8:
            return
        new_dist = max(dist - delta, 0.01)
        arm = arm / dist * new_dist
        self.pos = PyVec3(
            self.target.x + arm[0],
            self.target.y + arm[1],
            self.target.z + arm[2],
        )
        self.sync_yaw_pitch_from_target()

    @staticmethod
    def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotate vector *v* around unit *axis* by *angle* radians (Rodrigues)."""
        c, s = np.cos(angle), np.sin(angle)
        return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)

    def pan_around_target(self, dx: float, dy: float):
        """Translate both camera and target in screen-aligned directions.

        Args:
            dx: Horizontal offset in world units (positive = right).
            dy: Vertical offset in world units (positive = up).
        """
        right = self.get_right()
        up = self.get_up()
        offset = right * dx + up * dy
        self.pos = self.pos + offset
        self.target = self.target + offset

    def sync_yaw_pitch_from_target(self):
        """Recompute yaw/pitch so that the camera looks at ``target``.

        Call this after setting ``target`` externally (e.g. when pinning the
        orbit center on geometry).
        """
        d = self.target - self.pos
        dx, dy, dz = float(d.x), float(d.y), float(d.z)

        if self.up_axis == 0:  # X up
            dist_hz = np.sqrt(dy * dy + dz * dz)
            self.yaw = float(np.degrees(np.arctan2(dz, dy)))
            self.pitch = float(np.degrees(np.arctan2(dx, dist_hz))) if dist_hz > 1e-8 else (90.0 if dx > 0 else -90.0)
        elif self.up_axis == 2:  # Z up
            dist_hz = np.sqrt(dx * dx + dy * dy)
            self.yaw = float(np.degrees(np.arctan2(dy, dx)))
            self.pitch = float(np.degrees(np.arctan2(dz, dist_hz))) if dist_hz > 1e-8 else (90.0 if dz > 0 else -90.0)
        else:  # Y up
            dist_hz = np.sqrt(dx * dx + dz * dz)
            self.yaw = float(np.degrees(np.arctan2(dz, dx)))
            self.pitch = float(np.degrees(np.arctan2(dy, dist_hz))) if dist_hz > 1e-8 else (90.0 if dy > 0 else -90.0)

        self.pitch = max(min(self.pitch, 89.0), -89.0)
        self.yaw = (self.yaw + 180.0) % 360.0 - 180.0

    def get_front(self):
        """Get the camera front direction vector (read-only)."""
        from pyglet.math import Vec3 as PyVec3

        # Clamp pitch to avoid gimbal lock
        pitch = max(min(self.pitch, 89.0), -89.0)

        # Calculate front vector directly in the coordinate system based on up_axis
        # This ensures yaw/pitch work correctly for each coordinate system

        if self.up_axis == 0:  # X up
            # Yaw rotates around X (vertical), pitch is elevation
            front_x = np.sin(np.deg2rad(pitch))
            front_y = np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_z = np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            return PyVec3(front_x, front_y, front_z).normalize()

        elif self.up_axis == 2:  # Z up
            # Yaw rotates around Z (vertical), pitch is elevation
            front_x = np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_y = np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_z = np.sin(np.deg2rad(pitch))
            return PyVec3(front_x, front_y, front_z).normalize()

        else:  # Y up (default)
            # Yaw rotates around Y (vertical), pitch is elevation
            front_x = np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_y = np.sin(np.deg2rad(pitch))
            front_z = np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            return PyVec3(front_x, front_y, front_z).normalize()

    def get_right(self):
        """Get the camera right direction vector (read-only)."""
        from pyglet.math import Vec3 as PyVec3

        return PyVec3.cross(self.get_front(), self.get_up()).normalize()

    def get_up(self):
        """Get the camera up direction vector (read-only)."""
        from pyglet.math import Vec3 as PyVec3

        # World up vector based on up axis
        if self.up_axis == 0:  # X up
            world_up = PyVec3(1.0, 0.0, 0.0)
        elif self.up_axis == 2:  # Z up
            world_up = PyVec3(0.0, 0.0, 1.0)
        else:  # Y up (default)
            world_up = PyVec3(0.0, 1.0, 0.0)

        # Compute right vector and use it to get proper up vector
        front = self.get_front()
        right = PyVec3.cross(front, world_up).normalize()
        return PyVec3.cross(right, front).normalize()

    def get_view_matrix(self, scaling=1.0):
        """
        Compute view matrix handling up axis properly.

        Args:
            scaling (float): Scene scaling factor

        Returns:
            np.ndarray: 4x4 view matrix
        """
        from pyglet.math import Mat4, Vec3

        # Get camera vectors (already transformed for up axis)
        pos = Vec3(*(self.pos / scaling))
        front = Vec3(*self.get_front())
        up = Vec3(*self.get_up())

        return np.array(Mat4.look_at(pos, pos + front, up), dtype=np.float32)

    def get_projection_matrix(self):
        """
        Compute projection matrix.

        Returns:
            np.ndarray: 4x4 projection matrix
        """
        from pyglet.math import Mat4 as PyMat4

        if self.height == 0:
            return np.eye(4, dtype=np.float32)

        aspect_ratio = self.width / self.height
        return np.array(PyMat4.perspective_projection(aspect_ratio, self.near, self.far, self.fov))

    def get_world_ray(self, x: float, y: float):
        """Get the world ray for a given pixel.

        returns:
            p: wp.vec3, ray origin
            d: wp.vec3, ray direction
        """
        from pyglet.math import Vec3 as PyVec3

        aspect_ratio = self.width / self.height

        # pre-compute factor from vertical FOV
        fov_rad = np.radians(self.fov)
        alpha = float(np.tan(fov_rad * 0.5))  # = tan(fov/2)

        # build an orthonormal basis (front, right, up)
        front = self.get_front()
        right = self.get_right()
        up = self.get_up()

        # normalised pixel coordinates
        u = 2.0 * (x / self.width) - 1.0  # [-1, 1] left → right
        v = 2.0 * (y / self.height) - 1.0  # [-1, 1] bottom → top

        # ray direction in world space (before normalisation)
        direction = front + right * u * alpha * aspect_ratio + up * v * alpha
        direction = direction / float(np.linalg.norm(direction))

        return self.pos, PyVec3(*direction)

    def update_screen_size(self, width, height):
        """Update screen dimensions."""
        self.width = width
        self.height = height
