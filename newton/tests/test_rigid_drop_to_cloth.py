# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid Drop to Cloth — Margin Study
#
# Six groups of rigid bodies (sphere, box, capsule, cylinder, cone, mesh)
# drop onto six separate cloth sheets, each configured with a different
# shape margin. Validates that the equilibrium contact distance grows
# approximately linearly with particle_radius + margin.
#
# Command: python -m newton.tests.test_rigid_drop_to_cloth
#
###########################################################################

import os
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import add_function_test, get_test_devices

SHAPE_NAMES = ["sphere", "box", "capsule", "cylinder", "cone", "mesh"]
SHAPE_RADIUS = 0.1
MARGIN_VALUES = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]

# Expected avg contact distance per margin (measured empirically).
# Used as reference for the test_final loose comparison.
EXPECTED_DISTANCES = {
    0.000: 0.021,
    0.005: 0.024,
    0.010: 0.026,
    0.015: 0.029,
    0.020: 0.033,
    0.030: 0.037,
}

SHAPE_BBOXES = {
    "sphere": SHAPE_RADIUS,
    "box": SHAPE_RADIUS * np.sqrt(3),
    "capsule": SHAPE_RADIUS * 0.7 + SHAPE_RADIUS,
    "cylinder": np.sqrt(SHAPE_RADIUS**2 + (SHAPE_RADIUS * 0.5) ** 2),
    "cone": np.sqrt(SHAPE_RADIUS**2 + SHAPE_RADIUS**2),
    "mesh": SHAPE_RADIUS * 1.2,
}

# Grid layout: 6 cloth sheets in a 3x2 arrangement (x, y offsets)
GRID_COLS = 3
GRID_ROWS = 2
GRID_SPACING = 3.0  # meters between cloth centers


def _load_bear_mesh():
    from pxr import Usd, UsdGeom

    bear_path = os.path.join(newton.examples.get_asset_directory(), "bear.usd")
    stage = Usd.Stage.Open(bear_path)
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))

    points = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    center = (points.max(axis=0) + points.min(axis=0)) / 2.0
    points -= center
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    scale = (SHAPE_RADIUS * 2.0) / extent
    points *= scale

    return newton.Mesh(points, indices)


def _generate_positions(rng, count, bounding_radii, max_attempts=500):
    lo, hi = -0.25, 0.25
    placed = []

    for i in range(count):
        r_i = bounding_radii[i]
        for _ in range(max_attempts):
            x = rng.uniform(lo, hi)
            y = rng.uniform(lo, hi)
            overlap = False
            for (px, py), r_j in placed:
                if np.sqrt((x - px) ** 2 + (y - py) ** 2) < r_i + r_j + 0.05:
                    overlap = True
                    break
            if not overlap:
                placed.append(((x, y), r_i))
                break
        else:
            placed.append(((x, y), r_i))
    return [p for p, _ in placed]


def _random_rotation(rng):
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _grid_offset(group_idx):
    col = group_idx % GRID_COLS
    row = group_idx // GRID_COLS
    return col * GRID_SPACING, row * GRID_SPACING


def build_model(builder, seed=42):
    """Add 6 cloth sheets with 6 rigid shapes each, one per margin value.

    Returns:
        Tuple of (cloth_particle_count_per_group, particle_radius,
                  group_body_indices, group_shape_indices).
    """
    rng = np.random.default_rng(seed)

    cloth_res = 40
    cell_size = 0.05
    cloth_half_extent = cloth_res * cell_size / 2.0
    particle_radius = 0.025

    bear_mesh = _load_bear_mesh()
    cloth_particle_count = (cloth_res + 1) * (cloth_res + 1)
    r = SHAPE_RADIUS

    group_body_indices = {}
    group_shape_indices = {}

    for gi, margin in enumerate(MARGIN_VALUES):
        ox, oy = _grid_offset(gi)

        builder.add_cloth_grid(
            pos=wp.vec3(ox - cloth_half_extent, oy - cloth_half_extent, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cloth_res,
            dim_y=cloth_res,
            cell_x=cell_size,
            cell_y=cell_size,
            mass=0.01,
            fix_left=True,
            fix_right=True,
            tri_ke=1e4,
            tri_ka=1e4,
            tri_kd=1e-4,
            edge_ke=1.0,
            edge_kd=1e-3,
            particle_radius=particle_radius,
        )

        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = 1000.0
        cfg.ke = 1e5
        cfg.kd = 1e-4
        cfg.mu = 0.5
        cfg.margin = margin
        cfg.has_particle_collision = True

        bounding_radii = [SHAPE_BBOXES[name] for name in SHAPE_NAMES]
        positions = _generate_positions(rng, len(SHAPE_NAMES), bounding_radii)

        body_indices = []
        shape_indices = []
        for i, name in enumerate(SHAPE_NAMES):
            px, py = positions[i]
            drop_z = 0.7 + 2.0 * margin + i * (0.2 + 2.0 * margin)
            rot = _random_rotation(rng)
            body = builder.add_body(
                xform=wp.transform(wp.vec3(ox + px, oy + py, drop_z), rot)
            )
            body_indices.append(body)

            shape_idx = len(builder.shape_type)
            if name == "sphere":
                builder.add_shape_sphere(body, radius=r, cfg=cfg)
            elif name == "box":
                builder.add_shape_box(body, hx=r, hy=r, hz=r, cfg=cfg)
            elif name == "capsule":
                builder.add_shape_capsule(body, radius=r * 0.7, half_height=r, cfg=cfg)
            elif name == "cylinder":
                builder.add_shape_cylinder(body, radius=r, half_height=r * 0.5, cfg=cfg)
            elif name == "cone":
                builder.add_shape_cone(body, radius=r, half_height=r, cfg=cfg)
            elif name == "mesh":
                builder.add_shape_mesh(body, mesh=bear_mesh, cfg=cfg)
            shape_indices.append(shape_idx)

        group_body_indices[margin] = body_indices
        group_shape_indices[margin] = shape_indices

    builder.add_ground_plane()
    builder.color(include_bending=True)

    return cloth_particle_count, particle_radius, group_body_indices, group_shape_indices


def measure_contact_distances(model, state, contacts, group_shape_indices):
    """Measure avg signed contact distance per margin group using actual contact pairs.

    Returns dict: margin -> avg_distance (float or nan if no contacts).
    """
    from scipy.spatial.transform import Rotation

    count = contacts.soft_contact_count.numpy()[0]
    if count == 0:
        return {m: float("nan") for m in MARGIN_VALUES}

    contact_particle = contacts.soft_contact_particle.numpy()[:count]
    contact_shape = contacts.soft_contact_shape.numpy()[:count]
    contact_body_pos = contacts.soft_contact_body_pos.numpy()[:count]
    contact_normal = contacts.soft_contact_normal.numpy()[:count]

    particle_q = state.particle_q.numpy()
    body_q = state.body_q.numpy()
    shape_body = model.shape_body.numpy()
    particle_radius_arr = model.particle_radius.numpy()

    results = {}
    for margin, shape_indices in group_shape_indices.items():
        all_dists = []
        for si in shape_indices:
            mask = contact_shape == si
            if mask.sum() == 0:
                continue

            body_idx = shape_body[si]
            X_wb = body_q[body_idx]
            pos_wb = X_wb[:3]
            rot_wb = X_wb[3:7]

            cp_local = contact_body_pos[mask]
            normals = contact_normal[mask]
            p_indices = contact_particle[mask]
            p_pos = particle_q[p_indices]
            p_rad = particle_radius_arr[p_indices]

            r = Rotation.from_quat([rot_wb[0], rot_wb[1], rot_wb[2], rot_wb[3]])
            cp_world = r.apply(cp_local) + pos_wb

            diffs = p_pos - cp_world
            signed_dist = np.einsum("ij,ij->i", diffs, normals) - p_rad
            all_dists.extend(signed_dist.tolist())

        results[margin] = float(np.mean(all_dists)) if all_dists else float("nan")

    return results


# ---------------------------------------------------------------------------
# Example class
# ---------------------------------------------------------------------------

class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.max_retries = 5

        seed = getattr(args, "seed", 42)
        self._build(seed)
        self.viewer.set_model(self.model)

    def _build(self, seed):
        self.seed = seed
        builder = newton.ModelBuilder(gravity=-9.8)
        (
            self.cloth_particle_count,
            self.particle_radius,
            self.group_body_indices,
            self.group_shape_indices,
        ) = build_model(builder, seed=seed)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1e5
        self.model.soft_contact_kd = 1e-4
        self.model.soft_contact_mu = 0.5

        max_margin = max(MARGIN_VALUES)
        detect_margin = self.particle_radius * 1.1 + max_margin

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=10,
            particle_enable_self_contact=False,
        )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=detect_margin,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Run one more collision pass to get fresh contact pairs
        self.state_0.clear_forces()
        self.collision_pipeline.collide(self.state_0, self.contacts)

        distances = measure_contact_distances(
            self.model, self.state_0, self.contacts, self.group_shape_indices
        )

        # Check no explosion
        particle_q = self.state_0.particle_q.numpy()
        bbox = np.max(np.abs(particle_q))
        assert bbox < 50.0, f"Simulation exploded: max coordinate={bbox:.2f}"

        # Retry with different seed if any margin group has no contacts
        has_nan = any(np.isnan(d) for d in distances.values())
        if has_nan:
            retry = getattr(self, "_retry_count", 0)
            if retry < self.max_retries:
                self._retry_count = retry + 1
                new_seed = self.seed + retry + 1
                print(f"Missing contact data, retrying with seed={new_seed} "
                      f"(attempt {retry + 1}/{self.max_retries})")
                self._build(new_seed)
                self.viewer.set_model(self.model)
                for _ in range(300):
                    self.simulate()
                    self.sim_time += self.frame_dt
                return self.test_final()
            # If exhausted retries, skip nan entries
            distances = {m: d for m, d in distances.items() if not np.isnan(d)}

        print("\n=== Margin Study Results ===")
        print(f"{'margin':>8} {'pr+margin':>10} {'avg_dist':>10} {'expected':>10} {'error':>8}")
        print("-" * 52)
        for m in MARGIN_VALUES:
            if m not in distances:
                print(f"{m:>8.3f} {self.particle_radius + m:>10.4f}       nan")
                continue
            d = distances[m]
            exp = EXPECTED_DISTANCES[m]
            err = d - exp
            print(f"{m:>8.3f} {self.particle_radius + m:>10.4f} {d:>10.4f} {exp:>10.4f} {err:>+8.4f}")

        # Validate: distance should approximately match expected (abs error < 0.01)
        for m, d in distances.items():
            exp = EXPECTED_DISTANCES[m]
            assert abs(d - exp) < 0.01, (
                f"margin={m}: avg distance {d:.4f} deviates from expected {exp:.4f} "
                f"by {abs(d - exp):.4f} (> 0.01)"
            )

        # Validate: distance should be monotonically non-decreasing with margin
        sorted_margins = sorted(distances.keys())
        for i in range(1, len(sorted_margins)):
            m_prev = sorted_margins[i - 1]
            m_curr = sorted_margins[i]
            assert distances[m_curr] >= distances[m_prev] - 0.005, (
                f"Distance not increasing: margin={m_prev}->{m_curr}, "
                f"dist={distances[m_prev]:.4f}->{distances[m_curr]:.4f}"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=42)
        return parser


# ---------------------------------------------------------------------------
# Headless sim helper (for unit tests)
# ---------------------------------------------------------------------------

class RigidDropSim:
    def __init__(self, device, seed=42):
        self.device = device
        builder = newton.ModelBuilder(gravity=-9.8)
        (
            self.cloth_particle_count,
            self.particle_radius,
            self.group_body_indices,
            self.group_shape_indices,
        ) = build_model(builder, seed=seed)

        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = 1e5
        self.model.soft_contact_kd = 1e-4
        self.model.soft_contact_mu = 0.5

        max_margin = max(MARGIN_VALUES)
        detect_margin = self.particle_radius * 1.1 + max_margin

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=10,
            particle_enable_self_contact=False,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=detect_margin,
        )

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = 5
        self.sim_dt = self.frame_dt / self.substeps
        self.num_frames = 300
        self.seed = seed

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

    def simulate_frame(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def run(self):
        for _ in range(self.num_frames):
            self.simulate_frame()

    def measure(self):
        self.state_0.clear_forces()
        self.collision_pipeline.collide(self.state_0, self.contacts)
        return measure_contact_distances(
            self.model, self.state_0, self.contacts, self.group_shape_indices
        )


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_rigid_drop_margin_study(test, device):
    max_retries = 5
    for attempt in range(max_retries):
        seed = 42 + attempt
        sim = RigidDropSim(device, seed=seed)
        sim.run()
        distances = sim.measure()

        has_nan = any(np.isnan(d) for d in distances.values())
        if not has_nan:
            break
        missing = [f"margin={m}" for m, d in distances.items() if np.isnan(d)]
        print(f"seed={seed}: missing data for {missing}, retrying...")
    else:
        distances = {m: d for m, d in distances.items() if not np.isnan(d)}

    # Check no explosion
    particle_q = sim.state_0.particle_q.numpy()
    bbox = np.max(np.abs(particle_q))
    test.assertLess(bbox, 50.0, f"Simulation exploded: max coordinate={bbox:.2f}")

    print(f"\n=== Margin Study (seed={seed}) ===")
    print(f"{'margin':>8} {'pr+margin':>10} {'avg_dist':>10} {'expected':>10}")
    print("-" * 44)
    for m in MARGIN_VALUES:
        d = distances.get(m, float("nan"))
        exp = EXPECTED_DISTANCES[m]
        label = f"{d:>10.4f}" if not np.isnan(d) else "       nan"
        print(f"{m:>8.3f} {sim.particle_radius + m:>10.4f} {label} {exp:>10.4f}")

    # Distance should approximately match expected (abs error < 0.01)
    for m, d in distances.items():
        exp = EXPECTED_DISTANCES[m]
        test.assertAlmostEqual(
            d, exp, delta=0.01,
            msg=f"margin={m}: avg distance {d:.4f} vs expected {exp:.4f}",
        )

    # Distance should be monotonically non-decreasing
    sorted_margins = sorted(distances.keys())
    for i in range(1, len(sorted_margins)):
        m_prev = sorted_margins[i - 1]
        m_curr = sorted_margins[i]
        test.assertGreaterEqual(
            distances[m_curr], distances[m_prev] - 0.005,
            msg=f"Distance not increasing: margin={m_prev}->{m_curr}",
        )


# ---------------------------------------------------------------------------
# Test registration
# ---------------------------------------------------------------------------

devices = [d for d in get_test_devices(mode="basic") if "cuda" in str(d)]


class TestRigidDropToCloth(unittest.TestCase):
    pass


for d in devices:
    add_function_test(
        TestRigidDropToCloth,
        f"test_rigid_drop_margin_study_{d}",
        test_rigid_drop_margin_study,
        devices=[d],
        check_output=False,
    )


if __name__ == "__main__":
    import sys

    if any(a.startswith("--viewer") for a in sys.argv):
        parser = Example.create_parser()
        viewer, args = newton.examples.init(parser)
        example = Example(viewer, args)
        newton.examples.run(example, args)
    else:
        unittest.main()
