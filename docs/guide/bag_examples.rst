.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

KFC Bag Examples
================

This page documents the local KFC bag examples in ``newton/examples/bag``.
They are heavier than the standard examples and are intended for debugging
deformable bag workflows across multiple solver backends.

.. note::

   This is fork-local documentation that complements the parent Newton
   project docs. Some workflows require local solver checkouts or executables
   that are not part of the upstream Newton package.

The examples share the same visual target: a full-resolution KFC paper bag
mesh, three rigid objects inside the bag, and optional replay capture. Most
solver backends use a lower-resolution proxy mesh for simulation and a
barycentric map to render the full-resolution bag.

Example Entry Points
--------------------

Run these examples through the example launcher from the repository root:

.. code-block:: console

   uv run -m newton.examples kfc_bag_drop_vbd
   uv run -m newton.examples kfc_bag_lift_vbd
   uv run -m newton.examples kfc_bag_lift_ppfcs
   uv run -m newton.examples kfc_bag_lift_ansys

The examples are:

``kfc_bag_drop_vbd``
   Baseline Newton VBD cloth example. A filled bag drops from a short height
   and settles on the ground. Use this first when checking the shared mesh,
   content placement, and replay capture paths.

``kfc_bag_lift_vbd``
   Newton VBD lift example. A Franka FR3 hand closes on the top of the bag and
   lifts it using contact-only finger pads.

``kfc_bag_lift_ppfcs``
   PPF-CTS lift example. The bag deformation and contact solve run in
   ``ppf-contact-solver`` while Newton replays the generated frames with the
   FR3 hand and bag contents.

``kfc_bag_lift_ansys``
   LS-DYNA lift example. The script writes a keyword deck, launches LS-DYNA,
   streams ``d3plot`` output, and replays the solve in Newton.

Shared Options
--------------

The VBD and PPF-CTS examples expose shared proxy mesh options:

.. code-block:: console

   --target-faces 1200
   --proxy-mode cgal-isotropic-remesh

``--target-faces`` controls the approximate solver-side triangle count. Higher
values preserve more shape detail but increase solve cost. ``--proxy-mode`` can
select ``cgal-isotropic-remesh``, ``meshlab-isotropic-remesh``,
``surface-decimate``, or ``qem-decimate`` depending on which optional mesh
tools are installed.

The lift examples also expose gripper controls:

.. code-block:: console

   --closed-width-cm 0.6
   --small-pad

``--closed-width-cm`` sets the final gap between the finger pads. Smaller
values pinch the bag more aggressively. ``--small-pad`` reduces the visual and
contact patch area to make gripper contact tests more localized.

Replay Capture
--------------

Use replay capture when you want deterministic frames and a stitched video:

.. code-block:: console

   uv run -m newton.examples kfc_bag_lift_vbd --capture-replay --capture-frames 180

Captured frames are written under ``outputs/replay_capture/run_<timestamp>``.
At the end of the run, the helper attempts to write ``replay.mp4`` or
``replay.gif`` depending on ``--capture-format``:

.. code-block:: console

   --capture-replay
   --capture-frames 300
   --capture-fps 60
   --capture-dir outputs/replay_capture
   --capture-format mp4

``--save-mp4`` is still available on the VBD examples for direct viewer
recording through ffmpeg, but replay capture is usually easier to inspect
because it keeps the individual PNG frames.

PPF-CTS Workflow
----------------

The PPF-CTS example expects a built ``ppf-contact-solver`` checkout. By
default, it looks for a ``ppf-contact-solver`` directory beside the Newton
source tree:

.. code-block:: console

   uv run -m newton.examples kfc_bag_lift_ppfcs --ppfcs-dir ppf-contact-solver

On Windows, build PPF-CTS first:

.. code-block:: console

   ppf-contact-solver\build-win-native\warmup.bat /nopause
   ppf-contact-solver\build-win-native\build.bat /nopause

On Linux, the example expects the release binary from:

.. code-block:: console

   cargo build --release

The default job directory is ``outputs/ppfcs/kfc_bag_lift``. Use
``--job-dir`` when comparing runs or preserving solver output:

.. code-block:: console

   uv run -m newton.examples kfc_bag_lift_ppfcs --job-dir outputs/ppfcs/try_01

LS-DYNA Workflow
----------------

The LS-DYNA example requires an LS-DYNA executable. Pass it explicitly when the
default local path does not match your machine:

.. code-block:: console

   uv run -m newton.examples kfc_bag_lift_ansys --lsdyna-exe "D:\path\to\ls-dyna.exe"

Useful LS-DYNA options:

.. code-block:: console

   --lsdyna-root "D:\Program Files\LS-DYNA Suite R16.1 Student"
   --job-dir outputs/lsdyna/kfc_bag_lift_ansys
   --output-dt 0.0166667
   --ncpu 4
   --memory 200m

The script writes ``input.k`` and solver diagnostics under ``--job-dir``. It
also writes ``lsdyna_debug_summary.json`` so interrupted or failed streaming
runs can be inspected after the viewer exits.

Troubleshooting
---------------

If a proxy mesh backend is missing, rerun with a different ``--proxy-mode`` or
install the backend named in the error message.

If replay capture creates PNG frames but no video, check that ffmpeg or the
image writer dependencies are available. The PNG frames are still the source of
truth for the replay.

If the lift slips or the bag falls during the final phase, try reducing
``--closed-width-cm``, disabling ``--small-pad``, or lowering
``--target-faces`` while debugging contact behavior.
