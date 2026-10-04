Add an experimental ``particle_self_contact_method="alm_dat"`` option to
``SolverVBD``. It keeps a certified anchor pose per step, replaces the penalty
self-contact force by a compliant ALM row per detected pair (cushion within
the self-contact margin, hard wall at the stored division plane), carries the
contact multipliers across steps by matching the detector's sorted rows, and
truncates the accumulated motion against the stored planes at a fixed cadence
(``particle_dat_plane_update_interval``) and once more before the step commits
(``particle_dat_truncation_iterations``). The default ``"penalty_dat"`` path is
unchanged. ``alm_dat`` requires one self-contact detection per step and does
not yet include friction, damping, or rigid-soft DAT.
