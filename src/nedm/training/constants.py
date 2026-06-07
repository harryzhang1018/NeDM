from __future__ import annotations

DEFAULT_STATE_FIELDS = [
    "vel_body_x_mps",
    "vel_body_y_mps",
    "roll_rad",
    "pitch_rad",
    "roll_rate_radps",
    "ang_vel_body_y_radps",
    "yaw_rate_radps",
]

DEFAULT_ACTION_FIELDS = [
    "driver_steering",
    "driver_throttle",
    "driver_braking",
]

DEFAULT_ROLLOUT_FIELDS = [
    "pos_x_m",
    "pos_y_m",
    "yaw_rad",
]

