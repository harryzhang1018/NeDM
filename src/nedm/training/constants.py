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

TIRE_FORCE_OMEGA_STATE_FIELDS = [
    "tire_fl_force_wheel_fx_n",
    "tire_fl_force_wheel_fy_n",
    "tire_fl_force_wheel_fz_n",
    "tire_fr_force_wheel_fx_n",
    "tire_fr_force_wheel_fy_n",
    "tire_fr_force_wheel_fz_n",
    "tire_rl_force_wheel_fx_n",
    "tire_rl_force_wheel_fy_n",
    "tire_rl_force_wheel_fz_n",
    "tire_rr_force_wheel_fx_n",
    "tire_rr_force_wheel_fy_n",
    "tire_rr_force_wheel_fz_n",
    "tire_fl_spindle_omega_radps",
    "tire_fr_spindle_omega_radps",
    "tire_rl_spindle_omega_radps",
    "tire_rr_spindle_omega_radps",
]

TIRE_NORMAL_FORCE_OMEGA_STATE_FIELDS = [
    "tire_fl_force_wheel_fz_n",
    "tire_fr_force_wheel_fz_n",
    "tire_rl_force_wheel_fz_n",
    "tire_rr_force_wheel_fz_n",
    "tire_fl_spindle_omega_radps",
    "tire_fr_spindle_omega_radps",
    "tire_rl_spindle_omega_radps",
    "tire_rr_spindle_omega_radps",
]

STATE_FIELD_PRESETS = {
    "default": DEFAULT_STATE_FIELDS,
    "tire_force_omega": DEFAULT_STATE_FIELDS + TIRE_FORCE_OMEGA_STATE_FIELDS,
    "tire_normal_force_omega": DEFAULT_STATE_FIELDS + TIRE_NORMAL_FORCE_OMEGA_STATE_FIELDS,
}

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
