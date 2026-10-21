/*
 * autogear_ml -- learned gear selection for the Shifter Pro, without a torque sensor.
 *
 *   inputs (1 Hz)      cadence, slope, acceleration, coasting flag
 *   ag_features        14 gear-independent inputs (derived speed, averages, trends, restart counter)
 *   ag_gbdt            100 boosted trees of depth 4 -> target log-development (13.9 KB of flash)
 *   ag_controller      hysteresis + persistence -> at most one step per decision, none while coasting
 *
 * Typical loop, once per second:
 *     float x[AG_N_INPUTS];
 *     ag_features_push(&feat, cadence, ag_controller_development(&ctrl), slope, acc, coasting, x);
 *     int8_t step = ag_controller_step(&ctrl, ag_gbdt_predict(x), !coasting && cadence > 20.0f);
 *     if (step) servo_move_one_step(step);
 */
#ifndef AUTOGEAR_ML_H
#define AUTOGEAR_ML_H

#include "ag_controller.h"
#include "ag_features.h"
#include "ag_gbdt.h"

#endif
