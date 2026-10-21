/*
 * ag_features -- the 14 inputs of the gear model, computed on the device once per second.
 *
 * Mirrors src/autogear/features.py. While the rider pedals, the device derives its speed from the
 * crank cadence and the development of the gear it has engaged itself:
 *     v_hat = cadence * development / 60
 * so no wheel-speed sensor is needed and v_hat does not depend on which gear is engaged.
 * While coasting, v_hat integrates the longitudinal acceleration (IMU, gravity removed).
 *
 * Memory: one ring buffer of 16 samples per signal (~200 bytes in total with float).
 */
#ifndef AG_FEATURES_H
#define AG_FEATURES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AG_N_INPUTS 14
#define AG_HISTORY 16                   /* >= longest window (15 s) + 1 */
#define AG_PEDAL_MIN_CADENCE_RPM 20.0f  /* below this, the chain is considered not turning */

#ifdef AG_REAL_DOUBLE
typedef double ag_real_t;               /* host builds: bit-closer to the Python reference */
#else
typedef float ag_real_t;                /* default on the Cortex-M4F: single-precision FPU */
#endif

typedef struct {
    ag_real_t v_hat[AG_HISTORY];
    ag_real_t slope[AG_HISTORY];
    ag_real_t acc[AG_HISTORY];
    uint8_t coasting[AG_HISTORY];
    ag_real_t v_first;                  /* first v_hat of the segment (lag padding) */
    ag_real_t v_prev;
    ag_real_t restart_s;                /* seconds since pedalling resumed, capped at 10 */
    uint32_t n;                         /* samples since the segment started */
} ag_features_t;

/* Start a new segment (after a stop or at power-up). */
void ag_features_reset(ag_features_t *f);

/*
 * Push one second of measurements and compute the 14 inputs.
 *   cadence_rpm    crank cadence from the Hall sensor
 *   development_m  development of the engaged gear, metres per crank revolution
 *   slope          road slope as a fraction (barometer or IMU pitch)
 *   acc_ms2        longitudinal acceleration, gravity removed (IMU)
 *   coasting       1 when cadence < 15 rpm
 * Returns v_hat (m/s). `out` receives the inputs in the order of AG feature names.
 */
float ag_features_push(ag_features_t *f, float cadence_rpm, float development_m, float slope,
                       float acc_ms2, uint8_t coasting, float out[AG_N_INPUTS]);

#ifdef __cplusplus
}
#endif
#endif
