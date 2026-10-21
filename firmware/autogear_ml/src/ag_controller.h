/*
 * ag_controller -- hysteresis controller: target log-development -> one-step shifts.
 *
 * Mirrors src/autogear/controller.py. Shift up when, for k consecutive seconds while pedalling,
 *     target - log L(g) >= (1/2 + h) * (log L(g+1) - log L(g)),
 * down in the symmetric case, hold otherwise. One step at a time; never while coasting.
 * Setting selected in the study: h = 0.6, k = 3 s (optionally k_down = 1 s for faster downshifts).
 */
#ifndef AG_CONTROLLER_H
#define AG_CONTROLLER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AG_MAX_GEARS 16

typedef struct {
    float log_levels[AG_MAX_GEARS];  /* natural log of each gear's development, ascending */
    uint8_t n_gears;
    uint8_t gear;                    /* engaged gear, 1..n_gears */
    float h;                         /* hysteresis margin, in steps */
    uint8_t k_up, k_down;            /* persistence, in seconds */
    uint8_t d;                       /* anti-reversal delay, in seconds (0 = off) */
    uint16_t up, down;               /* consecutive seconds beyond each boundary */
    int8_t last_dir;
    int32_t last_t, t;
} ag_controller_t;

/* developments in metres per crank revolution, ascending (smallest gear first) */
void ag_controller_init(ag_controller_t *c, const float *developments_m, uint8_t n_gears,
                        uint8_t initial_gear, float h, uint8_t k_up, uint8_t k_down, uint8_t d);

/* One decision per second. Returns -1 (shift down), 0 (hold) or +1 (shift up) and updates gear. */
int8_t ag_controller_step(ag_controller_t *c, float target_log_dev, uint8_t pedalling);

/* Development of the engaged gear, metres per crank revolution. */
float ag_controller_development(const ag_controller_t *c);

#ifdef __cplusplus
}
#endif
#endif
