#include "ag_controller.h"

#include <math.h>

void ag_controller_init(ag_controller_t *c, const float *developments_m, uint8_t n_gears,
                        uint8_t initial_gear, float h, uint8_t k_up, uint8_t k_down, uint8_t d) {
    if (n_gears > AG_MAX_GEARS) n_gears = AG_MAX_GEARS;
    for (uint8_t i = 0; i < n_gears; ++i) c->log_levels[i] = logf(developments_m[i]);
    c->n_gears = n_gears;
    c->gear = initial_gear < 1 ? 1 : (initial_gear > n_gears ? n_gears : initial_gear);
    c->h = h;
    c->k_up = k_up;
    c->k_down = k_down;
    c->d = d;
    c->up = c->down = 0;
    c->last_dir = 0;
    c->last_t = -1000000000;
    c->t = 0;
}

float ag_controller_development(const ag_controller_t *c) {
    return expf(c->log_levels[c->gear - 1]);
}

int8_t ag_controller_step(ag_controller_t *c, float target, uint8_t pedalling) {
    const float *lg = c->log_levels;
    const uint8_t g = c->gear, K = c->n_gears;
    int8_t step = 0;
    if (pedalling && isfinite(target)) {
        /* lg[g - 1] is log L(g); lg[g] is log L(g + 1); lg[g - 2] is log L(g - 1) */
        if (g < K && target - lg[g - 1] >= (0.5f + c->h) * (lg[g] - lg[g - 1])) ++c->up;
        else c->up = 0;
        if (g > 1 && lg[g - 1] - target >= (0.5f + c->h) * (lg[g - 1] - lg[g - 2])) ++c->down;
        else c->down = 0;
        if (c->up >= c->k_up) { step = 1; c->up = 0; }
        else if (c->down >= c->k_down) { step = -1; c->down = 0; }
        if (step && c->last_dir && step != c->last_dir && c->t - c->last_t < c->d) step = 0;
    }
    if (step) { c->last_dir = step; c->last_t = c->t; }
    c->gear = (uint8_t)(g + step);
    ++c->t;
    return step;
}
