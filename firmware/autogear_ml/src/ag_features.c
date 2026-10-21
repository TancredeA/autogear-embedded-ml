#include "ag_features.h"

#include <string.h>

void ag_features_reset(ag_features_t *f) { memset(f, 0, sizeof(*f)); }

/* value pushed `back` samples ago (0 = current); valid for back < min(n, AG_HISTORY) */
static ag_real_t past(const ag_real_t *buf, uint32_t n, uint32_t back) {
    return buf[(n - 1 - back) % AG_HISTORY];
}

/* mean of the last w samples, or of all samples when fewer are available */
static ag_real_t mean_last(const ag_real_t *buf, uint32_t n, uint32_t w) {
    uint32_t m = n < w ? n : w;
    ag_real_t s = 0;
    for (uint32_t i = 0; i < m; ++i) s += past(buf, n, i);
    return s / (ag_real_t)m;
}

static ag_real_t mean_last_u8(const uint8_t *buf, uint32_t n, uint32_t w) {
    uint32_t m = n < w ? n : w, s = 0;
    for (uint32_t i = 0; i < m; ++i) s += buf[(n - 1 - i) % AG_HISTORY];
    return (ag_real_t)s / (ag_real_t)m;
}

/* x(t) - x(t - lag), with x(t - lag) = x(0) at the start of the segment */
static ag_real_t lag_diff(const ag_real_t *buf, uint32_t n, uint32_t lag, ag_real_t first) {
    ag_real_t now = past(buf, n, 0);
    return now - (n > lag ? past(buf, n, lag) : first);
}

float ag_features_push(ag_features_t *f, float cadence_rpm, float development_m, float slope,
                       float acc_ms2, uint8_t coasting, float out[AG_N_INPUTS]) {
    ag_real_t v;
    if (!coasting && cadence_rpm > AG_PEDAL_MIN_CADENCE_RPM) {
        v = (ag_real_t)cadence_rpm * (ag_real_t)development_m / (ag_real_t)60;
    } else {
        v = (f->n ? f->v_prev : (ag_real_t)0) + (ag_real_t)acc_ms2;
        if (v < 0) v = 0;
    }
    uint32_t slot = f->n % AG_HISTORY;
    f->v_hat[slot] = v;
    f->slope[slot] = (ag_real_t)slope;
    f->acc[slot] = (ag_real_t)acc_ms2;
    f->coasting[slot] = coasting ? 1 : 0;
    if (f->n == 0) f->v_first = v;
    f->restart_s = coasting ? 0 : (f->restart_s + 1 > 10 ? 10 : f->restart_s + 1);
    f->v_prev = v;
    uint32_t n = ++f->n;

    out[0] = (float)v;
    out[1] = (float)mean_last(f->v_hat, n, 5);
    out[2] = (float)mean_last(f->v_hat, n, 10);
    out[3] = (float)lag_diff(f->v_hat, n, 3, f->v_first);
    out[4] = (float)lag_diff(f->v_hat, n, 8, f->v_first);
    out[5] = (float)slope;
    out[6] = (float)mean_last(f->slope, n, 5);
    out[7] = (float)mean_last(f->slope, n, 15);
    out[8] = (float)acc_ms2;
    out[9] = (float)mean_last(f->acc, n, 3);
    out[10] = (float)mean_last(f->acc, n, 5);
    out[11] = (float)(coasting ? 1 : 0);
    out[12] = (float)mean_last_u8(f->coasting, n, 5);
    out[13] = (float)f->restart_s;
    return (float)v;
}
