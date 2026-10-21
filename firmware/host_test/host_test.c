/*
 * Host test of the firmware library against the Python reference (vectors.csv, written by
 * scripts/09_export_firmware_model.py).
 *
 *   1. model        C inference on the Python inputs vs the Python prediction
 *   2. features     C inputs computed from the raw signals vs the Python inputs
 *   3. controller   C controller fed with the Python targets vs the Python gear sequence (exact)
 *   4. device loop  everything in C, as on the bicycle: the device measures the cadence produced
 *                   by its own gear, derives v_hat from it, predicts and shifts
 * Exit code 0 when every check passes.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "autogear_ml.h"

#define MAX_ROWS 200000

typedef struct {
    float cadence, slope, acc, x[AG_N_INPUTS], target;
    int gear, coasting, gear_py;
} row_t;

static row_t rows[MAX_ROWS];

static int run_segment(const row_t *r, int n, const float *levels, int K, int g0, double *err_model,
                       double *err_feat, long *ctrl_same, long *loop_same, long *loop_pm1) {
    ag_features_t f;
    ag_controller_t ref_ctrl, dev_ctrl;
    float x[AG_N_INPUTS];
    ag_features_reset(&f);
    ag_controller_init(&ref_ctrl, levels, (uint8_t)K, (uint8_t)g0, 0.6f, 3, 3, 0);
    ag_controller_init(&dev_ctrl, levels, (uint8_t)K, (uint8_t)g0, 0.6f, 3, 3, 0);
    ag_features_t dev_f;
    ag_features_reset(&dev_f);
    for (int t = 0; t < n; ++t) {
        /* 1. model on the Python inputs */
        double e = fabs((double)ag_gbdt_predict(r[t].x) - (double)r[t].target);
        if (e > *err_model) *err_model = e;
        /* 2. inputs from the rider's own trajectory (development of the rider's gear) */
        ag_features_push(&f, r[t].cadence, levels[r[t].gear - 1], r[t].slope, r[t].acc,
                         (uint8_t)r[t].coasting, x);
        for (int j = 0; j < AG_N_INPUTS; ++j) {
            double ej = fabs((double)x[j] - (double)r[t].x[j]);
            if (ej > *err_feat) *err_feat = ej;
        }
        /* 3. controller on the Python targets, with the Python pedalling gate (rider's cadence) */
        if (ref_ctrl.gear == r[t].gear_py) ++*ctrl_same;
        uint8_t ped_ref = !r[t].coasting && r[t].cadence > AG_PEDAL_MIN_CADENCE_RPM;
        ag_controller_step(&ref_ctrl, r[t].target, ped_ref);
        /* 4. device loop: the measured cadence is the one produced by the device's own gear */
        int gd = dev_ctrl.gear;
        uint8_t moving = !r[t].coasting && r[t].cadence > AG_PEDAL_MIN_CADENCE_RPM;
        float cad = moving ? r[t].cadence * levels[r[t].gear - 1] / levels[gd - 1] : r[t].cadence;
        if (gd == r[t].gear_py) ++*loop_same;
        if (abs(gd - r[t].gear_py) <= 1) ++*loop_pm1;
        ag_features_push(&dev_f, cad, ag_controller_development(&dev_ctrl), r[t].slope, r[t].acc,
                         (uint8_t)r[t].coasting, x);
        uint8_t ped_dev = !r[t].coasting && cad > AG_PEDAL_MIN_CADENCE_RPM;
        ag_controller_step(&dev_ctrl, ag_gbdt_predict(x), ped_dev);
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "vectors.csv";
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot open %s (run scripts/09_export_firmware_model.py)\n", path); return 2; }
    static char line[4096];
    float levels[AG_MAX_GEARS];
    int K = 0, g0 = 1, n = 0, segments = 0;
    long total = 0, ctrl_same = 0, loop_same = 0, loop_pm1 = 0;
    double err_model = 0, err_feat = 0;
    while (fgets(line, sizeof line, fp)) {
        if (line[0] == 'S') {
            if (n) { run_segment(rows, n, levels, K, g0, &err_model, &err_feat, &ctrl_same, &loop_same, &loop_pm1); total += n; }
            n = 0; ++segments;
            strtok(line, ",");                                  /* "S" */
            strtok(NULL, ",");                                  /* segment index */
            K = atoi(strtok(NULL, ",")); g0 = atoi(strtok(NULL, ","));
            for (int i = 0; i < K; ++i) levels[i] = strtof(strtok(NULL, ","), NULL);
        } else if (line[0] == 'D' && n < MAX_ROWS) {
            row_t *r = &rows[n++];
            strtok(line, ",");                                  /* "D" */
            r->cadence = strtof(strtok(NULL, ","), NULL);
            r->gear = atoi(strtok(NULL, ","));
            r->coasting = atoi(strtok(NULL, ","));
            r->slope = strtof(strtok(NULL, ","), NULL);
            r->acc = strtof(strtok(NULL, ","), NULL);
            for (int j = 0; j < AG_N_INPUTS; ++j) r->x[j] = strtof(strtok(NULL, ","), NULL);
            r->target = strtof(strtok(NULL, ","), NULL);
            r->gear_py = atoi(strtok(NULL, ","));
        }
    }
    if (n) { run_segment(rows, n, levels, K, g0, &err_model, &err_feat, &ctrl_same, &loop_same, &loop_pm1); total += n; }
    fclose(fp);

    printf("segments %d, seconds %ld\n", segments, total);
    printf("1. model       max |C - Python| prediction      %.2e\n", err_model);
    printf("2. features    max |C - Python| input           %.2e\n", err_feat);
    printf("3. controller  same gear as Python              %.4f %%\n", 100.0 * ctrl_same / total);
    printf("4. device loop same gear as Python              %.2f %%   (within one gear: %.2f %%)\n",
           100.0 * loop_same / total, 100.0 * loop_pm1 / total);
    int ok = err_model < 1e-5 && err_feat < 1e-4 && ctrl_same == total && loop_pm1 >= 0.99 * total;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
