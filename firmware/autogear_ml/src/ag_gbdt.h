/*
 * ag_gbdt -- inference of the boosted regression trees (target log-development).
 *
 * The trees are stored as complete binary trees of depth AG_TREE_DEPTH (ag_model_data.c, generated
 * by scripts/09_export_firmware_model.py). Node i has children 2i+1 (x < threshold) and 2i+2.
 * A prediction is exactly AG_N_TREES * AG_TREE_DEPTH comparisons and AG_N_TREES additions, the
 * same cost for every input. No heap, no recursion, no runtime library.
 */
#ifndef AG_GBDT_H
#define AG_GBDT_H

#include "ag_model_data.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Predicted log-development (natural log of metres per crank revolution) of the target gear. */
float ag_gbdt_predict(const float x[AG_N_FEATURES]);

#ifdef __cplusplus
}
#endif
#endif
