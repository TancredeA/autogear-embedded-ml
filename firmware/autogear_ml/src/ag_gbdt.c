#include "ag_gbdt.h"

float ag_gbdt_predict(const float x[AG_N_FEATURES]) {
    float acc = ag_base_score;
    const float *thr = ag_thresholds;
    const uint8_t *feat = ag_features;
    const float *leaf = ag_leaves;
    for (int t = 0; t < AG_N_TREES; ++t) {
        unsigned i = 0;
        for (int d = 0; d < AG_TREE_DEPTH; ++d) {
            i = 2u * i + 1u + (unsigned)(x[feat[i]] >= thr[i]);
        }
        acc += leaf[i - AG_TREE_INTERNAL];
        thr += AG_TREE_INTERNAL;
        feat += AG_TREE_INTERNAL;
        leaf += AG_TREE_LEAVES;
    }
    return acc;
}
