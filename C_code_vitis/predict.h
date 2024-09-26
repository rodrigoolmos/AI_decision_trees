#include <stdint.h>

#define N_NODE_AND_LEAFS 256    // Adjust according to the maximum number of nodes and leaves in your trees
#define N_TREES 32             	// Adjust according to the number of trees in your model
#define N_FEATURE 32            // Adjust according to the number of features in your model
#define MAX_BURST_FEATURES 256  // Adjust according to the number burst features

typedef union {
    float f;
    int32_t i;
} float_int_union;

struct tree_camps {
    uint8_t leaf_or_node;
    uint8_t feature_index;
    uint8_t next_node_right_index;
    uint8_t padding;
    float_int_union float_int_union;
};

typedef union {
    struct tree_camps tree_camps;
    uint64_t compact_data;
} tree_data;

void predict(uint64_t tree[N_TREES][N_NODE_AND_LEAFS],
            float bram_features_ping[MAX_BURST_FEATURES][N_FEATURE],
            float bram_features_pong[MAX_BURST_FEATURES][N_FEATURE], 
            int32_t prediction_ping[MAX_BURST_FEATURES],
            int32_t prediction_pong[MAX_BURST_FEATURES], 
            int32_t *features_burst_length,
			int32_t ping_pong);
