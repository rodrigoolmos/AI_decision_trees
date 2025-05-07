// Copyright (c) 2011-2024 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0
#ifndef __ESP_CFG_000_H__
#define __ESP_CFG_000_H__

#include "libesp.h"
#include "trees_vivado.h"

typedef int64_t token_t;

/* <<--params-def-->> */
#define READ_INFERENCE 1
#define LOAD_FEATURES 32
#define LOAD_TREES 32768

/* <<--params-->> */
const int32_t read_inference = READ_INFERENCE;
const int32_t load_features = LOAD_FEATURES;
const int32_t load_trees = LOAD_TREES;

#define NACC 1

struct trees_vivado_access trees_cfg_000[] = {{
    /* <<--descriptor-->> */
		.read_inference = READ_INFERENCE,
		.load_features = LOAD_FEATURES,
		.load_trees = LOAD_TREES,
    .src_offset    = 0,
    .dst_offset    = 0,
    .esp.coherence = ACC_COH_NONE,
    .esp.p2p_store = 0,
    .esp.p2p_nsrcs = 0,
    .esp.p2p_srcs  = {"", "", "", ""},
}};

esp_thread_info_t cfg_000[] = {{
    .run       = true,
    .devname   = "trees_vivado.0",
    .ioctl_req = TREES_VIVADO_IOC_ACCESS,
    .esp_desc  = &(trees_cfg_000[0].esp),
}};

///////////////////////////////////////////////////////////////////////////////////

#define N_NODE_AND_LEAFS 256    // Adjust according to the maximum number of nodes and leaves in your trees
#define N_TREES 128             // Adjust according to the number of trees in your model
#define N_FEATURE 32            // Adjust according to the number of features in your model
#define N_ITEMS 768             // Adjust according to the number of items in your model
#define MAX_TEST_SAMPLES 30000  // Adjust according to the maximum number of test samples
#define MAX_LINE_LENGTH 1024    // Adjust according to the maximum line length in your CSV file
#define N_CLASSES 32            // Adjust according to the number of classes in your model

typedef union {
    float f;
    int32_t i;
} float_int_union_t;

struct tree_camps {
    uint8_t leaf_or_node;
    uint8_t feature_index;
    uint8_t next_node_right_index;
    uint8_t padding;
    float_int_union_t float_int_union;
};

typedef union {
    struct tree_camps tree_camps;
    uint64_t compact_data;
} tree_data;

struct feature {
    float features[N_FEATURE];
    uint8_t prediction;
};


#endif /* __ESP_CFG_000_H__ */
