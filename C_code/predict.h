#include <stdint.h>

#define N_NODE_AND_LEAFS 256  // Adjust according to the maximum number of nodes and leaves in your trees
#define N_TREES 100           // Adjust according to the number of trees in your model
#define N_FEATURE 100    // Adjust according to the number of features in your model

void predict(uint8_t tree_leaf_node[N_TREES][N_NODE_AND_LEAFS], 
             uint8_t tree_right_indexs[N_TREES][N_NODE_AND_LEAFS],
             uint8_t tree_feture_indexs[N_TREES][N_NODE_AND_LEAFS],
             float tree_node_leaf_value[N_TREES][N_NODE_AND_LEAFS],
             float features[N_FEATURE], uint8_t *prediction);