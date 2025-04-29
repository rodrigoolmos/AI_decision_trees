#include "predict.h"

#define N_CLASSES 32  /* sustituir por el número de clases real */

void predict(uint64_t bram_tree[N_TREES][N_NODE_AND_LEAFS],
             float   bram_features_ping[MAX_BURST_FEATURES][N_FEATURE],
             float   bram_features_pong[MAX_BURST_FEATURES][N_FEATURE],
             int32_t prediction_ping[MAX_BURST_FEATURES],
             int32_t prediction_pong[MAX_BURST_FEATURES],
             int32_t *features_burst_length,
             int32_t *load_trees,
             int32_t *trees_used,
             int32_t  ping_pong) {

    static uint64_t tree[N_TREES][N_NODE_AND_LEAFS];
    static int8_t  local_ping_pong = 0;

    float  local_features_ping[N_FEATURE];
    float  local_features_pong[N_FEATURE];
    int32_t vals[N_TREES];

    #pragma HLS TOP name=predict
    #pragma HLS INTERFACE mode=bram    port=prediction_ping
    #pragma HLS INTERFACE mode=bram    port=prediction_pong
    #pragma HLS INTERFACE mode=bram    port=bram_features_ping
    #pragma HLS INTERFACE mode=bram    port=bram_features_pong
    #pragma HLS INTERFACE mode=bram    port=bram_tree
    #pragma HLS INTERFACE mode=s_axilite port=features_burst_length bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=ping_pong          bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=load_trees         bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=trees_used         bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return             bundle=control

    #pragma HLS ARRAY_PARTITION dim=1 type=complete variable=tree
    #pragma HLS ARRAY_PARTITION dim=1 type=complete variable=local_features_ping
    #pragma HLS ARRAY_PARTITION dim=1 type=complete variable=local_features_pong

    // Carga de árboles si aplica
    if (*load_trees & 0x1) {
        copy_trees: for (uint32_t t = 0; t < *trees_used; t++) {
            for (uint32_t n = 0; n < N_NODE_AND_LEAFS; n++) {
                tree[t][n] = bram_tree[t][n];
            }
        }
    }

    // Inicialización de features ping/pong
    if (!local_ping_pong) {
        copy_pong_init: for (int i = 0; i < N_FEATURE; i++) {
            local_features_pong[i] = (ping_pong & 0x1)
                                   ? bram_features_ping[0][i]
                                   : bram_features_pong[0][i];
        }
    } else {
        copy_ping_init: for (int i = 0; i < N_FEATURE; i++) {
            local_features_ping[i] = (ping_pong & 0x1)
                                   ? bram_features_ping[0][i]
                                   : bram_features_pong[0][i];
        }
    }

    // Bucle por burst de features
    burst_loop: for (int j = 0; j < *features_burst_length; j++) {
    #pragma HLS loop_tripcount min=1 max=MAX_BURST_FEATURES

        // 1) Recoge en vals[t] el valor de hoja de cada árbol
        if (local_ping_pong) {
            predict_ping: for (int t = 0; t < *trees_used; t++) {
            #pragma HLS UNROLL factor=N_TREES_IP
                // Traversal en local_features_ping
                uint8_t node = 0;
                tree_data td;
                while (1) {
                #pragma HLS loop_tripcount min=1 max=8
                    td.compact_data = tree[t][node];
                    uint8_t fi = td.tree_camps.feature_index;
                    float   th = td.tree_camps.float_int_union.f;
                    uint8_t nl = node + 1;
                    uint8_t nr = td.tree_camps.next_node_right_index;
                    node = ((*(int32_t*)&local_features_ping[fi]) 
                            < (*(int32_t*)&th)) ? nl : nr;
                    if (!(td.tree_camps.leaf_or_node & 0x1)) break;
                }
                vals[t] = td.tree_camps.float_int_union.i;
            }
            // Pre-carga siguiente burst en pong
            copy_pong: for (int i = 0; i < N_FEATURE && j+1 < *features_burst_length; i++) {
            #pragma HLS loop_tripcount min=1 max=N_FEATURE
                local_features_pong[i] = (ping_pong & 0x1)
                                       ? bram_features_ping[j+1][i]
                                       : bram_features_pong[j+1][i];
            }
        } else {
            predict_pong: for (int t = 0; t < *trees_used; t++) {
            #pragma HLS UNROLL factor=N_TREES_IP
                // Traversal en local_features_pong
                uint8_t node = 0;
                tree_data td;
                while (1) {
                #pragma HLS loop_tripcount min=1 max=8
                    td.compact_data = tree[t][node];
                    uint8_t fi = td.tree_camps.feature_index;
                    float   th = td.tree_camps.float_int_union.f;
                    uint8_t nl = node + 1;
                    uint8_t nr = td.tree_camps.next_node_right_index;
                    node = ((*(int32_t*)&local_features_pong[fi]) 
                            < (*(int32_t*)&th)) ? nl : nr;
                    if (!(td.tree_camps.leaf_or_node & 0x1)) break;
                }
                vals[t] = td.tree_camps.float_int_union.i;
            }
            // Pre-carga siguiente burst en ping
            copy_ping: for (int i = 0; i < N_FEATURE && j+1 < *features_burst_length; i++) {
            #pragma HLS loop_tripcount min=1 max=N_FEATURE
                local_features_ping[i] = (ping_pong & 0x1)
                                       ? bram_features_ping[j+1][i]
                                       : bram_features_pong[j+1][i];
            }
        }

        // 2) Cálculo del MODO (voto mayoritario)
        int32_t counts[N_CLASSES];
        #pragma HLS ARRAY_PARTITION variable=counts complete
        // Inicializa contadores
        init_counts: for (int c = 0; c < N_CLASSES; c++) {
        #pragma HLS UNROLL
            counts[c] = 0;
        }
        // Cuenta cada voto
        count_votes: for (int t = 0; t < *trees_used; t++) {
        #pragma HLS UNROLL factor=N_TREES_IP
            int32_t cls = vals[t];
            if(cls != NULL_VOTE)
                counts[cls]++;
        }
        // Busca la clase ganadora
        int32_t best      = 0;
        int32_t best_count = counts[0];
        find_best: for (int c = 1; c < N_CLASSES; c++) {
        #pragma HLS UNROLL
            if (counts[c] > best_count) {
                best_count = counts[c];
                best       = c;
            }
        }

        // 3) Guarda la predicción
        if (ping_pong & 0x1) {
            prediction_ping[j] = best;
        } else {
            prediction_pong[j] = best;
        }

        // Alterna ping/pong para el siguiente burst
        local_ping_pong = ~local_ping_pong;
    }
}
