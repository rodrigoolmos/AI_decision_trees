// Copyright (c) 2011-2024 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0
#include "../inc/espacc_config.h"
#include "../inc/espacc.h"
#include "hls_stream.h"
#include "hls_math.h"
#include <cstring>

void load(word_t _inbuff[SIZE_IN_CHUNK_DATA], dma_word_t *in1,
          /* <<--compute-params-->> */
	 const unsigned read_inference,
	 const unsigned load_features,
	 const unsigned load_trees,
	  dma_info_t &load_ctrl, int chunk, int batch)
{
load_data:

    const unsigned length = round_up(load_trees+load_features, VALUES_PER_WORD) / 1;
    const unsigned index = length * (batch * 1 + chunk);

    unsigned dma_length = length / VALUES_PER_WORD;
    unsigned dma_index = index / VALUES_PER_WORD;

    load_ctrl.index = dma_index;
    load_ctrl.length = dma_length;
    load_ctrl.size = SIZE_WORD_T;
    load_ctrl.user = 0;

    for (unsigned i = 0; i < dma_length; i++) {
        // Leer el elemento completo del stream en una variable temporal
        dma_word_t temp = in1[dma_index + i];

        load_label0: for (unsigned j = 0; j < VALUES_PER_WORD; j++) {
            _inbuff[i * VALUES_PER_WORD + j] = temp.word[j];
        }
    }
}

void store(word_t _outbuff[SIZE_OUT_CHUNK_DATA], dma_word_t *out,
          /* <<--compute-params-->> */
	 const unsigned read_inference,
	 const unsigned load_features,
	 const unsigned load_trees,
	   dma_info_t &store_ctrl, int chunk, int batch)
{
store_data:

    const unsigned length = round_up(read_inference, VALUES_PER_WORD) / 1;
    const unsigned store_offset = round_up(load_trees+load_features, VALUES_PER_WORD) * 1;
    const unsigned out_offset = store_offset;
    const unsigned index = out_offset + length * (batch * 1 + chunk);

    unsigned dma_length = length / VALUES_PER_WORD;
    unsigned dma_index = index / VALUES_PER_WORD;

    store_ctrl.index = dma_index;
    store_ctrl.length = dma_length;
    store_ctrl.size = SIZE_WORD_T;
    store_ctrl.user = 0;

    for (unsigned i = 0; i < dma_length; i++) {
        // Crear una variable temporal para el elemento completo del stream
        dma_word_t temp;
        
        store_label1: for (unsigned j = 0; j < VALUES_PER_WORD; j++) {
            temp.word[j] = _outbuff[i * VALUES_PER_WORD + j];
        }
        // Escribir el elemento completo en el stream
        out[dma_index + i] = temp;
    }
}


void predict(uint64_t tree[N_TREES][N_NODE_AND_LEAFS], float features[N_FEATURE],
             int32_t *prediction)
{

    int32_t sum = 0;
    int32_t leaf_value;

trees_loop:
    for (int t = 0; t < N_TREES; t++) {
#pragma HLS UNROLL factor = N_TREES

        uint8_t node_index = 0;
        uint8_t node_right;
        uint8_t node_left;
        uint8_t feature_index;
        float threshold;
        tree_data tree_data;

        for(int iteration = 0; iteration < N_NODE_AND_LEAFS; iteration++) {
            tree_data.compact_data = tree[t][node_index];
            feature_index          = tree_data.tree_camps.feature_index;
            threshold              = tree_data.tree_camps.float_int_union.f;
            node_left              = node_index + 1;
            node_right             = tree_data.tree_camps.next_node_right_index;

            node_index = *(int32_t *)&features[feature_index] < *(int32_t *)&threshold ? node_left :
                                                                                         node_right;

            if (!(tree_data.tree_camps.leaf_or_node & 0x01)) break;
        }

        leaf_value = tree_data.tree_camps.float_int_union.i;
        sum += leaf_value;
    }
    *prediction = sum;
}

void coppy_features(float features[N_FEATURE], word_t _inbuff[N_FEATURE / 2])
{
    for (int i = 0; i < N_FEATURE / 2; i++) {
        features[2 * i]     = *((float *)&_inbuff[i]);     // Parte baja de _inbuff[i]
        features[2 * i + 1] = *((float *)&_inbuff[i] + 1); // Parte alta de _inbuff[i]
    }
}

void compute(word_t _inbuff[SIZE_IN_CHUNK_DATA],
             /* <<--compute-params-->> */
             const unsigned load_trees, word_t _outbuff[SIZE_OUT_CHUNK_DATA])
{

    static __uint64_t trees[N_TREES][N_NODE_AND_LEAFS];
    float features[N_FEATURE];
    int32_t prediction;

#pragma HLS ARRAY_PARTITION variable = tree block factor = (N_TREES / 2) dim = 1
#pragma HLS ARRAY_PARTITION variable = features complete dim = 1

    if (load_trees == N_TREES * N_NODE_AND_LEAFS) {
        for (int t = 0; t < N_TREES; t++) {
            for (int n = 0; n < N_NODE_AND_LEAFS; n++) {
                trees[t][n] = _inbuff[t * N_NODE_AND_LEAFS + n];
            }
        }
    }
    else {
        coppy_features(features, _inbuff);
        predict(trees, features, &prediction);
        _outbuff[0] = (__uint64_t)prediction;
    }
}



void top(dma_word_t *out, dma_word_t *in1,
         /* <<--params-->> */
	 const unsigned conf_info_read_inference,
	 const unsigned conf_info_load_features,
	 const unsigned conf_info_load_trees,
	 dma_info_t &load_ctrl, dma_info_t &store_ctrl)
{

    /* <<--local-params-->> */
	 const unsigned read_inference = conf_info_read_inference;
	 const unsigned load_features = conf_info_load_features;
	 const unsigned load_trees = conf_info_load_trees;

    // Batching
batching:
    for (unsigned b = 0; b < 1; b++)
    {
        // Chunking
    go:
        for (int c = 0; c < 1; c++)
        {
            word_t _inbuff[SIZE_IN_CHUNK_DATA];
            word_t _outbuff[SIZE_OUT_CHUNK_DATA];

            load(_inbuff, in1,
                 /* <<--args-->> */
	 	 read_inference,
	 	 load_features,
	 	 load_trees,
                 load_ctrl, c, b);
            compute(_inbuff,
                    /* <<--args-->> */
	 	 load_trees,
                    _outbuff);
            store(_outbuff, out,
                  /* <<--args-->> */
	 	 read_inference,
	 	 load_features,
	 	 load_trees,
                  store_ctrl, c, b);
        }
    }
}
