#include "predict_esp.h"

void predict_esp(uint64_t _inbuff[N_TREES * N_NODE_AND_LEAFS],
             /* <<--compute-params-->> */
             const unsigned load_trees,
             uint32_t *_outbuff)
{
    static tree_data trees[N_TREES][N_NODE_AND_LEAFS];
    float *features;

    if (load_trees){
        for (int t = 0; t < N_TREES; t++){
            for (int n = 0; n < N_NODE_AND_LEAFS; n++){
                trees[t][n].compact_data = _inbuff[t * N_NODE_AND_LEAFS + n];
            }
        }
    }else{
        features = (float*)_inbuff;
        predict(trees, features, _outbuff);
    }


}