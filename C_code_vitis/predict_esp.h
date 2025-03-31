#include <stdint.h>
#include "predict.h"

void predict_esp(uint64_t _inbuff[N_TREES * N_NODE_AND_LEAFS],
    /* <<--compute-params-->> */
    const unsigned load_trees,
    uint32_t _outbuff[0]);