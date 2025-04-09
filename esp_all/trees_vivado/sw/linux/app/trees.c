// Copyright (c) 2011-2024 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0
#include "libesp.h"
#include "cfg.h"

static unsigned in_words_adj;
static unsigned out_words_adj;
static unsigned in_len;
static unsigned out_len;
static unsigned in_size;
static unsigned out_size;
static unsigned out_offset;
static unsigned size;


int read_n_features(const char *csv_file, int n, struct feature *features) {
    FILE *file = fopen(csv_file, "r");
    char line[MAX_LINE_LENGTH];
    int features_read = 0;
    int i;

    if (!file) {
        printf("Failed to open the features file %s\n", csv_file);
        return -1;
    }

    while (fgets(line, MAX_LINE_LENGTH, file) && features_read < n) {
        float temp[N_FEATURE + 1];
        char *token = strtok(line, ",");
        int index = 0;

        while (token != NULL && index < N_FEATURE + 1) {
            temp[index] = atof(token);
            token = strtok(NULL, ",");
            index++;
        }

        for (i = 0; i < index - 1; i++) {
            features[features_read].features[i] = temp[i];
        }
        features[features_read].prediction = (uint8_t) temp[index - 1];

        features_read++;
    }

    fclose(file);
    printf("Read %d features from %s\n", features_read, csv_file);
    return features_read;
}

void load_model(tree_data tree_data[N_TREES][N_NODE_AND_LEAFS], const char *filename)
{

    char magic_number[5] = {0};
    FILE *file           = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening the model file %s\n", filename);
        return;
    }

    fread(magic_number, 5, 1, file);

    if (!memcmp(magic_number, "model", 5)) {
        for (int t = 0; t < N_TREES; t++) {
            for (int n = 0; n < N_NODE_AND_LEAFS; n++) {
                fread(&tree_data[t][n], sizeof(uint64_t), 1, file);
            }
        }
    }
    else {
        printf("Unknown file type\n");
    }

    printf("Loaded model from %s\n", filename);

    fclose(file);
}

/* User-defined code */
static void init_parameters()
{
    if (DMA_WORD_PER_BEAT(sizeof(token_t)) == 0) {
        in_words_adj  = load_trees+load_features;
        out_words_adj = read_inference;
    }
    else {
        in_words_adj  = round_up(load_trees+load_features, DMA_WORD_PER_BEAT(sizeof(token_t)));
        out_words_adj = round_up(read_inference, DMA_WORD_PER_BEAT(sizeof(token_t)));
    }
    in_len     = in_words_adj * (1);
    out_len    = out_words_adj * (1);
    in_size    = in_len * sizeof(token_t);
    out_size   = out_len * sizeof(token_t);
    out_offset = in_len;
    size       = (out_offset * sizeof(token_t)) + out_size;
}

void coppy_trees(tree_data tree[N_TREES][N_NODE_AND_LEAFS], token_t *buf)
{
    for (int t = 0; t < N_TREES; t++) {
        for (int n = 0; n < N_NODE_AND_LEAFS; n++) {
            buf[t * N_NODE_AND_LEAFS + n] = tree[t][n].compact_data;
        }
    }
}

void evaluate_model(token_t *buf, struct feature *features, int read_samples)
{

    int accuracy = 0;
    int32_t prediction;

    printf("Loading trees...\n");
    trees_cfg_000[0].load_trees = N_TREES * N_NODE_AND_LEAFS;
    trees_cfg_000[0].load_features = 0;
    trees_cfg_000[0].read_inference = 1;
    cfg_000[0].hw_buf = buf;
    esp_run(cfg_000, NACC);

    printf("Performing inferences...\n");
    trees_cfg_000[0].load_trees = 0;
    trees_cfg_000[0].load_features = N_FEATURE;
    trees_cfg_000[0].read_inference = 1;
    for (size_t i = 0; i < read_samples; i++) {

        memcpy(buf, features[i].features, N_FEATURE * sizeof(uint32_t));
        cfg_000[0].hw_buf = buf;
        esp_run(cfg_000, NACC);

        prediction = buf[N_FEATURE];

        if (features[i].prediction == (prediction > 0)) accuracy++;
        printf("Sample %d: Prediction %d, Actual %d\n", i, prediction, features[i].prediction);
    }

    printf("Accuracy %f\n", 1.0 * accuracy / read_samples);
}

int main(int argc, char **argv)
{
    printf("\n====== %s ======\n\n", cfg_000[0].devname);
    
    token_t *buf;

    struct feature features_read[MAX_TEST_SAMPLES];
    int read_samples;
    tree_data tree_data[N_TREES][N_NODE_AND_LEAFS];
    printf("Loading features...\n");
    read_samples = read_n_features("diabetes.csv", MAX_TEST_SAMPLES, features_read);
    printf("Loading model...\n");
    load_model(tree_data, "diabetes.model");

    init_parameters();

    buf               = (token_t *)esp_alloc(size);

    printf("Allocating trees in the buffer\n");
    coppy_trees(tree_data, buf);

    printf("evaluate_model\n");
    evaluate_model(buf, features_read, read_samples);

    esp_free(buf);

    return 0;
}
