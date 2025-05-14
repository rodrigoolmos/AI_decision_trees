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
    FILE *file = fopen(filename, "rb");
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

void evaluate_model(token_t *buf, struct feature *features, int read_samples,
                    int n_classes)
{

    int32_t prediction;
    int accuracy[256] = {0};
    int accuracy_total = 0;
    int evaluated[256] = {0};
    int evaluated_total = 0;

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

        if (features[i].prediction == prediction){
            accuracy[features[i].prediction]++;
            accuracy_total++;
        }

        evaluated[features[i].prediction]++;
        evaluated_total++;
    }

    for (int i = 0; i <= n_classes; i++){
        printf("Accuracy %f class %i num instances %i\n", 
            1.0 * accuracy[i] / evaluated[i], i, evaluated[i]);
    }

    printf("Accuracy total %f evaluates samples %i of %i\n", 
                1.0 * accuracy_total / read_samples, evaluated_total, read_samples);


}

void make_prediction(uint64_t tree[N_TREES][N_NODE_AND_LEAFS],
                    float features[N_FEATURE], int32_t *prediction)
{
    int32_t sum = 0;
    int32_t leaf_value;
    int32_t counts[N_CLASSES] = {0};

    for (int t = 0; t < N_TREES; t++) {
        uint8_t node_index = 0;
        uint8_t node_right;
        uint8_t node_left;
        uint8_t feature_index;
        float threshold;
        tree_data tree_data;

        while(1){
            tree_data.compact_data = tree[t][node_index];
            feature_index = tree_data.tree_camps.feature_index;
            threshold = tree_data.tree_camps.float_int_union.f;
            node_left = node_index + 1;
            node_right = tree_data.tree_camps.next_node_right_index;

            node_index = *(int32_t*)&features[feature_index] < *(int32_t*)&threshold ? 
                                    node_left : node_right;

            if (!(tree_data.tree_camps.leaf_or_node & 0x01))
                break;
        }

        leaf_value = tree_data.tree_camps.float_int_union.i;
        if (leaf_value >= 0 && leaf_value < N_CLASSES) {
            counts[leaf_value]++;
        } 
    }

    // Busca la clase ganadora
    int32_t best      = 0;
    int32_t best_count = counts[0];
    find_best: for (int c = 1; c < N_CLASSES; c++) {
        if (counts[c] > best_count) {
            best_count = counts[c];
            best       = c;
        }
    }

    *prediction = best;
    
}

void software_prediction(struct feature *features, int read_samples,
                            uint64_t tree[N_TREES][N_NODE_AND_LEAFS],
                            int n_classes)
{
    int32_t prediction;
    int accuracy[256] = {0};
    int accuracy_total = 0;
    int evaluated[256] = {0};
    int evaluated_total = 0;

    for (size_t i = 0; i < read_samples; i++) {
        make_prediction(tree, features[i].features, &prediction);
        if (features[i].prediction == prediction){
            accuracy[features[i].prediction]++;
            accuracy_total++;
        }

        evaluated[features[i].prediction]++;
        evaluated_total++;

    }

    for (int i = 0; i <= n_classes; i++){
        printf("Accuracy %f class %i num instances %i\n", 
            1.0 * accuracy[i] / evaluated[i], i, evaluated[i]);
    }

    printf("Accuracy total %f evaluates samples %i of %i\n", 
                1.0 * accuracy_total / read_samples, evaluated_total, read_samples);

}

void find_n_classes(struct feature features[MAX_TEST_SAMPLES], int *n_classes, int read_samples)
{

    *n_classes = features[0].prediction;

    for (int i = 1; i < read_samples; i++) {
        if (*n_classes < features[i].prediction) { *n_classes = features[i].prediction; }
    }
}

int main(int argc, char **argv)
{
    struct timespec startn, endn;
    token_t *buf;
    struct feature features_read[MAX_TEST_SAMPLES];
    int n_classes;
    int read_samples;
    tree_data tree_data[N_TREES][N_NODE_AND_LEAFS];
    unsigned long long sw_ns;

    // Validación de los argumentos: se esperan dos argumentos (dataset y modelo)
    if (argc < 3) {
        printf("Uso: %s <dataset.csv> <modelo.model>\n", argv[0]);
        return 1;
    }

    printf("\n====== %s ======\n\n", cfg_000[0].devname);

    // Cargar dataset desde el archivo recibido por línea de comandos
    printf("Cargando features desde %s...\n", argv[1]);
    read_samples = read_n_features(argv[1], MAX_TEST_SAMPLES, features_read);
    if (read_samples < 0) {
        return 1;
    }

    find_n_classes(features_read, &n_classes, read_samples);
    printf("Num clases of the dataset %i", n_classes);

    // Cargar modelo desde el archivo recibido por línea de comandos
    printf("Cargando modelo desde %s...\n", argv[2]);
    load_model(tree_data, argv[2]);

    init_parameters();

    buf = (token_t *)esp_alloc(size);

    printf("Allocating trees in the buffer\n");
    coppy_trees(tree_data, buf);

    printf("evaluate_model hardware\n");
    gettime(&startn);
    evaluate_model(buf, features_read, read_samples, n_classes);
    gettime(&endn);
    sw_ns = ts_subtract(&startn, &endn);
    printf("  > Hardware test time: %llu ns\n", sw_ns);

    printf("evaluate_model software\n");
    gettime(&startn);
    software_prediction(features_read, read_samples, tree_data, n_classes);
    gettime(&endn);
    sw_ns = ts_subtract(&startn, &endn);
    printf("  > Software test time: %llu ns\n", sw_ns);

    esp_free(buf);

    return 0;
}
