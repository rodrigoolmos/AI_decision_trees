#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "train.h"

#define MAX_LINE_LENGTH 1024

void load_model(
            tree_data tree_data[N_TREES][N_NODE_AND_LEAFS],
            const char *filename) {

    char magic_number[5] = {0};
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening the file\n");
        return;
    }

    fread(magic_number, 5, 1, file);

    if (!memcmp(magic_number, "model", 5)){
        for (int t = 0; t < N_TREES; t++) {
            for (int n = 0; n < N_NODE_AND_LEAFS; n++) {
                fread(&tree_data[t][n], sizeof(uint64_t), 1, file);
            }
        }
    }else{
        perror("Unknown file type");
    }

    fclose(file);
}

int read_n_features(const char *csv_file, int n, struct feature *features, int *n_col) {
    FILE *file = fopen(csv_file, "r");
    char line[MAX_LINE_LENGTH];
    int features_read = 0;
    int i;

    if (!file) {
        perror("Failed to open the file");
        return -1;
    }

    while (fgets(line, MAX_LINE_LENGTH, file) && features_read < n) {
        float temp[N_FEATURE + 1];
        char *token = strtok(line, ",");
        *n_col = 0;

        while (token != NULL && (*n_col) < N_FEATURE + 1) {
            temp[*n_col] = atof(token);
            token = strtok(NULL, ",");
            (*n_col)++;
        }

        for (i = 0; i < *n_col - 1; i++) {
            features[features_read].features[i] = temp[i];
        }
        features[features_read].prediction = (uint8_t) temp[*n_col - 1];

        features_read++;
    }

    fclose(file);
    return features_read;
}

void evaluate_model(tree_data tree[N_TREES][N_NODE_AND_LEAFS], 
                    struct feature *features, int read_samples,
                    uint32_t *trees_used, int n_classes, float class_100x100[]){

    int accuracy[256] = {0};
    int accuracy_total = 0;
    int evaluated[256] = {0};
    int evaluated_total = 0;
    int32_t prediction[MAX_BURST_FEATURES];
    float features_burst[MAX_BURST_FEATURES][N_FEATURE];
    int32_t burst_size = MAX_BURST_FEATURES;
    clock_t start_time, end_time;
    double cpu_time_used;
    int32_t new_model = 1;
    start_time = clock();

    int ceil_div = (read_samples + MAX_BURST_FEATURES - 1) / MAX_BURST_FEATURES;

    // test pong
    for (int i = 0; i < ceil_div; i++){
        if (i == ceil_div -1){
            if (0 != read_samples % MAX_BURST_FEATURES){
                burst_size = read_samples % MAX_BURST_FEATURES;
            }
        }

        for (int j = 0; j < burst_size; j++){
                memcpy(features_burst[j], features[i * MAX_BURST_FEATURES + j].features, sizeof(float) *N_FEATURE);
        }
        
        predict(tree, NULL, features_burst, NULL, prediction, &burst_size, &new_model, trees_used, 0);

        for (int j = 0; j < burst_size; j++){
            if (features[i * MAX_BURST_FEATURES + j].prediction == prediction[j]) {
                accuracy[features[i * MAX_BURST_FEATURES + j].prediction]++;
                accuracy_total++;
            }
            
            evaluated[features[i * MAX_BURST_FEATURES + j].prediction]++;
            evaluated_total++;
        }
        new_model = 0;
    }
    
    for (int i = 0; i <= n_classes; i++){
        printf("Accuracy %f class %i num instances %i\n", 
            1.0 * accuracy[i] / evaluated[i], i, evaluated[i]);

        class_100x100[i] = 1.0 * accuracy[i] / evaluated[i];
    }
    

    printf("Accuracy total %f evaluates samples %i of %i\n", 
                1.0 * accuracy_total / read_samples, evaluated_total, read_samples);
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
}

int main() {

    int n_classes;
    
    struct feature features[MAX_TEST_SAMPLES];
    float class_100x100[256] = {0};
    int read_samples;
    uint32_t used_trees = N_TREES;
    int n_features;

    tree_data trees[N_TREES][N_NODE_AND_LEAFS] = {0};

    char *features_path = "/home/rodrigo/Documents/AI_decision_trees/datasets/kaggle/multi_class/updated_pollution_dataset.csv";
    char *model_path = "/home/rodrigo/Documents/AI_decision_trees/trained_models/updated_pollution_dataset.bin";

    printf("Executing model %s\n", model_path);
    printf("Executing dataset %s\n", features_path);
    read_samples = read_n_features(features_path, MAX_TEST_SAMPLES, features, &n_features);
    load_model(trees, model_path);
    n_features--; // remove predictions

    find_n_classes(features, &n_classes, read_samples);
    

    printf("Evaluation !!!!\n\n");
    evaluate_model(trees, features, read_samples,
                            &used_trees, n_classes, class_100x100);

    return 0;

}