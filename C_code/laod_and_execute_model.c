#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "predict.h"


#define MAX_LINE_LENGTH 1024
#define MAX_COLUMNS 10

#define MAX_TEST_SAMPLES 3000



struct feature {
    float features[N_FEATURE];
    uint8_t prediction;
};

struct dataset {
    struct feature *data;
    int num_rows;
    int num_cols;
};

void load_model(struct tree trees[], const char *filename) {
    char magic_number[5] = {0};
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening the file\n");
        return;
    }

    fread(magic_number, 5, 1, file);

    if (!memcmp(magic_number, "model", 5)){
        for (int t = 0; t < N_TREES; t++) {
            fread(trees[t].node_leaf_value, sizeof(float), N_NODE_AND_LEAFS, file);
            fread(trees[t].feature_index, sizeof(uint8_t), N_NODE_AND_LEAFS, file);
           // fread(trees[t].next_node_left_index, sizeof(uint8_t), N_NODE_AND_LEAFS, file);
            fread(trees[t].next_node_right_index, sizeof(uint8_t), N_NODE_AND_LEAFS, file);
            fread(trees[t].leaf_or_node, sizeof(uint8_t), N_NODE_AND_LEAFS, file);
        }
    }else{
        perror("Unknown file type");
    }
    


    fclose(file);
}

int read_n_features(const char *csv_file, int n, struct feature *features) {
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
        int index = 0;

        while (token != NULL && index < N_FEATURE + 1) {
            temp[index] = strtof(token, NULL);
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
    return features_read;
}

void evaluate_model(struct tree *trees, struct feature *features, int read_samples){
    int accuracy = 0;
    uint8_t prediction;

    for (size_t i = 0; i < read_samples; i++){
        predict(trees, features[i].features, &prediction);
        if (features[i].prediction == prediction)
            accuracy++;
    }

    printf("Accuracy %f\n", 1.0 * accuracy / read_samples);
}

int main() {
    struct tree trees[N_TREES];
    float prediction;
    struct feature features[MAX_TEST_SAMPLES];
    int read_samples;

    read_samples = read_n_features("../datasets/diabetes.csv", MAX_TEST_SAMPLES, features);
    load_model(trees, "../trained_models/diabetes.model");
    evaluate_model(trees, features, read_samples);
    
    read_samples = read_n_features("../datasets/Heart_Attack.csv", MAX_TEST_SAMPLES, features);
    load_model(trees, "../trained_models/heart_attack.model");
    evaluate_model(trees, features, read_samples);

    read_samples = read_n_features("../datasets/Lung_Cancer_processed_dataset.csv", MAX_TEST_SAMPLES, features);
    load_model(trees, "../trained_models/lung_cancer.model");
    evaluate_model(trees, features, read_samples);

    read_samples = read_n_features("../datasets/anemia_processed_dataset.csv", MAX_TEST_SAMPLES, features);
    load_model(trees, "../trained_models/anemia.model");
    evaluate_model(trees, features, read_samples);

    read_samples = read_n_features("../datasets/alzheimers_processed_dataset.csv", MAX_TEST_SAMPLES, features);
    load_model(trees, "../trained_models/alzheimers.model");
    evaluate_model(trees, features, read_samples);

    return 0;
}
