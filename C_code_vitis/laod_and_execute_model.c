#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "diabetes_model.h"
#include "diabetes_dataset.h"
#include "predict_esp.h"

#define ESP 1
#define NORMAL 0

#define MAX_LINE_LENGTH 1024
#define MAX_COLUMNS 10

#define MAX_TEST_SAMPLES 3000

#define N_ITE 400

struct dataset {
    struct feature *data;
    int num_rows;
    int num_cols;
};

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
    return features_read;
}

void evaluate_model(tree_data tree[N_TREES][N_NODE_AND_LEAFS], 
                    struct feature *features, int read_samples, 
                    float* time_used, int version){

    int accuracy = 0;
    int32_t prediction;
    clock_t start_time, end_time;
    start_time = clock();

    predict_esp(tree, 1, &prediction);

    for (size_t i = 0; i < read_samples; i++){
        if(version == NORMAL){
            predict(tree, features[i].features, &prediction);
        }else{
            predict_esp(features[i].features, 0, &prediction);
        }
        if (features[i].prediction == (prediction > 0))
            accuracy++;
    }

    printf("Accuracy %f\n", 1.0 * accuracy / read_samples);
    end_time = clock();
    *time_used = ((double)(end_time - start_time) / CLOCKS_PER_SEC)/read_samples;
    printf("Tiempo de ejecucion por feature: %f segundos\n", *time_used);
}

int main() {
    float time_used = 0;

    struct feature features_read[MAX_TEST_SAMPLES];
    int read_samples;
    tree_data tree_data[N_TREES][N_NODE_AND_LEAFS];

    read_samples = read_n_features("../datasets/diabetes.csv", MAX_TEST_SAMPLES, features_read);
    load_model(tree_data, "../trained_models/diabetes.model");
    evaluate_model(tree_data, features_read, read_samples, &time_used, NORMAL);

    for (int i = 0; i < 128; i++){
        for (int j = 0; j < 128; j++){
            if(tree_data[i][j].compact_data != tree[i][j]){
                printf("Error en el arbol %d, nodo %d\n", i, j);
            }
        }
    }
    

    evaluate_model(tree, features, N_ITEMS, &time_used, ESP);

    return 0;
}