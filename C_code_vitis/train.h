#include "predict.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <omp.h>

#define POPULATION 1024
#define MEMORY_ACU_SIZE 10
#define MAX_NO_IMPRU 10
#define N_CLASS 3




void generate_rando_trees(tree_data trees[N_TREES][N_NODE_AND_LEAFS], 
                    uint8_t n_features, uint16_t n_trees, float max_features[N_FEATURE], float min_features[N_FEATURE]);

void mutate_population(tree_data trees_population[POPULATION][N_TREES][N_NODE_AND_LEAFS],
                        float population_accuracy[POPULATION], float max_features[N_FEATURE],
                        float min_features[N_FEATURE], uint8_t n_features, float noise_factor,
                        uint32_t boosting_i);

void crossover(tree_data trees_population[POPULATION][N_TREES][N_NODE_AND_LEAFS], uint32_t boosting_i);

void reorganize_population(float population_accuracy[POPULATION], 
                    tree_data trees_population[POPULATION][N_TREES][N_NODE_AND_LEAFS]);

int augment_features(const struct feature *original_features, int n_features, int n_col,
                     float max_features[N_FEATURE], float min_features[N_FEATURE],
                     struct feature *augmented_features, int max_augmented_features, 
                     int augmentation_factor);