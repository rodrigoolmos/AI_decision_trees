#include "train.h"

uint8_t right_index[255] =  {128, 65, 34, 19, 12, 9, 8, 0, 0, 11, 0, 0, 16, 15, 0, 0, 18, 0, 0, 27, 24, 23, 0, 0, 26, 0, 0, 31, 30, 0, 0, 33, 0, 0, 50, 43, 40, 39, 0, 0, 42, 0, 0, 47, 46, 0, 0, 49, 0, 0, 58, 55, 54, 0, 0, 57, 0, 0, 62, 61, 0, 0, 64, 0, 0, 97, 82, 75, 72, 71, 0, 0, 74, 0, 0, 79, 78, 0, 0, 81, 0, 0, 90, 87, 86, 0, 0, 89, 0, 0, 94, 93, 0, 0, 96, 0, 0, 113, 106, 103, 102, 0, 0, 105, 0, 0, 110, 109, 0, 0, 112, 0, 0, 121, 118, 117, 0, 0, 120, 0, 0, 125, 124, 0, 0, 127, 0, 0, 192, 161, 146, 139, 136, 135, 0, 0, 138, 0, 0, 143, 142, 0, 0, 145, 0, 0, 154, 151, 150, 0, 0, 153, 0, 0, 158, 157, 0, 0, 160, 0, 0, 177, 170, 167, 166, 0, 0, 169, 0, 0, 174, 173, 0, 0, 176, 0, 0, 185, 182, 181, 0, 0, 184, 0, 0, 189, 188, 0, 0, 191, 0, 0, 224, 209, 202, 199, 198, 0, 0, 201, 0, 0, 206, 205, 0, 0, 208, 0, 0, 217, 214, 213, 0, 0, 216, 0, 0, 221, 220, 0, 0, 223, 0, 0, 240, 233, 230, 229, 0, 0, 232, 0, 0, 237, 236, 0, 0, 239, 0, 0, 248, 245, 244, 0, 0, 247, 0, 0, 252, 251, 0, 0, 254, 0, 0};

float generate_random_float(float min, float max, int* seed) {
    float random = (float)rand_r(seed) / RAND_MAX;

    float distance = max - min;
    float step = distance * 0.01; // 1% de la distancia

    return min + round(random * (distance / step)) * step; // Multiplicamos por step para ajustar el paso
}

float generate_random_float_0_1(int* seed) {
    int boolean = (rand_r(seed) % 2);

    return (float)boolean;
}

float generate_threshold(float min, float max,int* seed) {
    float random_threshold;

    if(min == 0 && max == 1){
        random_threshold = generate_random_float_0_1(seed);
    }else{
        random_threshold = generate_random_float(min, max, seed);
    }

    return random_threshold;
}

float generate_leaf_value(int *seed) {
    //float random_f = ((float) 2 * rand_r(seed) / (float)RAND_MAX) - 1.0;
    int random_f = (rand_r(seed) % N_CLASS);

    return random_f;
}

uint8_t generate_leaf_node(uint8_t prob__leaf_node, int *seed) {
    uint8_t random_8 = (uint8_t)rand_r(seed) % 100;

    return random_8 > prob__leaf_node;
}

uint8_t generate_feture_index(uint8_t feature_length, int *seed) {
    uint8_t random_8 = (uint8_t)rand_r(seed) % feature_length;

    return random_8;
}

void generate_rando_trees(tree_data trees[N_TREES][N_NODE_AND_LEAFS], 
                    uint8_t n_features, uint16_t boosting_i, float max_features[N_FEATURE], float min_features[N_FEATURE]) {

    srand(clock());
    uint8_t n_feature;
    int seed = trees;

    #pragma omp parallel for schedule(static)
    for (uint32_t tree_i = boosting_i * N_BOOSTING; 
                tree_i < (boosting_i + 1) * N_BOOSTING && tree_i < N_TREES; tree_i++){
        for (uint32_t node_i = 0; node_i < N_NODE_AND_LEAFS - 1; node_i++){
            seed = seed + omp_get_thread_num() + time(NULL) + tree_i + node_i;
            trees[tree_i][node_i].tree_camps.feature_index = generate_feture_index(n_features, &seed);
            n_feature = trees[tree_i][node_i].tree_camps.feature_index;
            
            trees[tree_i][node_i].tree_camps.leaf_or_node = 
                   (right_index[node_i] == 0) ? 0x00 : generate_leaf_node(60, &seed);

            if (node_i < 4){
                trees[tree_i][node_i].tree_camps.leaf_or_node = 1;
            }

            if (trees[tree_i][node_i].tree_camps.leaf_or_node == 0){
                trees[tree_i][node_i].tree_camps.float_int_union.i =
                    generate_leaf_value(&seed);
            }else{
                trees[tree_i][node_i].tree_camps.float_int_union.f =
                    generate_threshold(min_features[n_feature], max_features[n_feature], &seed);
            }
               
            trees[tree_i][node_i].tree_camps.next_node_right_index = right_index[node_i];
        }
    }
}

void mutate_trees(tree_data input_tree[N_TREES][N_NODE_AND_LEAFS], 
                 tree_data output_tree[N_TREES][N_NODE_AND_LEAFS],
                 uint8_t n_features, float mutation_rate, 
                 uint32_t boosting_i, float max_features[N_FEATURE], float min_features[N_FEATURE], int *seed) {

    uint32_t mutation_threshold = mutation_rate * RAND_MAX;
    uint8_t n_feature;
    uint32_t mutation_value;
    memcpy(output_tree, input_tree, sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
    
    for (uint32_t tree_i = boosting_i * N_BOOSTING; 
                tree_i < (boosting_i + 1) * N_BOOSTING && tree_i < N_TREES; tree_i++){

        for (uint32_t node_i = 0; node_i < N_NODE_AND_LEAFS - 1; node_i++){
            *seed = *seed + tree_i;
            mutation_value = rand_r(seed);
            if (mutation_value < mutation_threshold){
                *seed = *seed + node_i;
                output_tree[tree_i][node_i].tree_camps.feature_index = generate_feture_index(n_features, seed);
                n_feature = output_tree[tree_i][node_i].tree_camps.feature_index;
                output_tree[tree_i][node_i].tree_camps.leaf_or_node =  
                    (right_index[node_i] == 0) ? 0x00 : generate_leaf_node(60, seed);
                if (node_i < 4){
                    output_tree[tree_i][node_i].tree_camps.leaf_or_node = 1;
                }

                if (output_tree[tree_i][node_i].tree_camps.leaf_or_node == 0){
                    output_tree[tree_i][node_i].tree_camps.float_int_union.i =
                        generate_leaf_value(seed);
                }else{
                    output_tree[tree_i][node_i].tree_camps.float_int_union.f =
                        generate_threshold(min_features[n_feature], max_features[n_feature], seed);
                }

                output_tree[tree_i][node_i].tree_camps.next_node_right_index = right_index[node_i];
            }
        }
    }
}

void tune_nodes(tree_data input_tree[N_TREES][N_NODE_AND_LEAFS], 
                 tree_data output_tree[N_TREES][N_NODE_AND_LEAFS],
                 uint8_t n_features, float mutation_rate, 
                 uint32_t boosting_i, float max_features[N_FEATURE], float min_features[N_FEATURE], int *seed) {

    uint32_t mutation_threshold = mutation_rate * RAND_MAX;
    uint8_t n_feature;
    memcpy(output_tree, input_tree, sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
    
    for (uint32_t tree_i = boosting_i * N_BOOSTING; 
                tree_i < (boosting_i + 1) * N_BOOSTING && tree_i < N_TREES; tree_i++){

        *seed = *seed + tree_i;
        uint32_t mutation_value = rand_r(seed);
        if (mutation_value < mutation_threshold){
            for (uint32_t node_i = 0; node_i < N_NODE_AND_LEAFS - 1; node_i++){
                *seed = *seed + node_i;

                n_feature = output_tree[tree_i][node_i].tree_camps.feature_index;

                if (output_tree[tree_i][node_i].tree_camps.leaf_or_node){
                    output_tree[tree_i][node_i].tree_camps.float_int_union.f +=
                        generate_threshold(min_features[n_feature]/10, max_features[n_feature]/10, seed);
                }
            }
        }
    }
}

void reproducee_trees(tree_data mother[N_TREES][N_NODE_AND_LEAFS],
                        tree_data father[N_TREES][N_NODE_AND_LEAFS],
                        tree_data son[N_TREES][N_NODE_AND_LEAFS], uint32_t boosting_i){


    for (uint32_t tree_i = boosting_i * N_BOOSTING; 
                tree_i < (boosting_i + 1) * N_BOOSTING && tree_i < N_TREES; tree_i++){

        if(rand() % 2){
            memcpy(son[tree_i], mother[tree_i], sizeof(tree_data) * N_NODE_AND_LEAFS);
        }else{
            memcpy(son[tree_i], father[tree_i], sizeof(tree_data) * N_NODE_AND_LEAFS);
        }
    }
}

void crossover(tree_data trees_population[POPULATION][N_TREES][N_NODE_AND_LEAFS], uint32_t boosting_i){

    for (uint32_t p = POPULATION - POPULATION/10; p < POPULATION; p++){
        int index_mother = rand() % (POPULATION/80);
        int index_father = rand() % (POPULATION/80) + POPULATION/80;

        reproducee_trees(trees_population[index_mother], trees_population[index_father],
                                trees_population[p], boosting_i);
    }

}

void mutate_population(tree_data trees_population[POPULATION][N_TREES][N_NODE_AND_LEAFS],
                        float population_accuracy[POPULATION], float max_features[N_FEATURE],
                        float min_features[N_FEATURE], uint8_t n_features, float mutation_factor, 
                        uint32_t boosting_i){

    printf("Número de hilos: %d\n", omp_get_max_threads());

    #pragma omp parallel for schedule(static)
    for (uint32_t p = POPULATION/4; p < POPULATION; p++) {
        unsigned int seed = omp_get_thread_num() + time(NULL) + p;
        int index_elite = rand_r(&seed) % (POPULATION/4);

        tree_data local_tree[N_TREES][N_NODE_AND_LEAFS];
        memcpy(local_tree, trees_population[index_elite], sizeof(local_tree));
        int threshold = (int)((POPULATION/8)* population_accuracy[index_elite]);
        if (index_elite < threshold || mutation_factor >= (MEMORY_ACU_SIZE - 2)*0.02){
            tune_nodes(local_tree, trees_population[p], n_features,
                        0.5 + mutation_factor*3,
                        boosting_i, max_features, min_features, &seed);
        }else{
            mutate_trees(local_tree, trees_population[p], n_features,
                        0.5 + mutation_factor,
                        boosting_i, max_features, min_features, &seed);
        }
        
    }
}

void swapf(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

void swap_trees(tree_data trees1[N_TREES][N_NODE_AND_LEAFS], 
                tree_data trees2[N_TREES][N_NODE_AND_LEAFS]) {
    tree_data temp_trees[N_TREES][N_NODE_AND_LEAFS];
    memcpy(temp_trees, trees1, sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
    memcpy(trees1, trees2, sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
    memcpy(trees2, temp_trees, sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
}

int partition(float population_accuracy[POPULATION], 
              tree_data trees_population[POPULATION][N_TREES][N_NODE_AND_LEAFS], 
              int low, int high) {
    float pivot = population_accuracy[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (population_accuracy[j] > pivot) {
            i++;
            swapf(&population_accuracy[i], &population_accuracy[j]);
            swap_trees(trees_population[i], trees_population[j]);
        }
    }
    swapf(&population_accuracy[i + 1], &population_accuracy[high]);
    swap_trees(trees_population[i + 1], trees_population[high]);
    return i + 1;
}

void quicksort(float population_accuracy[POPULATION], 
               tree_data trees_population[POPULATION][N_TREES][N_NODE_AND_LEAFS], 
               int low, int high) {
    if (low < high) {
        int pi = partition(population_accuracy, trees_population, low, high);

        quicksort(population_accuracy, trees_population, low, pi - 1);
        quicksort(population_accuracy, trees_population, pi + 1, high);
    }
}

void randomize_percent(float population_accuracy[POPULATION], 
                          tree_data trees_population[POPULATION][N_TREES][N_NODE_AND_LEAFS],
                          float percentage_randomize) {
    int N = POPULATION;
    int M = N * percentage_randomize;  // Número de elementos a aleatorizar
    if (M < 1) M = 1;  // Asegurar al menos un elemento

    // Crear una lista de índices excluyendo el primer individuo
    int indices[N - 1];
    for (int i = 1; i < N; i++) {
        indices[i - 1] = i;
    }

    // Mezclar los índices para una selección aleatoria
    for (int i = N - 2; i > 0; i--) {
        int j = rand() % (i + 1);  // Número aleatorio entre 0 e i
        // Intercambiar indices[i] y indices[j]
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // Los primeros M índices son nuestra selección aleatoria
    // Extraer los elementos seleccionados
    float selected_accuracy[M];
    tree_data selected_trees[M][N_TREES][N_NODE_AND_LEAFS];

    for (int i = 0; i < M; i++) {
        int idx = indices[i];
        selected_accuracy[i] = population_accuracy[idx];
        memcpy(selected_trees[i], trees_population[idx], sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
    }

    // Mezclar los elementos seleccionados
    for (int i = M - 1; i > 0; i--) {
        int j = rand() % (i + 1);  // Número aleatorio entre 0 e i
        // Intercambiar accuracies
        float temp_accuracy = selected_accuracy[i];
        selected_accuracy[i] = selected_accuracy[j];
        selected_accuracy[j] = temp_accuracy;

        // Intercambiar árboles
        tree_data temp_tree[N_TREES][N_NODE_AND_LEAFS];
        memcpy(temp_tree, selected_trees[i], sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
        memcpy(selected_trees[i], selected_trees[j], sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
        memcpy(selected_trees[j], temp_tree, sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
    }

    // Colocar los elementos mezclados de vuelta en la población
    for (int i = 0; i < M; i++) {
        int idx = indices[i];
        population_accuracy[idx] = selected_accuracy[i];
        memcpy(trees_population[idx], selected_trees[i], sizeof(tree_data) * N_TREES * N_NODE_AND_LEAFS);
    }
}

void reorganize_population(float population_accuracy[POPULATION], 
                    tree_data trees_population[POPULATION][N_TREES][N_NODE_AND_LEAFS]) {
    quicksort(population_accuracy, trees_population, 0, POPULATION - 1);
    randomize_percent(population_accuracy, trees_population, 0.25);
}

void find_max_min_features(struct feature features[MAX_TEST_SAMPLES],
                                float max_features[N_FEATURE], 
                                float min_features[N_FEATURE],
                                int read_samples) {

    for (int j = 0; j < N_FEATURE; j++) {
        max_features[j] = features[0].features[j];
        min_features[j] = features[0].features[j];
    }

    for (int i = 1; i < read_samples; i++) {
        for (int j = 0; j < N_FEATURE; j++) {
            if (features[i].features[j] > max_features[j]) {
                max_features[j] = features[i].features[j];
            }
            if (features[i].features[j] < min_features[j]) {
                min_features[j] = features[i].features[j];
            }
        }
    }
}

void swap_features(struct feature* a, struct feature* b) {
    struct feature temp = *a;
    *a = *b;
    *b = temp;
}

void shuffle(struct feature* array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap_features(&array[i], &array[j]);
    }
}

int augment_features(const struct feature *original_features, int n_features, int n_col,
                     float max_features[N_FEATURE], float min_features[N_FEATURE],
                     struct feature *augmented_features, int max_augmented_features, 
                     int augmentation_factor) {

    int total_augmented = 0;
    int i, j, k;
    int seed;
    float noise_level = 0.05f;  // Nivel de ruido (ajustable según necesidad)

    // Semilla para el generador de números aleatorios
    srand((unsigned int)time(NULL));

    for (i = 0; i < n_features; i++) {
        // Verificar si hay espacio en augmented_features
        if (total_augmented >= max_augmented_features) {
            break;
        }

        // Copiar la muestra original al arreglo de aumentados
        augmented_features[total_augmented] = original_features[i];
        total_augmented++;

        // Generar muestras aumentadas
        for (j = 0; j < augmentation_factor; j++) {
            if (total_augmented >= max_augmented_features) {
                break;
            }

            struct feature new_feature = original_features[i];

            // Agregar ruido aleatorio a cada característica
            for (k = 0; k < n_col && k < N_FEATURE; k++) {
                seed = i*n_col*augmentation_factor + j*n_col + k;
                if (!(min_features == 0 && max_features == 0)){
                    float noise = generate_random_float(min_features[k]/10, max_features[k]/10, &seed);
                    new_feature.features[k] += noise;
                }
                
            }

            // Mantener la misma predicción
            new_feature.prediction = original_features[i].prediction;

            // Agregar la nueva muestra al arreglo de aumentados
            augmented_features[total_augmented] = new_feature;
            total_augmented++;
        }
    }

    return total_augmented;
}