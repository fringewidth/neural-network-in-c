#pragma once
#include "mtrx.h"

typedef struct nn_layer {
    int in;
    int out;
    mat* weights;
    mat* bias;
    void (*actfn)(float*);
} layer;

layer init_layer(int, int, void (*)(float*));
mat* forward(layer, mat*);
void relu(float*);
