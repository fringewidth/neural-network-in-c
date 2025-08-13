#pragma once
#include "mtrx.h"

typedef struct nn_layer {
    int in;
    int out;
    mat* weights;
    mat* bias;
    void (*actfn)(float*);
} layer;

layer init_layer(const int in, const int out, void (*actfn)(float*));
mat* forward(const layer l, mat* input);

// In-place activation used by layers
void relu(float* x);

// Scalar versions used by autograd
float relu_value(const float* a);
float relu_grad_value(const float* a);
