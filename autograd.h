#pragma once
#include "mtrx.h"

typedef struct autograd_tensor {
    mat* matrix;
    mat* grad;
    struct autograd_tensor* children[2];
    enum operation {
        ADD,
        MUL,
        LINTF, // (matrix * vector)
        RELU,
        NONE
    } op;
} tensor;

tensor* init_tensor(const int h, const int w);
tensor* init_tensor_random(const int h, const int w, float mean, float std);
tensor* add_tnsr(tensor* a, tensor*  b);
tensor* mul_tnsr(tensor* a, tensor* b);
tensor* relu_tnsr(tensor* a);
tensor* mm_tnsr(tensor* a, tensor* b);
void backward(tensor* d);


