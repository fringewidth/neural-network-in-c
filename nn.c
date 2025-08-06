#include "nn.h"
#include "matfns.h"
#include <stdlib.h>

layer init_layer(int in, int out, void (*actfn)(float*)) {
    mat* weights = init_mat(out, in);
    mat* bias = init_mat(out, 1);
    if (!weights || !bias) {
        layer ret = {0};
        return ret;
    }
    layer ret = {in, out, weights, bias, actfn};
    return ret;
}

mat* forward(layer l, mat* input) {
    mat* result = mm(l.weights, input);
    if (!result) {
        return NULL;
    }
    if (l.bias->h != result->h) {
        return NULL;
    }
    mtrx_elemwise_ip(result, l.bias, add_ip);
    mtrx_elemwise_ip_unary(result, l.actfn);
    return result;
}

void relu(float* x) {
    if (*x < 0)
        *x = 0;
}