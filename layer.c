#include "layer.h"
#include "matfns.h"
#include <math.h>
#include <stdlib.h>
#include "mtrx.h"

layer init_layer(int in, int out, void (*actfn)(float*)) {
    float stddev = sqrt(2.0 / in);
    mat* weights = init_mat_random(out, in, 0.0, stddev);
    mat* bias = init_mat(out, 1); 
    if (!weights || !bias) {
        layer ret = {0};
        return ret;
    }
    layer ret = {in, out, weights, bias, actfn};
    return ret;
}

mat* forward(layer l, mat* input) {
    if (!input || !l.weights || !l.bias) {
        return NULL;
    }
    
    mat* result = mm(l.weights, input);
    if (!result) {
        return NULL;
    }
    if (l.bias->h != result->h) {
        return NULL;
    }
    mtrx_elemwise_ip(result, l.bias, add_ip);
    if (l.actfn) {
        mtrx_elemwise_ip_unary(result, l.actfn);
    }
    return result;
}

void relu(float* x) {
    if (*x < 0)
        *x = 0;
}
