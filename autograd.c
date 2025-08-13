#include "autograd.h"
#include "mtrx.h"
#include "matfns.h"
#include <stdlib.h>
#include "memman.h"
#include "layer.h"

static inline void accumulate_grad(tensor* t, mat* grad_contribution) {
    if (!t || !t->grad || !grad_contribution) return;
    mtrx_elemwise_ip(t->grad, grad_contribution, add_ip);
    free_now(grad_contribution->data);
    free_now(grad_contribution);
}

tensor* init_tensor(const int h, const int w) {
    mat* matrix = init_mat(h, w);
    mat* grad = init_mat(h, w);
    tensor* ret = malloc(sizeof(tensor));
    if (!ret) {
        return NULL;
    }
    push_free(&frees, ret);
    ret->matrix = matrix;
    ret->grad = grad;
    ret->children[0] = NULL;
    ret->children[1] = NULL;
    ret->op = NONE;
    return ret;
}

tensor* init_tensor_random(const int h, const int w, float mean, float std) {
    mat* matrix = init_mat_random(h, w, mean, std);
    mat* grad = init_mat(h, w);
    tensor* ret = malloc(sizeof(tensor));
    if (!ret) {
        return NULL;
    }
    push_free(&frees, ret);
    ret->matrix = matrix;
    ret->grad = grad;
    ret->children[0] = NULL;
    ret->children[1] = NULL;
    ret->op = NONE;
    return ret;
}

tensor* add_tnsr(tensor* a, tensor*  b) {
    tensor* ret = malloc(sizeof(tensor));
    if (!ret) {
        return NULL;
    }
    push_free(&frees, ret);
    ret->matrix = mtrx_elemwise(a->matrix, b->matrix, add);
    ret->grad = init_mat(ret->matrix->h, ret->matrix->w);
    ret->children[0] = a;
    ret->children[1] = b;
    ret->op = ADD;
    return ret;
}

tensor* mul_tnsr(tensor* a, tensor* b) {
    tensor* ret = malloc(sizeof(tensor));
    if (!ret) {
        return NULL;
    }
    push_free(&frees, ret);
    ret->matrix = mtrx_elemwise(a->matrix, b->matrix, mul);
    ret->grad = init_mat(ret->matrix->h, ret->matrix->w);
    ret->children[0] = a;
    ret->children[1] = b;
    ret->op = MUL;
    return ret;
}

tensor* relu_tnsr(tensor* a) {
    tensor* ret = malloc(sizeof(tensor));
    if (!ret) {
        return NULL;
    }
    push_free(&frees, ret);
    ret->matrix = mtrx_elemwise_unary(a->matrix, relu_value);
    ret->grad = init_mat(ret->matrix->h, ret->matrix->w);
    ret->children[0] = a;
    ret->children[1] = NULL;
    ret->op = RELU;
    return ret;
}


tensor* mm_tnsr(tensor* a, tensor* b) {
    tensor* ret = malloc(sizeof(tensor));
    if (!ret) {
        return NULL;
    }
    push_free(&frees, ret);
    ret->matrix = mm(a->matrix, b->matrix);
    ret->grad = init_mat(ret->matrix->h, ret->matrix->w);
    ret->children[0] = a;
    ret->children[1] = b;
    ret->op = LINTF;
    return ret;
}

void traverse_bckwrd_dag(tensor* d) {
    if (d->op == NONE) return;
    switch (d->op) {
        case ADD:
            mtrx_elemwise_ip(d->children[0]->grad, d->grad, add_ip);
            mtrx_elemwise_ip(d->children[1]->grad, d->grad, add_ip);
            traverse_bckwrd_dag(d->children[0]);
            traverse_bckwrd_dag(d->children[1]);
            break;
        case MUL:
            accumulate_grad(d->children[0], mtrx_elemwise(d->children[1]->matrix, d->grad, mul));
            accumulate_grad(d->children[1], mtrx_elemwise(d->children[0]->matrix, d->grad, mul));
            traverse_bckwrd_dag(d->children[0]);
            traverse_bckwrd_dag(d->children[1]);
            break;
        case LINTF: {
            mat* tb = tr(d->children[1]->matrix);
            mat* contrib0 = mm(d->grad, tb);
            accumulate_grad(d->children[0], contrib0);
            free_now(tb);

            mat* ta = tr(d->children[0]->matrix);
            mat* contrib1 = mm(ta, d->grad);
            accumulate_grad(d->children[1], contrib1);
            free_now(ta);
            traverse_bckwrd_dag(d->children[0]);
            traverse_bckwrd_dag(d->children[1]);
            break;
        }
        case RELU: {
            mat* mask = mtrx_elemwise_unary(d->children[0]->matrix, relu_grad_value);
            mat* contrib = mtrx_elemwise(mask, d->grad, mul);
            accumulate_grad(d->children[0], contrib);
            free_now(mask->data);
            free_now(mask);
            traverse_bckwrd_dag(d->children[0]);
            break;
        }
        default:
            break;
    }
}

void backward(tensor* d) {
    if (d->op == NONE) {
        return;
    }
    d->grad = init_mat_ones(d->matrix->h, d->matrix->w);
    traverse_bckwrd_dag(d);
}

