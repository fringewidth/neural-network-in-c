#include "mtrx.h"
#include "memman.h"
#include <stdio.h>
#include <stdlib.h>

float* idx_mat(const mat* mtrx, const int i, const int j) {
    return &mtrx->data[i * mtrx->w + j];
}

float* idx_mat_t(const mat* mtrx, const int i, const int j) {
    return &mtrx->data[j * mtrx->h + i];
}

mat* init_mat(const int h, const int w) {
    mat* new = (mat*)malloc(sizeof(mat));
    if (!new) {
        return NULL;
    }
    push_free(&frees, new);
    new->h = h;
    new->w = w;
    new->idxfn = idx_mat;
    new->data = (float*)malloc(h * w * sizeof(float));
    if (!new->data) {
        return NULL;
    }
    push_free(&frees, new->data);
    for (int i = 0; i < h * w; i++) {
        new->data[i] = i;
    }
    return new;
}

mat* tr(const mat* mtrx) {
    mat* new = malloc(sizeof(mat));
    if (!new) {
        return NULL;
    }
    push_free(&frees, new);
    new->h = mtrx->w;
    new->w = mtrx->h;
    new->idxfn = idx_mat_t;
    new->data = mtrx->data;
    return new;
}

mat* mtrx_elemwise(mat* a, mat* b, float (*op)(float*, float*)) {
    if (a->h != b->h || a->w != b->w)
        return NULL;
    mat* c = init_mat(a->h, a->w);
    if (!c)
        return NULL;
    for (int i = 0; i < a->h; i++) {
        for (int j = 0; j < a->w; j++) {
            *c->idxfn(c, i, j) = op(a->idxfn(a, i, j), b->idxfn(b, i, j));
        }
    }
    return c;
}

void mtrx_elemwise_ip(mat* a, mat* b, void (*op)(float*, float*)) {
    if (a->h != b->h || a->w != b->w)
        return;
    for (int i = 0; i < a->h; i++) {
        for (int j = 0; j < a->w; j++) {
            op(a->idxfn(a, i, j), b->idxfn(b, i, j));
        }
    }
}

void mtrx_elemwise_ip_unary(mat* a, void (*op)(float*)) {
    for (int i = 0; i < a->h; i++) {
        for (int j = 0; j < a->w; j++) {
            op(a->idxfn(a, i, j));
        }
    }
}

mat* mm(const mat* a, const mat* b) {
    if (a->w != b->h)
        return NULL;
    mat* c = init_mat(a->h, b->w);
    if (!c) {
        return NULL;
    }
    for (int i = 0; i < a->h; i++) {
        for (int j = 0; j < b->w; j++) {
            *c->idxfn(c, i, j) = 0;
            for (int k = 0; k < b->h; k++) {
                *c->idxfn(c, i, j) += *a->idxfn(a, i, k) * *b->idxfn(b, k, j);
            }
        }
    }
    return c;
}

void pprint(const mat* mtrx) {
    for (int i = 0; i < mtrx->h; i++) {
        for (int j = 0; j < mtrx->w; j++) {
            printf("%f\t", *mtrx->idxfn(mtrx, i, j));
        }
        printf("\n");
    }
    printf("\n");
}