#include "mtrx.h"
#include "memman.h"
#include "random.h"
#include <stdio.h>
#include <stdlib.h>

float* idx_mat(const mat* mtrx, const int i, const int j) {
    if (!mtrx || !mtrx->data || i < 0 || j < 0 || i >= mtrx->h || j >= mtrx->w) {
        return NULL;
    }
    return &mtrx->data[i * mtrx->w + j];
}

float* idx_mat_t(const mat* mtrx, const int i, const int j) {
    if (!mtrx || !mtrx->data || i < 0 || j < 0 || i >= mtrx->w || j >= mtrx->h) {
        return NULL;
    }
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
        new->data[i] = 0.0f;
    }
    return new;
}

mat* init_mat_random(const int h, const int w, float mean, float stddev) {
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
        new->data[i] = (float)rnormal(mean, stddev);
    }
    return new;
}

mat* tr(const mat* mtrx) {
    if (!mtrx) {
        return NULL;
    }
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
    if (!a || !b || !op || a->h != b->h || a->w != b->w)
        return NULL;
    mat* c = init_mat(a->h, a->w);
    if (!c)
        return NULL;
    for (int i = 0; i < a->h; i++) {
        for (int j = 0; j < a->w; j++) {
            float* a_val = a->idxfn(a, i, j);
            float* b_val = b->idxfn(b, i, j);
            float* c_val = c->idxfn(c, i, j);
            if (a_val && b_val && c_val) {
                *c_val = op(a_val, b_val);
            }
        }
    }
    return c;
}

void mtrx_elemwise_ip(mat* a, mat* b, void (*op)(float*, float*)) {
    if (!a || !b || !op || a->h != b->h || a->w != b->w)
        return;
    for (int i = 0; i < a->h; i++) {
        for (int j = 0; j < a->w; j++) {
            float* a_val = a->idxfn(a, i, j);
            float* b_val = b->idxfn(b, i, j);
            if (a_val && b_val) {
                op(a_val, b_val);
            }
        }
    }
}

void mtrx_elemwise_ip_unary(mat* a, void (*op)(float*)) {
    if (!a || !op)
        return;
    for (int i = 0; i < a->h; i++) {
        for (int j = 0; j < a->w; j++) {
            float* a_val = a->idxfn(a, i, j);
            if (a_val) {
                op(a_val);
            }
        }
    }
}

mat* mm(const mat* a, const mat* b) {
    if (!a || !b || a->w != b->h)
        return NULL;
    mat* c = init_mat(a->h, b->w);
    if (!c) {
        return NULL;
    }
    for (int i = 0; i < a->h; i++) {
        for (int j = 0; j < b->w; j++) {
            float* c_val = c->idxfn(c, i, j);
            if (c_val) {
                *c_val = 0;
                for (int k = 0; k < b->h; k++) {
                    float* a_val = a->idxfn(a, i, k);
                    float* b_val = b->idxfn(b, k, j);
                    if (a_val && b_val) {
                        *c_val += *a_val * *b_val;
                    }
                }
            }
        }
    }
    return c;
}

void pprint(const mat* mtrx) {
    if (!mtrx || !mtrx->data || !mtrx->idxfn) {
        printf("NULL matrix\n");
        return;
    }
    for (int i = 0; i < mtrx->h; i++) {
        for (int j = 0; j < mtrx->w; j++) {
            float* val = mtrx->idxfn(mtrx, i, j);
            if (val) {
                printf("%f\t", *val);
            } else {
                printf("NULL\t");
            }
        }
        printf("\n");
    }
    printf("\n");
}
