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

float* idx_bdcst_h(const mat* mtrx, const int i, const int j) {
    if (!mtrx || !mtrx->data) return NULL;
    if (i < 0) return NULL; 
    if (j < 0 || j >= mtrx->w) return NULL;
    return &mtrx->data[j];
}

float* idx_bdcst_w(const mat* mtrx, const int i, const int j) {
    if (!mtrx || !mtrx->data) return NULL;
    if (j < 0) return NULL;
    if (i < 0 || i >= mtrx->h) return NULL;
    return &mtrx->data[i];
}

float* idx_bdcst_scalar(const mat* mtrx, const int i, const int j) {
    (void)i;    
    (void)j;
    if (!mtrx || !mtrx->data) return NULL;
    return &mtrx->data[0];
}

idx_fn_ptr tmap(idx_fn_ptr fn) {
    if(fn == idx_mat) return idx_mat_t;
    if(fn == idx_mat_t) return idx_mat;
    if(fn == idx_bdcst_h) return idx_bdcst_w;
    if(fn == idx_bdcst_w) return idx_bdcst_h;
    // if(fn == idx_bdcst_scalar) return idx_bdcst_scalar;
    return fn;
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

mat* init_mat_random(const int h, const int w, float mean, float std) {
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
        new->data[i] = (float)rnormal(mean, std);
    }
    return new;
}

mat* init_mat_scalar(const float val) {
    mat* out = init_mat(1, 1);
    if (out && out->data) {
        out->data[0] = val;
        return out;
    }
    return NULL;
}

mat* init_mat_ones(const int h, const int w) {
    mat* out = init_mat(h, w);
    if (out && out->data) {
        for (int i = 0; i < h * w; i++) {
            out->data[i] = 1.0f;
        }
    }
    return out;
}

mat* tr(const mat* mtrx) {
    if (!mtrx) {
        return NULL;
    }
    mat* new = (mat*)malloc(sizeof(mat));
    if (!new) {
        return NULL;
    }
    push_free(&frees, new);
    new->h = mtrx->w;
    new->w = mtrx->h;
    new->idxfn = tmap(mtrx->idxfn);
    new->data = mtrx->data;
    return new;
}

struct operands {
    mat *a;
    mat *b;
};

mat* make_view(mat* mtrx, int h, int w, idx_fn_ptr idxfn) { 
    mat* view = (mat*)malloc(sizeof(mat));
    if(!view) return NULL;
    push_free(&frees, view);
    view->data = mtrx->data;
    view->h = h;
    view->w = w;
    view->idxfn = idxfn;
    return view;
}

struct operands broadcast(struct operands opnds) {
    mat* a = opnds.a;
    mat* b = opnds.b;
    if (!a || !b) return opnds;

    if (a->h == b->h && a->w == b->w) return opnds;
    if (a->h == 1 && a->w == 1) {
        mat* view = make_view(a, b->h, b->w, idx_bdcst_scalar);
        if (view) opnds.a = view;
        return opnds;
    }

    if (a->h == b->h && a->w == 1) {
        mat* view = make_view(a, a->h, b->w, idx_bdcst_w);
        if (view) opnds.a = view;
        return opnds;
    }
    if (a->w == b->w && a->h == 1) {
        mat* view = make_view(a, b->h, a->w, idx_bdcst_h);
        if (view) opnds.a = view;
        return opnds;
    }
    if (b->h == a->h && b->w == 1) {
        mat* view = make_view(b, b->h, a->w, idx_bdcst_w);
        if (view) opnds.b = view;
        return opnds;
    }
    if (b->w == a->w && b->h == 1) {
        mat* view = make_view(b, a->h, b->w, idx_bdcst_h);
        if (view) opnds.b = view;
        return opnds;
    }

    return opnds;
}

struct operands broadcast_ip(struct operands opnds) {
    mat* a = opnds.a;
    mat* b = opnds.b;
    if (!a || !b) return opnds;

    if(b->h == a->h && b->w == 1) return broadcast(opnds);
    if(b->w == a->w && b->h == 1) return broadcast(opnds);
    return opnds;
}

mat* mtrx_elemwise(mat* a, mat* b, float (*op)(const float*, const float*)) {
    if (!a || !b || !op ) return NULL;
    struct operands opnds = {a, b};
    opnds = broadcast(opnds);
    a = opnds.a;
    b = opnds.b;
    if (a->h != b->h || a->w != b->w) return NULL;

    mat* c = init_mat(a->h, a->w);
    if (!c) return NULL;
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

mat* mtrx_elemwise_unary(mat* a, float (*op)(const float*)) {
    if (!a || !op) return NULL;
    mat* c = init_mat(a->h, a->w);
    if (!c) return NULL;
    for (int i = 0; i < a->h; i++) {
        for (int j = 0; j < a->w; j++) {
            float* a_val = a->idxfn(a, i, j);
            float* c_val = c->idxfn(c, i, j);
            if (a_val && c_val) {
                *c_val = op(a_val);
            }
        }
    }
    return c;
}

void mtrx_elemwise_ip(mat* a, mat* b, void (*op)(float*, const float*)) {
    if (!a || !b || !op ) return;
    struct operands opnds = {a, b};
    opnds = broadcast_ip(opnds);
    b = opnds.b;
    if (a->h != b->h || a->w != b->w) return;
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
    if (!a || !op) return;
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
                // *c_val = 0;
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
