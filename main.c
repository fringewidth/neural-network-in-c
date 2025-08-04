#include <stdio.h>
#include <stdlib.h>

typedef struct free_stack {
  void **freebuf;
  int frees_idx;
  int frees_max;
} freestack;

freestack frees;

void push_free(freestack *frees, void *ptr) {
  if (frees->frees_idx >= frees->frees_max) {
    frees->frees_max *= 2;
    frees->freebuf = realloc(frees->freebuf, frees->frees_max * sizeof(void *));
  }
  frees->freebuf[frees->frees_idx++] = ptr;
}

void free_them_all() {
  int len = frees.frees_idx;
  while (len--) {
    free(frees.freebuf[len]);
  }
  free(frees.freebuf);
}

typedef struct matrix {
  int h;
  int w;
  float *data;
  float *(*idxfn)(const struct matrix *mtrx, const int i, const int j);
} mat;

float *idx_mat(const mat *mtrx, const int i, const int j) {
  return &mtrx->data[i * mtrx->w + j];
}

float *idx_mat_t(const mat *mtrx, const int i, const int j) {
  return &mtrx->data[j * mtrx->h + i];
}

mat *init_mat(const int h, const int w) {
  mat *new = (mat *)malloc(sizeof(mat));
  if (!new) {
    return NULL;
  }
  push_free(&frees, new);
  new->h = h;
  new->w = w;
  new->idxfn = idx_mat;
  new->data = (float *)malloc(h * w * sizeof(float));
  if (!new->data) {
    return NULL;
  }
  push_free(&frees, new->data);
  for (int i = 0; i < h * w; i++) {
    new->data[i] = i;
  }
  return new;
}

mat *tr(const mat *mtrx) {
  mat *new = malloc(sizeof(mat));
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

float add(float *a, float *b) { return *a + *b; }

void add_ip(float *a, float *b) { *a += *b; }

float sub(float *a, float *b) { return *a - *b; }

void sub_ip(float *a, float *b) { *a -= *b; }

float mul(float *a, float *b) { return *a * *b; }

void mul_ip(float *a, float *b) { *a *= *b; }

mat *mtrx_elemwise(mat *a, mat *b, float (*op)(float *, float *)) {
  if (a->h != b->h || a->w != b->w)
    return NULL;
  mat *c = init_mat(a->h, a->w);
  if (!c)
    return NULL;
  for (int i = 0; i < a->h; i++) {
    for (int j = 0; j < a->w; j++) {
      *c->idxfn(c, i, j) = op(a->idxfn(a, i, j), b->idxfn(b, i, j));
    }
  }
}

mat *mtrx_elemwise_ip(mat *a, mat *b, void (*op)(float *, float *)) {
  if (a->h != b->h || a->w != b->w)
    return NULL;
  for (int i = 0; i < a->h; i++) {
    for (int j = 0; j < a->w; j++) {
      op(a->idxfn(a, i, j), b->idxfn(b, i, j));
    }
  }
}

mat *mm(const mat *a, const mat *b) {
  if (a->w != b->h)
    return NULL;
  mat *c = init_mat(a->h, b->w);
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

void pprint(const mat *mtrx) {
  for (int i = 0; i < mtrx->h; i++) {
    for (int j = 0; j < mtrx->w; j++) {
      printf("%f\t", *mtrx->idxfn(mtrx, i, j));
    }
    printf("\n");
  }
  printf("\n");
}

int main() {
  frees.frees_idx = 0;
  frees.frees_max = 128;
  frees.freebuf = malloc(frees.frees_max * sizeof(void *));
  if (!frees.freebuf) {
    free_them_all();
    return 1;
  }
  mat *mtrx = init_mat(3, 4);
  if (!mtrx) {
    free_them_all();
    return 1;
  }
  pprint(mtrx);
  mat *mtrx_t = tr(mtrx);
  pprint(mtrx_t);
  mat *prod = mm(mtrx, mtrx_t);
  mtrx_elemwise_ip(prod, prod, add_ip);
  pprint(prod);
  free_them_all();
}
