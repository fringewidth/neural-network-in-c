#include "memman.h"
#include "mtrx.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>

freestack frees;

int main() {
    frees.frees_idx = 0;
    frees.frees_max = 128;
    frees.freebuf = malloc(frees.frees_max * sizeof(void*));
    if (!frees.freebuf) {
        free_them_all();
        return 1;
    }
    mat* mtrx = init_mat(5, 1);
    if (!mtrx) {
        free_them_all();
        return 1;
    }
    layer l = init_layer(5, 3, relu);
    mat* out = forward(l, mtrx);
    if (out) {
        pprint(out);
    }
    free_them_all();
}