#include "memman.h"
#include "layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

freestack frees;

int main() {
    srand((unsigned)time(NULL));

    frees.frees_idx = 0;
    frees.frees_max = 128;
    frees.freebuf = malloc(frees.frees_max * sizeof(void*));
    if (!frees.freebuf) {
        free_them_all();
        return 1;
    }

    free_them_all();
    return 0;
}
