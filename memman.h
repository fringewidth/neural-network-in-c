#pragma once
/*
don't go expecting some genius here. i did the bare minimum useless infrastructure that
frees all pointers before the program ends.
i know, i know, it's pointless and doesn't help with OOMs but 
i don't expect this program to take much memory. i'm just doing it for my ocd.
*/

typedef struct free_stack {
    void** freebuf;
    int frees_idx;
    int frees_max;
} freestack;

extern freestack frees;

void push_free(freestack* frees, void* ptr);

void free_them_all();
