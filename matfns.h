#pragma once

static inline float add(const float* a, const float* b) { return *a + *b; }

static inline void add_ip(float* a, const float* b) { *a += *b; }

static inline float sub(const float* a, const float* b) { return *a - *b; }

static inline void sub_ip(float* a, const float* b) { *a -= *b; }

static inline float mul(const float* a, const float* b) { return *a * *b; }

static inline void mul_ip(float* a, const float* b) { *a *= *b; }

