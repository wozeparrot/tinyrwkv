#pragma once

#define half __fp16

#define TINYRWKV_DIM 0
#define TINYRWKV_LAYERS 0
#define TINYRWKV_DTYPE float

void tinyrwkv_infer(float *input, float *output, TINYRWKV_DTYPE *weights);
