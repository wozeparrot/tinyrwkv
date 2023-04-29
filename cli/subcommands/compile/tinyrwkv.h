#pragma once

#include <stdlib.h>

#define half __fp16

#define TINYRWKV_VOCAB 0
#define TINYRWKV_DIM 0
#define TINYRWKV_LAYERS 0
#define TINYRWKV_DTYPE float

#define TINYRWKV_STATE_SIZE (sizeof(float) * TINYRWKV_LAYERS * 5 * TINYRWKV_DIM)
#define TINYRWKV_INPUT_SIZE (sizeof(float) * TINYRWKV_DIM) + TINYRWKV_STATE_SIZE
#define TINYRWKV_OUTPUT_SIZE                                                   \
  (sizeof(float) * TINYRWKV_VOCAB) + TINYRWKV_STATE_SIZE

typedef struct {
  float *emb;
  TINYRWKV_DTYPE *weights;

} tinyrwkv_t;

tinyrwkv_t *tinyrwkv_init(float *emb, TINYRWKV_DTYPE *weights) {
  tinyrwkv_t *tinyrwkv = (tinyrwkv_t *)malloc(sizeof(tinyrwkv_t));
  tinyrwkv->emb = emb;
  tinyrwkv->weights = weights;

  return tinyrwkv;
}

void tinyrwkv_free(tinyrwkv_t *tinyrwkv) { free(tinyrwkv); }

void tinyrwkv_index_embed(tinyrwkv_t *tinyrwkv, unsigned int index,
                          float *input) {
  memcpy(input, (tinyrwkv->emb) + (index * TINYRWKV_DIM),
         sizeof(float) * TINYRWKV_DIM);
}

void tinyrwkv_infer(tinyrwkv_t *tinyrwkv, float *input, float *output);
