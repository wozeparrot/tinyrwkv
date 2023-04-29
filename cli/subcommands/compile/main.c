#include "fcntl.h"
#include "sys/mman.h"
#include "sys/stat.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tinyrwkv.h"
#include "tokenizers.h"

typedef struct {
  int index;
  float value;
} index_value;

int compare_index_value(const void *a, const void *b) {
  index_value *ia = (index_value *)a;
  index_value *ib = (index_value *)b;
  return (int)(100.f * (ib->value - ia->value));
}

int sample(float *logits, float temperature, float tau) {
  // try to bypass nan
  for (unsigned int i = 0; i < 50277; i++) {
    if (logits[i] != logits[i]) {
      logits[i] = 0;
    }
  }

  if (temperature == 0) {
    // greedy sampling
    float max = -INFINITY;
    int max_index = 0;
    for (unsigned int i = 0; i < 50277; i++) {
      if (logits[i] > max) {
        max = logits[i];
        max_index = i;
      }
    }
    return max_index;
  }

  // -- typical sampling --
  // softmax
  float exp_sum = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    logits[i] = expf(logits[i]);
    exp_sum += logits[i];
  }

  float probs[50277];
  for (unsigned int i = 0; i < 50277; i++) {
    probs[i] = logits[i] / exp_sum;
  }

  // entropy
  float entropy = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    logits[i] = -logf(probs[i]);
    if (logits[i] == logits[i]) {
      entropy += probs[i] * logits[i];
    }
  }
  for (unsigned int i = 0; i < 50277; i++) {
    logits[i] = fabsf(logits[i] - entropy);
  }

  // sort keeping track of indices
  index_value iv[50277];
  for (unsigned int i = 0; i < 50277; i++) {
    iv[i].index = i;
    iv[i].value = logits[i];
  }
  qsort(iv, 50277, sizeof(index_value), compare_index_value);

  // sort probs using indices
  float sorted_probs[50277];
  for (unsigned int i = 0; i < 50277; i++) {
    sorted_probs[i] = probs[iv[i].index];
  }

  // cumulative sum
  float cumsum[50277];
  cumsum[0] = sorted_probs[0];
  for (unsigned int i = 1; i < 50277; i++) {
    cumsum[i] = cumsum[i - 1] + sorted_probs[i];
  }

  // calculate cutoff
  int cutoff = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    if (cumsum[i] < tau) {
      cutoff += 1;
    } else
      break;
  }

  // set probs to 0 if logits greater than cutoff
  for (unsigned int i = 0; i < 50277; i++) {
    if (logits[i] > iv[cutoff].value) {
      probs[i] = 0;
    }
  }

  // temperature
  for (unsigned int i = 0; i < 50277; i++) {
    probs[i] = powf(probs[i], 1.0 / temperature);
  }

  // normalize
  float sum = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    sum += probs[i];
  }
  for (unsigned int i = 0; i < 50277; i++) {
    probs[i] = probs[i] / sum;
  }

  // sample
  float r = (float)rand() / (float)RAND_MAX;
  float cumsum2 = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    cumsum2 += probs[i];
    if (r < cumsum2) {
      return i;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  // init random
  srand(time(NULL));

  fprintf(stderr, "Loading embedding weights...\n");
  // load embedding weights using mmap
  int fe = open("emb.bin", O_RDONLY);
  struct stat fesb;
  fstat(fe, &fesb);
  float *emb = mmap(NULL, fesb.st_size, PROT_READ, MAP_SHARED, fe, 0);
  assert(emb != MAP_FAILED);

  fprintf(stderr, "Loading weights...\n");
  // load weights using mmap
  int fw = open("weights.bin", O_RDONLY);
  struct stat fwsb;
  fstat(fw, &fwsb);
  TINYRWKV_DTYPE *weights =
      mmap(NULL, fwsb.st_size, PROT_READ, MAP_SHARED, fw, 0);
  assert(weights != MAP_FAILED);

  fprintf(stderr, "Loading initial state...\n");
  // load init state using mmap
  int fs = open("state.bin", O_RDONLY);
  struct stat fssb;
  fstat(fs, &fssb);
  float *init_state = mmap(NULL, fssb.st_size, PROT_READ, MAP_SHARED, fs, 0);
  assert(init_state != MAP_FAILED);

  tinyrwkv_t *tinyrwkv = tinyrwkv_init(emb, weights);

  fprintf(stderr, "Loading tokenizer...\n");
  // setup tokenizer
  void *tokenizer = tk_from_file("tokenizer.json");
  if (tokenizer == NULL) {
    fprintf(stderr, "Failed to load tokenizer\n");
    return 1;
  }

  // read temperature and tau from args
  float temperature = 0.85;
  float tau = 0.95;
  if (argc > 2) {
    temperature = atof(argv[1]);
    tau = atof(argv[2]);
  }
  fprintf(stderr, "Using temperature: %f, tau: %f\n", temperature, tau);

  // setup input
  float *input = malloc(TINYRWKV_INPUT_SIZE);
  memcpy(input + TINYRWKV_DIM, init_state, TINYRWKV_STATE_SIZE);

  // setup output
  float *output = malloc(TINYRWKV_OUTPUT_SIZE);

  // input string from stdin
  char input_str[4096];
  int read = fread(&input_str, sizeof(input_str), 1, stdin);

  printf("%s", input_str);
  fflush(stdout);

  // tokenize input
  unsigned int tokenized_len;
  unsigned int *input_tokens = tk_encode(tokenizer, input_str, &tokenized_len);

  // preprocess input by running it through the model
  for (int i = 0; i < tokenized_len - 1; i++) {
    tinyrwkv_index_embed(tinyrwkv, input_tokens[i], input);
    tinyrwkv_infer(tinyrwkv, input, output);
    memcpy(input + TINYRWKV_DIM, output + TINYRWKV_VOCAB, TINYRWKV_STATE_SIZE);
  }
  free(input_tokens);

  unsigned int last_token = input_tokens[tokenized_len - 1];

  // run model
  while (1) {
    tinyrwkv_index_embed(tinyrwkv, last_token, input);
    tinyrwkv_infer(tinyrwkv, input, output);
    memcpy(input + TINYRWKV_DIM, output + TINYRWKV_VOCAB, TINYRWKV_STATE_SIZE);

    // -- sampling --
    last_token = sample(output, temperature, tau);
    if (last_token == 0 && argc < 4)
      break;

    char *decoded = tk_decode(tokenizer, &last_token, 1);
    printf("%s", decoded);
    fflush(stdout);
    free(decoded);
  }

  // cleanup
  free(input);
  free(output);
  tk_free(tokenizer);
  tinyrwkv_free(tinyrwkv);
}
