#pragma once

void *tk_from_file(const char *file);

void tk_free(void *tokenizer);

unsigned int *tk_encode(void *tokenizer, const char *text, unsigned int *len);

char *tk_decode(void *tokenizer, const unsigned int *tokens, unsigned int len);
