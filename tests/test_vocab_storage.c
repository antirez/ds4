#include "../ds4.h"

#include <stdio.h>

int main(void) {
    if (!ds4_test_vocab_storage_detached()) {
        fprintf(stderr, "FAIL: tokenizer strings still depend on source mapping\n");
        return 1;
    }
    puts("PASS: tokenizer storage survives source mapping removal");
    return 0;
}
