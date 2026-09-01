#include "../ds4.h"

#include <stdio.h>

int main(void) {
    if (!ds4_test_glm_chat_preamble()) {
        fputs("GLM chat preamble test failed\n", stderr);
        return 1;
    }
    puts("tokenizer tests: OK");
    return 0;
}
