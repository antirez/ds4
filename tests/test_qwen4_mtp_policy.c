#define DS4_NO_GPU
#include "../ds4.c"

/* Keep policy coverage independent of model weights and accelerator access.
 * Defaults preserve the existing predictor behavior; only explicit auto
 * changes the choice according to weight residency. */
int main(void) {
    const struct {
        const char *value;
        int resident;
        int streaming;
    } cases[] = {
        {NULL, 1, 1},
        {"", 1, 1},
        {"on", 1, 1},
        {"1", 1, 1},
        {"off", 0, 0},
        {"0", 0, 0},
        {"auto", 0, 1},
        {"invalid", -1, -1},
        {"2", -1, -1},
        {"-1", -1, -1},
        {"ON", -1, -1},
        {"OFF", -1, -1},
        {"AUTO", -1, -1},
        {" on", -1, -1},
        {"off ", -1, -1},
        {"auto-extra", -1, -1},
    };
    unsigned checks = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        for (int enabled = 0; enabled <= 1; enabled++) {
            for (int streaming = 0; streaming <= 1; streaming++) {
                /* With MTP disabled this setting has no effect, including
                 * malformed values: it must not enable or reject MTP. */
                const int expected = enabled ?
                    (streaming ? cases[i].streaming : cases[i].resident) : 0;
                const int actual = qwen4_mtp_prefill_resolve(enabled != 0,
                    streaming != 0, cases[i].value);
                if (actual != expected) {
                    fprintf(stderr,
                        "MTP prefill policy value=%s enabled=%d streaming=%d: got %d, expected %d\n",
                        cases[i].value ? cases[i].value : "<unset>",
                        enabled, streaming, actual, expected);
                    return 1;
                }
                checks++;
            }
        }
    }
    printf("PASS Qwen MTP prefill policy: %u cases, no model or GPU required\n", checks);
    return 0;
}
