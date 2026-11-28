/* Token ids of the Inkling 2 chat prompts the CLI builds (ds4_encode_chat_prompt and the message-by-message
 * ds4_chat_append_message path).  Needs a GGUF for its vocabulary; prints one JSON id array per scenario, which
 * tests/inkling2_chat_tokens.py --cli compares with llama.cpp's ids for the same conversations:
 *   1  system + user, default effort            (vector 002)
 *   2  user only, effort none                   (vector 020)
 *   3  five-turn conversation, appended message by message, effort default (vector 005) */
#define DS4_NO_GPU
#include "../ds4.c"

static void print_ids(const token_vec *t) {
    putchar('[');
    for (int i = 0; i < t->len; i++) printf("%s%d", i ? "," : "", t->v[i]);
    puts("]");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (!g_ds4_inkling2) {
        fprintf(stderr, "not an Inkling 2 GGUF\n");
        return 2;
    }
    ik_ue4m3_init();
    ds4_engine eng;
    memset(&eng, 0, sizeof(eng));
    vocab_load(&eng.vocab, &model);

    token_vec t = {0};
    encode_chat_prompt(&eng.vocab, "You are a terse assistant.", "What is 2+2?", DS4_THINK_HIGH, &t);
    print_ids(&t);
    token_vec_free(&t);

    memset(&t, 0, sizeof(t));
    encode_chat_prompt(&eng.vocab, NULL, "Hi", DS4_THINK_NONE, &t);
    print_ids(&t);
    token_vec_free(&t);

    memset(&t, 0, sizeof(t));
    ds4_chat_begin(&eng, &t);
    ds4_chat_append_think_prefix(&eng, &t, DS4_THINK_HIGH);
    ds4_chat_append_message(&eng, &t, "user", "Hi");
    ds4_chat_append_message(&eng, &t, "assistant", "Hello! How can I help?");
    ds4_chat_append_message(&eng, &t, "user", "Tell me a joke");
    ds4_chat_append_message(&eng, &t, "assistant", "Why did the chicken cross the road?");
    ds4_chat_append_message(&eng, &t, "user", "Why?");
    ds4_chat_append_assistant_prefix(&eng, &t, DS4_THINK_HIGH);
    print_ids(&t);
    token_vec_free(&t);
    return 0;
}
