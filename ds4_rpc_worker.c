/* ds4-rpc-worker: tail-side serve loop for pipeline-parallel inference.
 *
 * The worker owns layers [start, end) of one DS4 GGUF, listens on a TCP port,
 * accepts a single head connection, performs a handshake, and then runs a
 * decode-only request/reply loop until the head sends SHUTDOWN or the
 * connection drops.  Single-process scope on purpose: the head talks to one
 * worker per session.  Reconnect and concurrent serving are Phase 5.
 *
 * This binary intentionally has no inference logic of its own.  It is the
 * normal ds4 engine plus session, configured with a partial layer range, plus
 * a thin transport adapter.  Single-host ds4 / ds4-server keep working
 * unchanged whether or not a worker is running. */

#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#define _BSD_SOURCE
#define _DEFAULT_SOURCE

#include "ds4.h"
#include "ds4_rpc.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define WORKER_DEFAULT_PORT  46434u   /* "GD43" loosely, easy to remember */
#define WORKER_DEFAULT_CTX   4096

typedef struct {
    const char *model_path;
    const char *quant;                 /* --quant q2|q4, NULL if unspecified */
    const char *mtp_path;              /* optional MTP GGUF for speculative drafting */
    int         mtp_draft_tokens;      /* max drafts per speculative step */
    float       mtp_margin;            /* confidence margin for fast verifier */
    const char *bind_host;
    uint16_t    port;
    int         layer_start;
    int         layer_end;
    int         ctx_size;
    int         routed_quant_bits;     /* 2 or 4; if 0, infer from filename */
    int         n_layer_total;         /* expected DS4_N_LAYER; for handshake */
    int         n_embd;
    int         n_hc;
    int         n_vocab;
} worker_opts;

static void worker_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --layer-start N --layer-end M [options]\n"
            "\n"
            "Required:\n"
            "  --layer-start N        First layer owned (inclusive).  Typical: 21.\n"
            "  --layer-end M          Last layer owned (exclusive).  Typical: 43.\n"
            "\n"
            "Common:\n"
            "  -m, --model FILE       GGUF path.  Wins over --quant.  Default:\n"
            "                         auto-detect Q2 or Q4 in ./gguf/ (prefers Q2).\n"
            "  --quant Q              Pick canonical 'q2' or 'q4' file in ./gguf/.\n"
            "                         Must match the head's --quant for the\n"
            "                         handshake fingerprint to succeed.\n"
            "  --listen HOST          Bind address.  Default: 0.0.0.0\n"
            "  --port N               TCP port.  Default: %u\n"
            "  --ctx N                Context size for session.  Default: %u\n"
            "  --routed-quant-bits N  2 or 4; declared in handshake so the head\n"
            "                         can refuse if it mixed q2 and q4 weights.\n"
            "  --mtp [FILE]           Enable MTP for speculative decoding.  With\n"
            "                         a FILE argument, load that GGUF; bare --mtp\n"
            "                         resolves to the canonical MTP path in\n"
            "                         ./gguf/ (from ./download_model.sh mtp).\n"
            "                         When set, the worker runs MTP drafts after\n"
            "                         each decode and ships them to the head.\n"
            "  --mtp-draft N          Max draft tokens per speculative step (1-16).\n"
            "  --mtp-margin F         Confidence margin for the fast verifier.\n"
            "  -h, --help             This help.\n",
            prog, WORKER_DEFAULT_PORT, WORKER_DEFAULT_CTX);
}

static int parse_int_arg(const char *flag, const char *val, int *out) {
    if (!val || !val[0]) {
        fprintf(stderr, "ds4-rpc-worker: %s requires a value\n", flag);
        return 1;
    }
    char *end = NULL;
    long v = strtol(val, &end, 10);
    if (end == val || *end != '\0') {
        fprintf(stderr, "ds4-rpc-worker: %s: not an integer: %s\n", flag, val);
        return 1;
    }
    *out = (int)v;
    return 0;
}

static int parse_args(int argc, char **argv, worker_opts *o) {
    o->model_path = NULL; /* resolved after parsing via ds4_resolve_model_path */
    o->quant = NULL;
    o->bind_host = "0.0.0.0";
    o->port = WORKER_DEFAULT_PORT;
    o->layer_start = -1;
    o->layer_end = -1;
    o->ctx_size = WORKER_DEFAULT_CTX;
    o->routed_quant_bits = 0;
    o->mtp_path = NULL;
    o->mtp_draft_tokens = 1;
    o->mtp_margin = 0.0f;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *next = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            worker_usage(argv[0]);
            return -1;
        } else if (!strcmp(a, "-m") || !strcmp(a, "--model")) {
            if (!next) { fprintf(stderr, "%s needs a value\n", a); return 1; }
            o->model_path = next; i++;
        } else if (!strcmp(a, "--quant")) {
            if (!next) { fprintf(stderr, "%s needs a value\n", a); return 1; }
            o->quant = next; i++;
        } else if (!strcmp(a, "--listen")) {
            if (!next) { fprintf(stderr, "%s needs a value\n", a); return 1; }
            o->bind_host = next; i++;
        } else if (!strcmp(a, "--port")) {
            int p = 0;
            if (parse_int_arg(a, next, &p)) return 1;
            if (p < 1 || p > 65535) { fprintf(stderr, "--port out of range\n"); return 1; }
            o->port = (uint16_t)p; i++;
        } else if (!strcmp(a, "--ctx")) {
            if (parse_int_arg(a, next, &o->ctx_size)) return 1;
            i++;
        } else if (!strcmp(a, "--layer-start")) {
            if (parse_int_arg(a, next, &o->layer_start)) return 1;
            i++;
        } else if (!strcmp(a, "--layer-end")) {
            if (parse_int_arg(a, next, &o->layer_end)) return 1;
            i++;
        } else if (!strcmp(a, "--routed-quant-bits")) {
            if (parse_int_arg(a, next, &o->routed_quant_bits)) return 1;
            i++;
        } else if (!strcmp(a, "--mtp")) {
            /* Accept either "--mtp PATH" or bare "--mtp" (resolves to the
             * canonical MTP GGUF in ./gguf/ via ds4_resolve_mtp_path). */
            if (next && next[0] && next[0] != '-') {
                o->mtp_path = next; i++;
            } else {
                o->mtp_path = "auto";
            }
        } else if (!strcmp(a, "--mtp-draft")) {
            if (parse_int_arg(a, next, &o->mtp_draft_tokens)) return 1;
            if (o->mtp_draft_tokens < 1) o->mtp_draft_tokens = 1;
            if (o->mtp_draft_tokens > 16) o->mtp_draft_tokens = 16;
            i++;
        } else if (!strcmp(a, "--mtp-margin")) {
            if (!next) { fprintf(stderr, "%s needs a value\n", a); return 1; }
            char *endp = NULL;
            float v = strtof(next, &endp);
            if (endp == next) {
                fprintf(stderr, "ds4-rpc-worker: --mtp-margin not a float: %s\n", next);
                return 1;
            }
            o->mtp_margin = v;
            i++;
        } else {
            fprintf(stderr, "ds4-rpc-worker: unknown argument: %s\n", a);
            worker_usage(argv[0]);
            return 1;
        }
    }
    if (o->layer_start < 0 || o->layer_end <= 0) {
        fprintf(stderr, "ds4-rpc-worker: --layer-start and --layer-end are required\n");
        worker_usage(argv[0]);
        return 1;
    }
    if (o->layer_start >= o->layer_end) {
        fprintf(stderr, "ds4-rpc-worker: empty layer range [%d, %d)\n",
                o->layer_start, o->layer_end);
        return 1;
    }

    /* Final model-path resolution: -m wins, else --quant, else filesystem
     * probe (Q2 preferred), else fall back to ds4flash.gguf symlink.  The
     * resolved path must match what the head ships in its handshake config
     * for the fingerprint check to pass. */
    {
        char resolve_err[256] = {0};
        const char *resolved = ds4_resolve_model_path(o->model_path, o->quant,
                                                      resolve_err, sizeof(resolve_err));
        if (!resolved) {
            fprintf(stderr, "ds4-rpc-worker: %s\n", resolve_err);
            return 1;
        }
        if (resolve_err[0]) fprintf(stderr, "ds4-rpc-worker: %s\n", resolve_err);
        o->model_path = resolved;
    }
    /* MTP path: same pattern.  If --mtp wasn't passed, mtp_path stays NULL
     * and the engine just won't load MTP.  If --mtp was passed (with or
     * without an explicit path), resolve to either the explicit path or
     * the canonical gguf/<MTP>.gguf. */
    if (o->mtp_path) {
        char resolve_err[256] = {0};
        const char *resolved = ds4_resolve_mtp_path(o->mtp_path,
                                                    resolve_err, sizeof(resolve_err));
        if (!resolved) {
            fprintf(stderr, "ds4-rpc-worker: %s\n",
                    resolve_err[0] ? resolve_err :
                    "--mtp requested but no MTP GGUF found in ./gguf/");
            return 1;
        }
        o->mtp_path = resolved;
        fprintf(stderr, "ds4-rpc-worker: MTP path resolved to %s\n", resolved);
    }
    return 0;
}

/* Read file size and the first 32 bytes for the cheap fingerprint embedded
 * in the handshake.  Both sides compute the same thing; a difference rejects
 * the connection. */
static int load_model_fingerprint(const char *path, uint64_t *out_size,
                                  uint8_t out_sample[32]) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4-rpc-worker: stat(%s): %s\n", path, strerror(errno));
        return 1;
    }
    *out_size = (uint64_t)st.st_size;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4-rpc-worker: open(%s): %s\n", path, strerror(errno));
        return 1;
    }
    ssize_t r = read(fd, out_sample, 32);
    close(fd);
    if (r != 32) {
        fprintf(stderr, "ds4-rpc-worker: short read of %s for fingerprint\n", path);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    worker_opts opts;
    int parse = parse_args(argc, argv, &opts);
    if (parse < 0) return 0;
    if (parse != 0) return parse;

    fprintf(stderr,
            "ds4-rpc-worker: model=%s range=[%d, %d) listen=%s:%u ctx=%d\n",
            opts.model_path, opts.layer_start, opts.layer_end,
            opts.bind_host, (unsigned)opts.port, opts.ctx_size);

    /* Open the engine with the partial layer range.  Only the tail's layers
     * are bound; the head's globals (token_embd) are skipped, and validation
     * is gated to match. */
    ds4_engine_options eopt = {
        .model_path = opts.model_path,
        .mtp_path = opts.mtp_path,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 0,
        .mtp_draft_tokens = opts.mtp_draft_tokens,
        .mtp_margin = opts.mtp_margin,
        .warm_weights = false,
        .quality = true,
        .n_layer_start = opts.layer_start,
        .n_layer_end = opts.layer_end,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &eopt) != 0 || !engine) {
        fprintf(stderr, "ds4-rpc-worker: failed to open engine\n");
        return 1;
    }

    /* Build our half of the handshake config from the just-loaded engine and
     * a quick file-fingerprint read. */
    uint64_t model_bytes = 0;
    uint8_t  model_sample[32] = {0};
    if (load_model_fingerprint(opts.model_path, &model_bytes, model_sample) != 0) {
        ds4_engine_close(engine);
        return 1;
    }
    const int quant_bits = opts.routed_quant_bits != 0
        ? opts.routed_quant_bits
        : ds4_engine_routed_quant_bits(engine);

    const bool has_mtp = ds4_engine_has_mtp(engine);
    ds4_rpc_config cfg = {
        .version                 = DS4_RPC_VERSION,
        .n_layer_total           = ds4_model_n_layer(),
        .n_embd                  = ds4_model_n_embd(),
        .n_hc                    = ds4_model_n_hc(),
        .n_vocab                 = ds4_model_n_vocab(),
        .routed_quant_bits       = (uint32_t)quant_bits,
        .tail_layer_start        = (uint32_t)opts.layer_start,
        .tail_layer_end          = (uint32_t)opts.layer_end,
        .ctx_size                = (uint32_t)opts.ctx_size,
        .tail_has_mtp            = has_mtp ? 1u : 0u,
        .tail_mtp_draft_tokens   = has_mtp ? (uint32_t)opts.mtp_draft_tokens : 0u,
        .model_file_bytes        = model_bytes,
    };
    if (has_mtp) {
        fprintf(stderr,
                "ds4-rpc-worker: MTP enabled (max drafts/step = %d, margin=%.3f)\n",
                opts.mtp_draft_tokens, (double)opts.mtp_margin);
    }
    memcpy(cfg.model_sample, model_sample, 32);

    /* Listen for the head; accept one connection. */
    ds4_rpc_handle *rpc = NULL;
    char err[512] = {0};
    if (ds4_rpc_listen_one(opts.bind_host, opts.port, &rpc, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-rpc-worker: listen: %s\n", err);
        ds4_engine_close(engine);
        return 1;
    }
    ds4_rpc_config peer = {0};
    if (ds4_rpc_handshake_server(rpc, &cfg, &peer, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-rpc-worker: handshake: %s\n", err);
        ds4_rpc_close(rpc);
        ds4_engine_close(engine);
        return 1;
    }
    fprintf(stderr, "ds4-rpc-worker: handshake ok, ready to serve\n");

    /* Create the tail session.  KV state for our owned layers lives here and
     * persists across decode requests until the head sends RESET. */
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, opts.ctx_size) != 0 || !session) {
        fprintf(stderr, "ds4-rpc-worker: session create failed\n");
        ds4_rpc_close(rpc);
        ds4_engine_close(engine);
        return 1;
    }

    const uint64_t n_residual = ds4_residual_hc_floats();
    const uint64_t n_vocab = (uint64_t)cfg.n_vocab;
    const uint32_t prefill_cap = ds4_session_prefill_cap(session);
    const uint64_t batch_residual_floats = (uint64_t)prefill_cap * n_residual;
    float *residual = (float *)malloc((size_t)(n_residual * sizeof(float)));
    float *logits = (float *)malloc((size_t)(n_vocab * sizeof(float)));
    float *batch_residual = (float *)malloc((size_t)(batch_residual_floats * sizeof(float)));
    if (!residual || !logits || !batch_residual) {
        fprintf(stderr, "ds4-rpc-worker: alloc failed\n");
        free(residual); free(logits); free(batch_residual);
        ds4_session_free(session);
        ds4_rpc_close(rpc);
        ds4_engine_close(engine);
        return 1;
    }
    fprintf(stderr,
            "ds4-rpc-worker: prefill scratch %.2f MiB (prefill_cap=%u tokens)\n",
            (double)(batch_residual_floats * sizeof(float)) / (1024.0 * 1024.0),
            prefill_cap);

    /* Serve loop. */
    int rc = 0;
    bool running = true;
    uint64_t served = 0;
    while (running) {
        ds4_rpc_op op = 0;
        if (ds4_rpc_recv_op(rpc, &op, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-rpc-worker: peek op: %s\n", err);
            rc = 1;
            break;
        }
        switch (op) {
        case DS4_RPC_OP_DECODE_REQ: {
            uint32_t token = 0, pos = 0;
            bool want_drafts = false;
            if (ds4_rpc_decode_recv(rpc, &token, &pos, &want_drafts,
                                    residual, n_residual,
                                    err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: decode recv: %s\n", err);
                rc = 1; running = false; break;
            }

            if (ds4_session_import_residual_hc(session, residual, n_residual,
                                               err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: import residual: %s\n", err);
                (void)ds4_rpc_decode_reply(rpc, NULL, n_vocab, NULL, 0, NULL, 0);
                rc = 1; running = false; break;
            }

            const int eval_rc = want_drafts
                ? ds4_session_eval(session, (int)token, err, sizeof(err))
                : ds4_session_eval_no_draft(session, (int)token, err, sizeof(err));
            if (eval_rc != 0) {
                fprintf(stderr, "ds4-rpc-worker: session eval: %s\n", err);
                (void)ds4_rpc_decode_reply(rpc, NULL, n_vocab, NULL, 0, NULL, 0);
                rc = 1; running = false; break;
            }

            const float *src = ds4_session_logits(session);
            if (!src) {
                fprintf(stderr, "ds4-rpc-worker: session has no logits after eval\n");
                (void)ds4_rpc_decode_reply(rpc, NULL, n_vocab, NULL, 0, NULL, 0);
                rc = 1; running = false; break;
            }
            memcpy(logits, src, (size_t)(n_vocab * sizeof(float)));

            uint32_t drafts[DS4_RPC_MAX_DRAFTS] = {0};
            uint32_t n_drafts = 0;
            if (want_drafts && ds4_engine_has_mtp(engine)) {
                int produced = ds4_session_mtp_drafts_after_eval(
                    session, drafts, opts.mtp_draft_tokens,
                    err, sizeof(err));
                if (produced > 0) n_drafts = (uint32_t)produced;
                else if (err[0]) {
                    fprintf(stderr, "ds4-rpc-worker: mtp drafts: %s\n", err);
                }
            }

            if (ds4_rpc_decode_reply(rpc, logits, n_vocab,
                                     n_drafts > 0 ? drafts : NULL, n_drafts,
                                     err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: reply: %s\n", err);
                rc = 1; running = false; break;
            }
            served++;
            break;
        }

        case DS4_RPC_OP_VERIFY_BATCH: {
            uint32_t v_n_tokens = 0, v_pos_start = 0, v_n_expected = 0;
            uint64_t v_n_residual = 0;
            uint32_t expected[16] = {0};
            if (ds4_rpc_verify_batch_recv(rpc, &v_n_tokens, &v_pos_start,
                                          batch_residual, batch_residual_floats,
                                          &v_n_residual,
                                          expected, 16, &v_n_expected,
                                          err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: verify_batch recv: %s\n", err);
                rc = 1; running = false; break;
            }
            uint32_t n_accepted = 0;
            if (ds4_session_verify_batch_imported_hc(session, batch_residual,
                                                     v_n_tokens, v_pos_start,
                                                     expected, v_n_expected,
                                                     &n_accepted,
                                                     logits, n_vocab,
                                                     err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: verify_batch eval: %s\n", err);
                (void)ds4_rpc_verify_batch_reply(rpc, 0, NULL, 0, NULL, 0);
                rc = 1; running = false; break;
            }
            if (ds4_rpc_verify_batch_reply(rpc, n_accepted,
                                           n_accepted > 0 ? logits : NULL,
                                           n_accepted > 0 ? n_vocab : 0,
                                           err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: verify_batch reply: %s\n", err);
                rc = 1; running = false; break;
            }
            break;
        }

        case DS4_RPC_OP_MTP_TRIM: {
            uint32_t accepted = 0;
            if (ds4_rpc_mtp_trim_recv(rpc, &accepted, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: mtp_trim recv: %s\n", err);
                rc = 1; running = false; break;
            }
            ds4_session_mtp_trim_drafts(session, accepted);
            if (ds4_rpc_mtp_trim_reply(rpc, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: mtp_trim reply: %s\n", err);
                rc = 1; running = false; break;
            }
            break;
        }

        case DS4_RPC_OP_PREFILL_REQ: {
            uint32_t n_tok = 0, pos_start = 0;
            bool want_logits = false;
            uint64_t n_recv_floats = 0;
            if (ds4_rpc_prefill_recv(rpc, &n_tok, &pos_start, &want_logits,
                                     batch_residual, batch_residual_floats,
                                     &n_recv_floats,
                                     err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: prefill recv: %s\n", err);
                rc = 1; running = false; break;
            }

            if (ds4_session_eval_batch_imported_hc(session, batch_residual,
                                                   n_tok, pos_start, want_logits,
                                                   err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: prefill eval: %s\n", err);
                (void)ds4_rpc_prefill_reply(rpc, false, NULL, 0, NULL, 0);
                rc = 1; running = false; break;
            }

            const float *src = want_logits ? ds4_session_logits(session) : NULL;
            if (want_logits && !src) {
                fprintf(stderr, "ds4-rpc-worker: prefill eval produced no logits\n");
                (void)ds4_rpc_prefill_reply(rpc, false, NULL, 0, NULL, 0);
                rc = 1; running = false; break;
            }
            if (want_logits) {
                memcpy(logits, src, (size_t)(n_vocab * sizeof(float)));
            }
            if (ds4_rpc_prefill_reply(rpc, want_logits,
                                      want_logits ? logits : NULL,
                                      want_logits ? n_vocab : 0,
                                      err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: prefill reply: %s\n", err);
                rc = 1; running = false; break;
            }
            break;
        }

        case DS4_RPC_OP_REWIND: {
            uint32_t target = 0;
            if (ds4_rpc_rewind_recv(rpc, &target, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: rewind recv: %s\n", err);
                rc = 1; running = false; break;
            }
            ds4_session_rewind(session, (int)target);
            if (ds4_rpc_rewind_reply(rpc, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: rewind reply: %s\n", err);
                rc = 1; running = false; break;
            }
            break;
        }

        case DS4_RPC_OP_RESET: {
            if (ds4_rpc_reset_recv(rpc, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: reset recv: %s\n", err);
                rc = 1; running = false; break;
            }
            ds4_session_free(session);
            session = NULL;
            if (ds4_session_create(&session, engine, opts.ctx_size) != 0 || !session) {
                fprintf(stderr, "ds4-rpc-worker: session recreate failed\n");
                rc = 1; running = false; break;
            }
            if (ds4_rpc_reset_reply(rpc, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-rpc-worker: reset reply: %s\n", err);
                rc = 1; running = false; break;
            }
            break;
        }

        case DS4_RPC_OP_SHUTDOWN: {
            (void)ds4_rpc_shutdown_recv(rpc, err, sizeof(err));
            fprintf(stderr, "ds4-rpc-worker: head requested shutdown\n");
            running = false;
            break;
        }

        default:
            fprintf(stderr, "ds4-rpc-worker: unknown op %u\n", (unsigned)op);
            rc = 1; running = false; break;
        }
    }

    fprintf(stderr, "ds4-rpc-worker: served %llu decode requests\n",
            (unsigned long long)served);

    free(residual);
    free(logits);
    free(batch_residual);
    if (session) ds4_session_free(session);
    ds4_rpc_close(rpc);
    ds4_engine_close(engine);
    return rc;
}
