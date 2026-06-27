#ifndef DS4_DISTRIBUTED_TEST_INTERNAL_H
#define DS4_DISTRIBUTED_TEST_INTERNAL_H

/* Test-only view of selected distributed internals.
 *
 * Why this exists:
 * - The distributed topology/route/KV-owner tests need to exercise the real
 *   planner and validator logic from ds4_distributed.c.
 * - Those code paths operate on private runtime structs such as
 *   ds4_dist_coordinator_state, ds4_dist_worker_entry, ds4_dist_route_plan,
 *   and ds4_dist_session.
 * - We intentionally do not expose those internals, or richer test-adapter
 *   APIs, through ds4_distributed.h because that would turn test scaffolding
 *   into public production surface area.
 *
 * This header therefore provides a narrow test-only bridge:
 * - enough struct definitions for tests/ds4_test.c to assemble synthetic
 *   coordinator/worker/session state
 * - thin declarations for internal logic entry points that remain implemented
 *   in production code
 *
 * It should only be included by unit tests. Production code should include
 * ds4_distributed.h instead.
 */

#include "../ds4_distributed.h"

#include <netdb.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct ds4_dist_worker_entry {
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    char model_name[128];
    uint32_t model_id;
    uint32_t quant_bits;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t has_output;
    uint32_t has_hidden;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t listen_port;
    struct ds4_dist_worker_entry *next;
} ds4_dist_worker_entry;

typedef enum {
    DS4_DIST_TOPOLOGY_FORWARD = 0,
    DS4_DIST_TOPOLOGY_REVERSE = 1,
} ds4_dist_topology;

typedef struct {
    ds4_engine *engine;
    uint32_t model_id;
    uint32_t n_layers;
    uint32_t local_start;
    uint32_t local_end;
    uint32_t ctx_size;
    ds4_dist_topology topology;
    bool local_has_output;
    bool local_can_output_head;
    bool replay_check;
    bool debug;
    bool use_control_for_work;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    uint64_t generation;
    pthread_mutex_t mu;
    ds4_dist_worker_entry *workers;
    bool shutting_down;
} ds4_dist_coordinator_state;

typedef struct {
    char host[NI_MAXHOST];
    uint32_t port;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
    int fd;
} ds4_dist_route_entry;

typedef struct {
    ds4_dist_route_entry *entry;
    uint32_t count;
    void *blob;
    uint32_t blob_bytes;
} ds4_dist_route_plan;

typedef struct {
    uint32_t layer_start;
    uint32_t layer_end;
    const ds4_dist_route_entry *entry;
    bool is_local;
} ds4_dist_kv_route_owner;

struct ds4_dist_session {
    ds4_dist_coordinator_state state;
    int listen_fd;
    pthread_t accept_tid;
    bool accept_started;
    struct {
        ds4_dist_coordinator_state *state;
        int listen_fd;
    } accept_ctx;
    ds4_dist_route_plan plan;
    bool plan_ready;
    uint64_t plan_generation;
    uint64_t session_id;
    uint64_t request_id;
    uint64_t snapshot_request_id;
};

#define DS4_DIST_ROUTE_F_OUTPUT_LOGITS 0x00000001u

int ds4_dist_test_validate_coordinator_layers(
        const ds4_dist_options *opt,
        uint32_t n_layers,
        char *err,
        size_t errlen);
int ds4_dist_test_infer_coordinator_topology(
        const ds4_dist_options *opt,
        uint32_t n_layers,
        int *topology_out,
        char *err,
        size_t errlen);
bool ds4_dist_test_build_route_plan(
        ds4_dist_coordinator_state *state,
        ds4_dist_route_plan *plan,
        char *err,
        size_t errlen);
void ds4_dist_test_route_plan_free(ds4_dist_route_plan *plan);
uint32_t ds4_dist_test_kv_route_owner_count(const ds4_dist_session *d);
int ds4_dist_test_kv_route_build_owners(
        const ds4_dist_session *d,
        ds4_dist_kv_route_owner *owners,
        uint32_t owner_count,
        char *err,
        size_t errlen);

#endif
