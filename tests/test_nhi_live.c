/*
 * Opt-in, model-independent two-host test for the DS4 NHI CPU-copy backend.
 *
 * This binary deliberately stays out of `make test`: it requires a connected
 * /dev/tbstreamX endpoint on each host.  TCP carries only this test's HELLO,
 * READY, descriptor, and completion records; payloads must use NHI OOB mode.
 */

#include "ds4_transport.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define LIVE_MAGIC 0x44344c56u /* D4LV */
#define LIVE_VERSION 1u
#define LIVE_HELLO_BYTES 32u
#define LIVE_CONTROL_BYTES (16u + DS4_TRANSPORT_BULK_DESC_BYTES)
#define LIVE_DEFAULT_PORT "48444"
#define LIVE_DEFAULT_DEVICE "/dev/tbstream0"
#define LIVE_DEFAULT_TIMEOUT_SEC 30u
#define LIVE_MIN_MESSAGES 32u
#define LIVE_REQUIRED_WRAPS 2u
#define LIVE_MAX_PAYLOAD_BYTES (16u * 1024u * 1024u)
#define LIVE_MAX_SIZES 8u
#define LIVE_ENVELOPE_BYTES 64u

#define LIVE_ROLE_SERVER 1u
#define LIVE_ROLE_CLIENT 2u
#define LIVE_HELLO_F_GPU_ALIAS 0x00000001u

#define LIVE_CONTROL_READY 1u
#define LIVE_CONTROL_DESC 2u
#define LIVE_CONTROL_DONE 3u
#define LIVE_CONTROL_ERROR 4u

#define LIVE_SESSION_C2S UINT64_C(0x4e48494c49564501)
#define LIVE_SESSION_S2C UINT64_C(0x4e48494c49564502)

typedef struct {
    int server;
    int require_mapped;
    int require_copy;
    int require_gpu_alias;
    const char *host;
    const char *port;
    const char *device;
    uint32_t timeout_sec;
} live_options;

typedef struct {
    uint32_t role;
    uint32_t frame_size;
    uint32_t ring_size;
    uint32_t flags;
    uint64_t generation;
} live_hello;

typedef struct {
    uint32_t type;
    uint32_t status;
    ds4_transport_bulk_desc desc;
} live_control;

typedef struct {
    size_t sizes[LIVE_MAX_SIZES];
    size_t size_count;
    size_t max_payload;
    uint64_t messages;
    uint64_t frames_per_direction;
    uint64_t bytes_per_direction;
} live_plan;

typedef struct {
    uint64_t tx_mapped;
    uint64_t tx_copy;
    uint64_t rx_mapped;
    uint64_t rx_copy;
    uint64_t tx_gpu_alias;
    uint64_t rx_gpu_alias;
} live_path_counts;

#ifdef DS4_ROCM_BUILD
int ds4_test_nhi_gpu_alias_fill(void *device_ptr,
                                uint64_t bytes,
                                uint32_t direction,
                                uint64_t generation,
                                uint64_t message);
int ds4_test_nhi_gpu_alias_verify(const void *device_ptr,
                                  uint64_t bytes,
                                  uint32_t direction,
                                  uint64_t generation,
                                  uint64_t message,
                                  uint64_t *first_bad);
#endif

static int live_validate_paths(const live_options *options,
                               const live_plan *plan,
                               const live_path_counts *paths);

static void live_usage(FILE *out, const char *program) {
    fprintf(out,
            "Usage:\n"
            "  server: %s --listen [--port PORT] [--device PATH] [--timeout SEC]\n"
            "            [--require-mapped] [--require-copy] [--require-gpu-alias]\n"
            "  client: %s --connect HOST [--port PORT] [--device PATH] [--timeout SEC]\n"
            "            [--require-mapped] [--require-copy] [--require-gpu-alias]\n"
            "\n"
            "Run one command on each directly connected host. Defaults: port %s,\n"
            "device %s, timeout %u seconds. The test is destructive only to the\n"
            "exclusive endpoint session and fails if the device is already open.\n"
            "Traffic covers frame boundaries and at least two full ring wraps.\n",
            program, program, LIVE_DEFAULT_PORT, LIVE_DEFAULT_DEVICE,
            LIVE_DEFAULT_TIMEOUT_SEC);
}

static int live_error(const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    fputs("test_nhi_live: ", stderr);
    vfprintf(stderr, format, ap);
    fputc('\n', stderr);
    va_end(ap);
    return -1;
}

static int live_parse_u32(const char *text, uint32_t low, uint32_t high,
                          uint32_t *out) {
    if (!text || !text[0] || !out) return -1;
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < low || value > high) return -1;
    *out = (uint32_t)value;
    return 0;
}

static int live_parse_options(int argc, char **argv, live_options *options) {
    memset(options, 0, sizeof(*options));
    options->port = LIVE_DEFAULT_PORT;
    options->device = LIVE_DEFAULT_DEVICE;
    options->timeout_sec = LIVE_DEFAULT_TIMEOUT_SEC;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            live_usage(stdout, argv[0]);
            return 1;
        }
        if (strcmp(argv[i], "--listen") == 0) {
            if (options->server || options->host)
                return live_error("choose exactly one of --listen or --connect");
            options->server = 1;
            continue;
        }
        if (strcmp(argv[i], "--require-mapped") == 0) {
            options->require_mapped = 1;
            continue;
        }
        if (strcmp(argv[i], "--require-copy") == 0) {
            options->require_copy = 1;
            continue;
        }
        if (strcmp(argv[i], "--require-gpu-alias") == 0) {
            options->require_gpu_alias = 1;
            options->require_mapped = 1;
            continue;
        }
        if (strcmp(argv[i], "--connect") == 0) {
            if (options->server || options->host || i + 1 >= argc)
                return live_error("--connect requires one host and one role");
            options->host = argv[++i];
            if (!options->host[0]) return live_error("empty --connect host");
            continue;
        }
        if (strcmp(argv[i], "--port") == 0) {
            uint32_t port;
            if (i + 1 >= argc ||
                live_parse_u32(argv[i + 1], 1u, 65535u, &port) != 0)
                return live_error("--port must be an integer from 1 to 65535");
            options->port = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--device") == 0) {
            if (i + 1 >= argc || !argv[i + 1][0])
                return live_error("--device requires a path");
            options->device = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--timeout") == 0) {
            if (i + 1 >= argc ||
                live_parse_u32(argv[i + 1], 1u, 3600u,
                               &options->timeout_sec) != 0)
                return live_error("--timeout must be an integer from 1 to 3600");
            i++;
            continue;
        }
        return live_error("unknown argument: %s", argv[i]);
    }
    if (!options->server && !options->host) {
        live_usage(stderr, argv[0]);
        return live_error("choose --listen or --connect HOST");
    }
    return 0;
}

static void live_put_u32(unsigned char *p, uint32_t value) {
    value = htonl(value);
    memcpy(p, &value, sizeof(value));
}

static uint32_t live_get_u32(const unsigned char *p) {
    uint32_t value;
    memcpy(&value, p, sizeof(value));
    return ntohl(value);
}

static void live_put_u64(unsigned char *p, uint64_t value) {
    live_put_u32(p, (uint32_t)(value >> 32));
    live_put_u32(p + 4, (uint32_t)value);
}

static uint64_t live_get_u64(const unsigned char *p) {
    return ((uint64_t)live_get_u32(p) << 32) | live_get_u32(p + 4);
}

static int live_write(int fd, const void *buf, size_t bytes) {
    if (ds4_transport_tcp_write(fd, buf, bytes) == 0) return 0;
    return live_error("TCP write: %s", strerror(errno));
}

static int live_read(int fd, void *buf, size_t bytes) {
    int rc = ds4_transport_tcp_read(fd, buf, bytes);
    if (rc == 1) return 0;
    if (rc == 0) errno = ECONNRESET;
    return live_error("TCP read: %s", strerror(errno));
}

static int live_send_hello(int fd, const live_hello *hello) {
    unsigned char wire[LIVE_HELLO_BYTES];
    memset(wire, 0, sizeof(wire));
    live_put_u32(wire + 0, LIVE_MAGIC);
    live_put_u32(wire + 4, LIVE_VERSION);
    live_put_u32(wire + 8, hello->role);
    live_put_u32(wire + 12, hello->frame_size);
    live_put_u32(wire + 16, hello->ring_size);
    live_put_u32(wire + 20, hello->flags);
    live_put_u64(wire + 24, hello->generation);
    return live_write(fd, wire, sizeof(wire));
}

static int live_recv_hello(int fd, live_hello *hello) {
    unsigned char wire[LIVE_HELLO_BYTES];
    if (live_read(fd, wire, sizeof(wire)) != 0) return -1;
    if (live_get_u32(wire + 0) != LIVE_MAGIC ||
        live_get_u32(wire + 4) != LIVE_VERSION) {
        errno = EPROTO;
        return live_error("invalid peer HELLO record");
    }
    hello->role = live_get_u32(wire + 8);
    hello->frame_size = live_get_u32(wire + 12);
    hello->ring_size = live_get_u32(wire + 16);
    hello->flags = live_get_u32(wire + 20);
    if ((hello->flags & ~LIVE_HELLO_F_GPU_ALIAS) != 0) {
        errno = EPROTO;
        return live_error("peer HELLO has unsupported feature flags");
    }
    hello->generation = live_get_u64(wire + 24);
    return 0;
}

static int live_send_control(int fd, uint32_t type, uint32_t status,
                             const ds4_transport_bulk_desc *desc) {
    unsigned char wire[LIVE_CONTROL_BYTES];
    memset(wire, 0, sizeof(wire));
    live_put_u32(wire + 0, LIVE_MAGIC);
    live_put_u32(wire + 4, LIVE_VERSION);
    live_put_u32(wire + 8, type);
    live_put_u32(wire + 12, status);
    if (desc) {
        ds4_transport_bulk_desc_wire desc_wire;
        if (ds4_transport_bulk_desc_encode(desc, &desc_wire) != 0)
            return live_error("encode descriptor: %s", strerror(errno));
        memcpy(wire + 16, desc_wire.bytes, sizeof(desc_wire.bytes));
    }
    return live_write(fd, wire, sizeof(wire));
}

static int live_recv_control(int fd, live_control *control) {
    unsigned char wire[LIVE_CONTROL_BYTES];
    if (live_read(fd, wire, sizeof(wire)) != 0) return -1;
    if (live_get_u32(wire + 0) != LIVE_MAGIC ||
        live_get_u32(wire + 4) != LIVE_VERSION) {
        errno = EPROTO;
        return live_error("invalid peer control record");
    }
    memset(control, 0, sizeof(*control));
    control->type = live_get_u32(wire + 8);
    control->status = live_get_u32(wire + 12);
    if (control->type < LIVE_CONTROL_READY ||
        control->type > LIVE_CONTROL_ERROR) {
        errno = EPROTO;
        return live_error("unknown peer control type %u", control->type);
    }
    if ((control->type != LIVE_CONTROL_ERROR && control->status != 0) ||
        (control->type == LIVE_CONTROL_ERROR &&
         (control->status == 0 || control->status > INT32_MAX))) {
        errno = EPROTO;
        return live_error("invalid status %u for control type %u",
                          control->status, control->type);
    }
    if (control->type == LIVE_CONTROL_DESC) {
        ds4_transport_bulk_desc_wire desc_wire;
        memcpy(desc_wire.bytes, wire + 16, sizeof(desc_wire.bytes));
        if (ds4_transport_bulk_desc_decode(&desc_wire, &control->desc) != 0)
            return live_error("decode descriptor: %s", strerror(errno));
    } else {
        for (size_t i = 16; i < sizeof(wire); i++) {
            if (wire[i] != 0) {
                errno = EPROTO;
                return live_error("nonzero reserved control bytes");
            }
        }
    }
    return 0;
}

static void live_send_error_best_effort(int fd, int error_code) {
    if (fd < 0) return;
    if (error_code <= 0) error_code = EIO;
    (void)live_send_control(fd, LIVE_CONTROL_ERROR,
                            (uint32_t)error_code, NULL);
}

static int live_peer_error(const live_control *control) {
    if (control->type != LIVE_CONTROL_ERROR) return 0;
    int error_code = control->status > 0 && control->status <= INT32_MAX
        ? (int)control->status : EPROTO;
    errno = error_code;
    return live_error("peer reported failure: %s", strerror(error_code));
}

static int live_set_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFD, 0);
    if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) != 0) return -1;
    return 0;
}

static int live_set_socket_options(int fd, uint32_t timeout_sec) {
    struct timeval timeout;
    timeout.tv_sec = (time_t)timeout_sec;
    timeout.tv_usec = 0;
    int one = 1;
    if (live_set_cloexec(fd) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                   &timeout, sizeof(timeout)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
                   &timeout, sizeof(timeout)) != 0 ||
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0)
        return -1;
    return 0;
}

static int live_poll(int fd, short events, uint32_t timeout_sec) {
    struct pollfd pfd;
    memset(&pfd, 0, sizeof(pfd));
    pfd.fd = fd;
    pfd.events = events;
    const int timeout_ms = timeout_sec > (uint32_t)(INT32_MAX / 1000)
        ? INT32_MAX : (int)(timeout_sec * 1000u);
    for (;;) {
        int rc = poll(&pfd, 1, timeout_ms);
        if (rc > 0) return 0;
        if (rc == 0) {
            errno = ETIMEDOUT;
            return -1;
        }
        if (errno != EINTR) return -1;
    }
}

static int live_listen(const char *port, uint32_t timeout_sec) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    struct addrinfo *addresses = NULL;
    int gai_rc = getaddrinfo(NULL, port, &hints, &addresses);
    if (gai_rc != 0) {
        live_error("getaddrinfo for port %s: %s", port,
                   gai_strerror(gai_rc));
        errno = EINVAL;
        return -1;
    }

    int listener = -1;
    int saved = EADDRNOTAVAIL;
    for (const struct addrinfo *address = addresses;
         address; address = address->ai_next) {
        listener = socket(address->ai_family, address->ai_socktype,
                          address->ai_protocol);
        if (listener < 0) {
            saved = errno;
            continue;
        }
        int one = 1;
        (void)setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
                         &one, sizeof(one));
        if (live_set_cloexec(listener) == 0 &&
            bind(listener, address->ai_addr, address->ai_addrlen) == 0 &&
            listen(listener, 1) == 0)
            break;
        saved = errno;
        close(listener);
        listener = -1;
    }
    freeaddrinfo(addresses);
    if (listener < 0) {
        errno = saved;
        live_error("listen on port %s: %s", port, strerror(errno));
        return -1;
    }

    fprintf(stderr, "test_nhi_live: waiting on TCP port %s (%us timeout)\n",
            port, timeout_sec);
    if (live_poll(listener, POLLIN, timeout_sec) != 0) {
        saved = errno;
        live_error("accept wait: %s", strerror(saved));
        close(listener);
        errno = saved;
        return -1;
    }
    int fd;
    do {
        fd = accept(listener, NULL, NULL);
    } while (fd < 0 && errno == EINTR);
    saved = errno;
    close(listener);
    if (fd < 0) {
        errno = saved;
        live_error("accept: %s", strerror(errno));
        return -1;
    }
    if (live_set_socket_options(fd, timeout_sec) != 0) {
        saved = errno;
        live_error("accepted socket options: %s", strerror(saved));
        close(fd);
        errno = saved;
        return -1;
    }
    return fd;
}

static int live_connect(const char *host, const char *port,
                        uint32_t timeout_sec) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *addresses = NULL;
    int gai_rc = getaddrinfo(host, port, &hints, &addresses);
    if (gai_rc != 0) {
        live_error("getaddrinfo for %s:%s: %s", host, port,
                   gai_strerror(gai_rc));
        errno = EINVAL;
        return -1;
    }

    int fd = -1;
    int saved = ECONNREFUSED;
    for (const struct addrinfo *address = addresses;
         address; address = address->ai_next) {
        fd = socket(address->ai_family, address->ai_socktype,
                    address->ai_protocol);
        if (fd < 0) {
            saved = errno;
            continue;
        }
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
            saved = errno;
            close(fd);
            fd = -1;
            continue;
        }
        int rc = connect(fd, address->ai_addr, address->ai_addrlen);
        if (rc != 0 && errno == EINPROGRESS) {
            rc = live_poll(fd, POLLOUT, timeout_sec);
            if (rc == 0) {
                socklen_t len = sizeof(saved);
                if (getsockopt(fd, SOL_SOCKET, SO_ERROR,
                               &saved, &len) != 0)
                    saved = errno;
                rc = saved == 0 ? 0 : -1;
            } else {
                saved = errno;
            }
        } else if (rc != 0) {
            saved = errno;
        }
        if (rc == 0 && fcntl(fd, F_SETFL, flags) == 0) break;
        if (rc == 0) saved = errno;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(addresses);
    if (fd < 0) {
        errno = saved;
        live_error("connect to %s:%s: %s", host, port, strerror(errno));
        return -1;
    }
    if (live_set_socket_options(fd, timeout_sec) != 0) {
        saved = errno;
        live_error("connected socket options: %s", strerror(saved));
        close(fd);
        errno = saved;
        return -1;
    }
    return fd;
}

static uint64_t live_generation(void) {
    struct timespec realtime;
    struct timespec monotonic;
    memset(&realtime, 0, sizeof(realtime));
    memset(&monotonic, 0, sizeof(monotonic));
    (void)clock_gettime(CLOCK_REALTIME, &realtime);
    (void)clock_gettime(CLOCK_MONOTONIC, &monotonic);
    uint64_t value = (uint64_t)realtime.tv_sec;
    value ^= (uint64_t)realtime.tv_nsec << 32;
    value ^= (uint64_t)monotonic.tv_nsec << 1;
    value ^= (uint64_t)(unsigned)getpid() << 17;
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value ? value : UINT64_C(1);
}

static uint64_t live_now_ns(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static void live_add_size(live_plan *plan, uint64_t candidate,
                          size_t payload_cap) {
    if (candidate == 0 || candidate > payload_cap ||
        candidate > UINT32_MAX || plan->size_count >= LIVE_MAX_SIZES)
        return;
    const size_t value = (size_t)candidate;
    for (size_t i = 0; i < plan->size_count; i++) {
        if (plan->sizes[i] == value) return;
    }
    plan->sizes[plan->size_count++] = value;
    if (value > plan->max_payload) plan->max_payload = value;
}

static uint32_t live_frames(size_t bytes, uint32_t frame_size) {
    const uint64_t total = LIVE_ENVELOPE_BYTES + (uint64_t)bytes;
    return (uint32_t)((total + frame_size - 1u) / frame_size);
}

static int live_build_plan(uint32_t frame_size, uint32_t wrap_ring,
                           size_t max_oob, live_plan *plan) {
    memset(plan, 0, sizeof(*plan));
    if (frame_size <= LIVE_ENVELOPE_BYTES || wrap_ring < 2u ||
        max_oob == 0)
        return live_error("unusable negotiated NHI geometry");
    size_t payload_cap = max_oob;
    if (payload_cap > LIVE_MAX_PAYLOAD_BYTES)
        payload_cap = LIVE_MAX_PAYLOAD_BYTES;

    live_add_size(plan, 1u, payload_cap);
    live_add_size(plan, 64u, payload_cap);
    live_add_size(plan, frame_size / 2u, payload_cap);
    live_add_size(plan, (uint64_t)frame_size - LIVE_ENVELOPE_BYTES,
                  payload_cap);
    live_add_size(plan, (uint64_t)frame_size - LIVE_ENVELOPE_BYTES + 1u,
                  payload_cap);
    live_add_size(plan, (uint64_t)frame_size * 2u - LIVE_ENVELOPE_BYTES,
                  payload_cap);
    live_add_size(plan, (uint64_t)frame_size * 2u -
                  LIVE_ENVELOPE_BYTES + 17u, payload_cap);
    live_add_size(plan, payload_cap, payload_cap);
    if (plan->size_count == 0 || plan->max_payload == 0)
        return live_error("no payload sizes fit negotiated NHI geometry");

    const uint64_t required_frames =
        (uint64_t)LIVE_REQUIRED_WRAPS * wrap_ring;
    while (plan->messages < LIVE_MIN_MESSAGES ||
           plan->frames_per_direction < required_frames) {
        size_t bytes = plan->sizes[plan->messages % plan->size_count];
        plan->messages++;
        plan->bytes_per_direction += bytes;
        plan->frames_per_direction += live_frames(bytes, frame_size);
    }
    return 0;
}

static unsigned char live_pattern_byte(uint32_t direction,
                                       uint64_t generation,
                                       uint64_t message, size_t offset) {
    uint64_t value = (uint64_t)offset * UINT64_C(0x9e3779b185ebca87);
    value ^= (uint64_t)direction * UINT64_C(0xd6e8feb86659fd93);
    value ^= generation * UINT64_C(0xe7037ed1a0b428db);
    value ^= (message + 1u) * UINT64_C(0xa0761d6478bd642f);
    value ^= value >> 29;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 32;
    return (unsigned char)value;
}

static void live_fill_pattern(unsigned char *buf, size_t bytes,
                              uint32_t direction, uint64_t generation,
                              uint64_t message) {
    for (size_t i = 0; i < bytes; i++)
        buf[i] = live_pattern_byte(direction, generation, message, i);
}

static int live_verify_pattern(const unsigned char *buf, size_t bytes,
                               uint32_t direction, uint64_t generation,
                               uint64_t message) {
    for (size_t i = 0; i < bytes; i++) {
        unsigned char expected = live_pattern_byte(direction, generation,
                                                   message, i);
        if (buf[i] != expected) {
            errno = EILSEQ;
            return live_error("payload mismatch at message %" PRIu64
                              ", byte %zu: got 0x%02x, expected 0x%02x",
                              message + 1u, i, (unsigned)buf[i],
                              (unsigned)expected);
        }
    }
    return 0;
}

static uint32_t live_element_bits(size_t bytes, uint64_t message) {
    if (bytes % 4u == 0 && message % 3u == 2u) return 32u;
    if (bytes % 2u == 0 && message % 3u != 0u) return 16u;
    return 8u;
}

static void live_expected_tag(int server_to_client, uint64_t message,
                              size_t bytes, uint32_t *kind,
                              uint32_t *element_bits, uint64_t *session_id) {
    uint32_t bits = live_element_bits(bytes, message);
    if (server_to_client && bits == 32u && message % 2u == 0u)
        *kind = DS4_TRANSPORT_BULK_RESULT_LOGITS;
    else
        *kind = server_to_client
            ? DS4_TRANSPORT_BULK_RESULT_HIDDEN
            : DS4_TRANSPORT_BULK_INPUT_HIDDEN;
    *element_bits = bits;
    *session_id = server_to_client ? LIVE_SESSION_S2C : LIVE_SESSION_C2S;
}

static int live_prepare(ds4_transport *transport, int server_to_client,
                        uint64_t message, size_t bytes,
                        ds4_transport_bulk_desc *desc) {
    uint32_t kind;
    uint32_t element_bits;
    uint64_t session_id;
    live_expected_tag(server_to_client, message, bytes, &kind,
                      &element_bits, &session_id);
    char error[160];
    if (ds4_transport_prepare_bulk(transport, kind, session_id,
                                   message + 1u, (uint32_t)bytes,
                                   element_bits, desc,
                                   error, sizeof(error)) != 0)
        return live_error("prepare message %" PRIu64 ": %s (%s)",
                          message + 1u, error, strerror(errno));
    if (desc->mode != DS4_TRANSPORT_BULK_NHI_OOB)
        return live_error("message %" PRIu64
                          " unexpectedly selected inline TCP", message + 1u);
    return 0;
}

static int live_validate(ds4_transport *transport, int server_to_client,
                         uint64_t message, size_t bytes,
                         const ds4_transport_bulk_desc *desc) {
    uint32_t kind;
    uint32_t element_bits;
    uint64_t session_id;
    live_expected_tag(server_to_client, message, bytes, &kind,
                      &element_bits, &session_id);
    char error[160];
    if (ds4_transport_validate_oob_desc(
            transport, desc, kind, session_id, message + 1u,
            (uint32_t)bytes, element_bits, error, sizeof(error)) != 0)
        return live_error("validate message %" PRIu64 ": %s (%s)",
                          message + 1u, error, strerror(errno));
    return 0;
}

static int live_send_desc_and_payload(ds4_transport *transport, int fd,
                                      const ds4_transport_bulk_desc *desc,
                                      void *copy_payload, size_t bytes,
                                      uint32_t pattern_direction,
                                      int gpu_alias,
                                      live_path_counts *paths) {
    if (ds4_transport_mapped_leases_supported(transport)) {
        ds4_transport_lease *lease = NULL;
        char error[160];
        if (ds4_transport_tx_lease_acquire(transport, desc, &lease,
                                           error, sizeof(error)) == 0) {
            unsigned char *mapped = ds4_transport_lease_host_ptr(lease);
            void *device = ds4_transport_lease_device_ptr(lease);
            if (!mapped || ds4_transport_lease_bytes(lease) != bytes ||
                (gpu_alias && !device)) {
                int saved = EPROTO;
                (void)ds4_transport_lease_abort(lease);
                ds4_transport_lease_release(lease);
                errno = saved;
                return live_error("invalid mapped TX lease for message %"
                                  PRIu64, desc->request_id);
            }
            if (gpu_alias) {
#ifdef DS4_ROCM_BUILD
                if (!ds4_test_nhi_gpu_alias_fill(
                        device, bytes, pattern_direction, desc->generation,
                        desc->request_id - 1u)) {
                    int saved = EIO;
                    (void)ds4_transport_lease_abort(lease);
                    ds4_transport_lease_release(lease);
                    errno = saved;
                    return live_error("GPU fill of mapped TX message %"
                                      PRIu64 " failed", desc->request_id);
                }
                if (live_verify_pattern(mapped, bytes, pattern_direction,
                                        desc->generation,
                                        desc->request_id - 1u) != 0) {
                    int saved = errno ? errno : EILSEQ;
                    (void)ds4_transport_lease_abort(lease);
                    ds4_transport_lease_release(lease);
                    errno = saved;
                    return -1;
                }
#else
                (void)device;
                (void)ds4_transport_lease_abort(lease);
                ds4_transport_lease_release(lease);
                errno = ENOTSUP;
                return live_error("GPU alias mode requires the ROCm test binary");
#endif
            } else {
                live_fill_pattern(mapped, bytes, pattern_direction,
                                  desc->generation,
                                  desc->request_id - 1u);
            }
            if (live_send_control(fd, LIVE_CONTROL_DESC, 0, desc) != 0) {
                int saved = errno ? errno : EIO;
                /* A failed write may be partial, so it is never safe to roll
                 * the prepared descriptor back and retry it inline. */
                (void)ds4_transport_tx_lease_mark_control_sent(lease);
                (void)ds4_transport_lease_abort(lease);
                ds4_transport_lease_release(lease);
                errno = saved;
                return -1;
            }
            if (ds4_transport_tx_lease_mark_control_sent(lease) != 0) {
                int saved = errno ? errno : EIO;
                (void)ds4_transport_lease_abort(lease);
                ds4_transport_lease_release(lease);
                errno = saved;
                return live_error("mark mapped TX control for message %"
                                  PRIu64 ": %s", desc->request_id,
                                  strerror(saved));
            }
            if (ds4_transport_lease_commit(lease) != 0) {
                int saved = errno ? errno : EIO;
                ds4_transport_lease_release(lease);
                errno = saved;
                return live_error("commit mapped TX message %" PRIu64
                                  ": %s", desc->request_id,
                                  strerror(saved));
            }
            ds4_transport_lease_release(lease);
            paths->tx_mapped++;
            if (gpu_alias) paths->tx_gpu_alias++;
            return 0;
        }
        if (errno != ENOTSUP)
            return live_error("acquire mapped TX message %" PRIu64
                              ": %s (%s)", desc->request_id,
                              error, strerror(errno));
    }

    live_fill_pattern(copy_payload, bytes, pattern_direction,
                      desc->generation,
                      desc->request_id - 1u);
    if (live_send_control(fd, LIVE_CONTROL_DESC, 0, desc) != 0) return -1;
    if (ds4_transport_send_bulk_desc(transport, desc,
                                     copy_payload, bytes) != 0)
        return live_error("NHI send message %" PRIu64 ": %s",
                          desc->request_id, strerror(errno));
    paths->tx_copy++;
    return 0;
}

static int live_recv_desc_and_payload(ds4_transport *transport, int fd,
                                      int server_to_client, uint64_t message,
                                      size_t bytes, void *copy_payload,
                                      int gpu_alias,
                                      live_path_counts *paths) {
    live_control control = {0};
    if (live_recv_control(fd, &control) != 0) return -1;
    if (live_peer_error(&control) != 0) return -1;
    if (control.type != LIVE_CONTROL_DESC) {
        errno = EPROTO;
        return live_error("expected descriptor for message %" PRIu64
                          ", got control type %u", message + 1u,
                          control.type);
    }
    if (live_validate(transport, server_to_client, message, bytes,
                      &control.desc) != 0)
        return -1;

    const uint32_t pattern_direction = server_to_client
        ? LIVE_ROLE_SERVER : LIVE_ROLE_CLIENT;
    if (ds4_transport_mapped_leases_supported(transport)) {
        ds4_transport_lease *lease = NULL;
        char error[160];
        if (ds4_transport_rx_lease_acquire(transport, &control.desc, &lease,
                                           error, sizeof(error)) == 0) {
            const unsigned char *mapped =
                ds4_transport_lease_host_ptr(lease);
            const void *device = ds4_transport_lease_device_ptr(lease);
            int verify_rc;
            int verify_error = EPROTO;
            if (!mapped || ds4_transport_lease_bytes(lease) != bytes ||
                (gpu_alias && !device)) {
                verify_rc = live_error("invalid mapped RX lease for message %"
                                       PRIu64, message + 1u);
            } else if (gpu_alias) {
#ifdef DS4_ROCM_BUILD
                uint64_t first_bad = UINT64_MAX;
                const int alias_ok = ds4_test_nhi_gpu_alias_verify(
                    device, bytes, pattern_direction,
                    control.desc.generation, message, &first_bad);
                verify_rc = alias_ok
                    ? 0
                    : (first_bad == UINT64_MAX
                        ? live_error("GPU verify of mapped RX message %"
                                     PRIu64 " failed", message + 1u)
                        : live_error("GPU verify of mapped RX message %"
                                     PRIu64 " failed at byte %" PRIu64,
                                     message + 1u, first_bad));
                if (verify_rc == 0) {
                    verify_rc = live_verify_pattern(
                        mapped, bytes, pattern_direction,
                        control.desc.generation, message);
                }
                if (verify_rc != 0) verify_error = EILSEQ;
#else
                verify_rc = live_error(
                    "GPU alias mode requires the ROCm test binary");
                verify_error = ENOTSUP;
#endif
            } else {
                verify_rc = live_verify_pattern(
                    mapped, bytes, pattern_direction,
                    control.desc.generation, message);
                if (verify_rc != 0) verify_error = errno ? errno : EILSEQ;
            }
            /* Verification finishes before commit returns the exact frames
             * to the driver. Even corrupt data is reposted before failure is
             * reported so the local endpoint is not left with a live lease. */
            if (ds4_transport_lease_commit(lease) != 0) {
                int saved = errno ? errno : EIO;
                ds4_transport_lease_release(lease);
                errno = saved;
                return live_error("commit mapped RX message %" PRIu64
                                  ": %s", message + 1u,
                                  strerror(saved));
            }
            ds4_transport_lease_release(lease);
            if (verify_rc != 0) {
                errno = verify_error;
                return -1;
            }
            paths->rx_mapped++;
            if (gpu_alias) paths->rx_gpu_alias++;
            return 0;
        }
        if (errno != ENOTSUP)
            return live_error("acquire mapped RX message %" PRIu64
                              ": %s (%s)", message + 1u,
                              error, strerror(errno));
    }

    int rc = ds4_transport_recv_bulk_desc(transport, &control.desc,
                                          copy_payload, bytes);
    if (rc != 1)
        return live_error("NHI receive message %" PRIu64 ": %s",
                          message + 1u, strerror(errno));
    if (live_verify_pattern(copy_payload, bytes, pattern_direction,
                            control.desc.generation, message) != 0)
        return -1;
    paths->rx_copy++;
    return 0;
}

static int live_ready_barrier(int fd) {
    if (live_send_control(fd, LIVE_CONTROL_READY, 0, NULL) != 0) return -1;
    live_control control = {0};
    if (live_recv_control(fd, &control) != 0) return -1;
    if (live_peer_error(&control) != 0) return -1;
    if (control.type != LIVE_CONTROL_READY || control.status != 0) {
        errno = EPROTO;
        return live_error("invalid peer READY record");
    }
    return 0;
}

static void live_print_plan(const char *role, ds4_transport *transport,
                            uint32_t peer_ring, uint32_t wrap_ring,
                            const live_plan *plan) {
    double mib = (double)plan->bytes_per_direction / (1024.0 * 1024.0);
    double wraps = (double)plan->frames_per_direction / wrap_ring;
    fprintf(stderr,
            "test_nhi_live: %s ready: transport=%s generation=%" PRIu64
            " frame=%u local-ring=%u peer-ring=%u messages=%" PRIu64
            " payload/dir=%.2f MiB frames/dir=%" PRIu64
            " (%.2f ring wraps)\n",
            role, ds4_transport_name(transport),
            ds4_transport_generation(transport),
            ds4_transport_frame_size(transport),
            ds4_transport_ring_size(transport), peer_ring,
            plan->messages, mib, plan->frames_per_direction, wraps);
    fprintf(stderr, "test_nhi_live: %s mapped leases: %s\n", role,
            ds4_transport_mapped_leases_supported(transport)
                ? "available (used when payload span is contiguous)"
                : "unavailable (CPU-copy path only)");
}

static int live_run_server(ds4_transport *transport, int fd,
                           const live_options *options,
                           const live_plan *plan, unsigned char *tx,
                           unsigned char *rx, uint64_t *elapsed_ns,
                           live_path_counts *paths) {
    uint64_t start = live_now_ns();
    for (uint64_t message = 0; message < plan->messages; message++) {
        size_t bytes = plan->sizes[message % plan->size_count];
        if (live_recv_desc_and_payload(transport, fd, 0, message,
                                       bytes, rx,
                                       options->require_gpu_alias,
                                       paths) != 0) {
            int saved = errno ? errno : EILSEQ;
            live_send_error_best_effort(fd, saved);
            errno = saved;
            return -1;
        }
        ds4_transport_bulk_desc reply;
        if (live_prepare(transport, 1, message, bytes, &reply) != 0 ||
            live_send_desc_and_payload(transport, fd, &reply,
                                       tx, bytes, LIVE_ROLE_SERVER,
                                       options->require_gpu_alias,
                                       paths) != 0) {
            int saved = errno ? errno : EIO;
            live_send_error_best_effort(fd, saved);
            errno = saved;
            return -1;
        }
    }

    if (live_validate_paths(options, plan, paths) != 0) {
        int saved = errno ? errno : EPROTO;
        live_send_error_best_effort(fd, saved);
        errno = saved;
        return -1;
    }
    live_control control = {0};
    if (live_recv_control(fd, &control) != 0) return -1;
    if (live_peer_error(&control) != 0) return -1;
    if (control.type != LIVE_CONTROL_DONE || control.status != 0) {
        errno = EPROTO;
        return live_error("expected peer DONE record");
    }
    if (live_send_control(fd, LIVE_CONTROL_DONE, 0, NULL) != 0) return -1;
    uint64_t end = live_now_ns();
    *elapsed_ns = end >= start ? end - start : 0;
    return 0;
}

static int live_run_client(ds4_transport *transport, int fd,
                           const live_options *options,
                           const live_plan *plan, unsigned char *tx,
                           unsigned char *rx, uint64_t *elapsed_ns,
                           uint64_t *rtt_min_ns, uint64_t *rtt_max_ns,
                           uint64_t *rtt_sum_ns,
                           live_path_counts *paths) {
    *rtt_min_ns = UINT64_MAX;
    *rtt_max_ns = 0;
    *rtt_sum_ns = 0;
    uint64_t start = live_now_ns();
    for (uint64_t message = 0; message < plan->messages; message++) {
        size_t bytes = plan->sizes[message % plan->size_count];
        uint64_t rtt_start = live_now_ns();
        ds4_transport_bulk_desc request;
        if (live_prepare(transport, 0, message, bytes, &request) != 0 ||
            live_send_desc_and_payload(transport, fd, &request,
                                       tx, bytes, LIVE_ROLE_CLIENT,
                                       options->require_gpu_alias,
                                       paths) != 0 ||
            live_recv_desc_and_payload(transport, fd, 1, message,
                                       bytes, rx,
                                       options->require_gpu_alias,
                                       paths) != 0)
            return -1;
        uint64_t rtt_end = live_now_ns();
        uint64_t rtt = rtt_end >= rtt_start ? rtt_end - rtt_start : 0;
        if (rtt < *rtt_min_ns) *rtt_min_ns = rtt;
        if (rtt > *rtt_max_ns) *rtt_max_ns = rtt;
        if (UINT64_MAX - *rtt_sum_ns < rtt)
            return live_error("RTT accumulator overflow");
        *rtt_sum_ns += rtt;
    }
    if (live_validate_paths(options, plan, paths) != 0) {
        int saved = errno ? errno : EPROTO;
        live_send_error_best_effort(fd, saved);
        errno = saved;
        return -1;
    }
    if (live_send_control(fd, LIVE_CONTROL_DONE, 0, NULL) != 0) return -1;
    live_control control = {0};
    if (live_recv_control(fd, &control) != 0) return -1;
    if (live_peer_error(&control) != 0) return -1;
    if (control.type != LIVE_CONTROL_DONE || control.status != 0) {
        errno = EPROTO;
        return live_error("invalid peer DONE record");
    }
    uint64_t end = live_now_ns();
    *elapsed_ns = end >= start ? end - start : 0;
    return 0;
}

static void live_print_result(const char *role, const live_plan *plan,
                              uint32_t wrap_ring, uint64_t elapsed_ns,
                              uint64_t rtt_min_ns, uint64_t rtt_max_ns,
                              uint64_t rtt_sum_ns,
                              const live_path_counts *paths) {
    const double seconds = (double)elapsed_ns / 1000000000.0;
    const double total_mib = (double)plan->bytes_per_direction * 2.0 /
        (1024.0 * 1024.0);
    const double mib_per_sec = seconds > 0.0 ? total_mib / seconds : 0.0;
    const double wraps = (double)plan->frames_per_direction / wrap_ring;
    printf("test_nhi_live: PASS %s messages=%" PRIu64
           " wraps/dir=%.2f verified=%.2f MiB elapsed=%.3fs"
           " throughput=%.2f MiB/s",
           role, plan->messages, wraps, total_mib, seconds, mib_per_sec);
    if (rtt_min_ns != UINT64_MAX) {
        const double average_us = (double)rtt_sum_ns /
            (double)plan->messages / 1000.0;
        printf(" verified-rtt-us avg=%.2f min=%.2f max=%.2f",
               average_us, (double)rtt_min_ns / 1000.0,
               (double)rtt_max_ns / 1000.0);
    }
    printf(" paths=tx(mapped=%" PRIu64 ",copy=%" PRIu64
           ",gpu-alias=%" PRIu64 ")/rx(mapped=%" PRIu64
           ",copy=%" PRIu64 ",gpu-alias=%" PRIu64 ")\n",
           paths->tx_mapped, paths->tx_copy,
           paths->tx_gpu_alias, paths->rx_mapped, paths->rx_copy,
           paths->rx_gpu_alias);
}

static int live_validate_paths(const live_options *options,
                               const live_plan *plan,
                               const live_path_counts *paths) {
    if (paths->tx_mapped + paths->tx_copy != plan->messages ||
        paths->rx_mapped + paths->rx_copy != plan->messages)
        return live_error("internal path accounting mismatch");
    if (options->require_mapped &&
        (paths->tx_mapped == 0 || paths->rx_mapped == 0)) {
        errno = ENOTSUP;
        return live_error("required mapped TX/RX paths were not exercised");
    }
    if (options->require_copy &&
        (paths->tx_copy == 0 || paths->rx_copy == 0)) {
        errno = ENOTSUP;
        return live_error("required CPU-copy TX/RX paths were not exercised");
    }
    if (options->require_gpu_alias &&
        (paths->tx_gpu_alias == 0 || paths->rx_gpu_alias == 0 ||
         paths->tx_gpu_alias != paths->tx_mapped ||
         paths->rx_gpu_alias != paths->rx_mapped)) {
        errno = ENOTSUP;
        return live_error("required GPU-alias TX/RX paths were not exercised");
    }
    return 0;
}

int main(int argc, char **argv) {
    live_options options;
    int parse_rc = live_parse_options(argc, argv, &options);
    if (parse_rc > 0) return 0;
    if (parse_rc < 0) return 2;

#ifndef __linux__
    live_error("the NHI live test requires Linux");
    return 77;
#endif
#ifndef DS4_ROCM_BUILD
    if (options.require_gpu_alias) {
        live_error("--require-gpu-alias requires the ROCm test binary");
        return 77;
    }
#endif
    char timeout_text[16];
    snprintf(timeout_text, sizeof(timeout_text), "%u", options.timeout_sec);
    if (setenv("DS4_DIST_NHI_TIMEOUT_SEC", timeout_text, 1) != 0) {
        live_error("set NHI timeout: %s", strerror(errno));
        return 1;
    }

    int fd = options.server
        ? live_listen(options.port, options.timeout_sec)
        : live_connect(options.host, options.port, options.timeout_sec);
    if (fd < 0) return 1;

    int result = 1;
    ds4_transport *transport = NULL;
    unsigned char *tx = NULL;
    unsigned char *rx = NULL;
    char error[256];
    transport = ds4_transport_nhi_create(fd, options.device,
                                         error, sizeof(error));
    if (!transport) {
        live_error("create NHI transport on %s: %s (%s)",
                   options.device, error, strerror(errno));
        goto cleanup;
    }
    if (strcmp(ds4_transport_name(transport), "nhi-cpu-copy") != 0 ||
        (ds4_transport_caps(transport) & DS4_TRANSPORT_CAP_ZEROCOPY) == 0) {
        live_error("unexpected transport backend: %s",
                   ds4_transport_name(transport));
        goto cleanup;
    }

    live_hello local;
    local.role = options.server ? LIVE_ROLE_SERVER : LIVE_ROLE_CLIENT;
    local.frame_size = ds4_transport_frame_size(transport);
    local.ring_size = ds4_transport_ring_size(transport);
    local.flags = options.require_gpu_alias ? LIVE_HELLO_F_GPU_ALIAS : 0u;
    local.generation = options.server ? live_generation() : 0;
    live_hello peer = {0};
    if (live_send_hello(fd, &local) != 0 ||
        live_recv_hello(fd, &peer) != 0)
        goto cleanup;
    if (peer.role == local.role ||
        (peer.role != LIVE_ROLE_SERVER && peer.role != LIVE_ROLE_CLIENT) ||
        peer.frame_size != local.frame_size ||
        peer.ring_size != local.ring_size || peer.ring_size < 2u ||
        peer.flags != local.flags ||
        (options.server ? peer.generation != 0 : peer.generation == 0)) {
        live_error("incompatible peer HELLO geometry or role");
        live_send_error_best_effort(fd, EPROTO);
        goto cleanup;
    }
    uint64_t generation = options.server
        ? local.generation : peer.generation;
    if (ds4_transport_configure_link(transport, generation,
                                     peer.frame_size, peer.ring_size,
                                     error, sizeof(error)) != 0) {
        live_error("configure NHI link: %s (%s)", error, strerror(errno));
        live_send_error_best_effort(fd, errno);
        goto cleanup;
    }
    if (options.require_mapped &&
        !ds4_transport_mapped_leases_supported(transport)) {
        live_error("mapped leases are required but unavailable");
        live_send_error_best_effort(fd, ENOTSUP);
        goto cleanup;
    }
    if (live_ready_barrier(fd) != 0) goto cleanup;

    /* This hardware gate requires equal physical geometry, so two traversals
     * here are literal wraps of both endpoints rather than only of a smaller
     * negotiated credit window. */
    uint32_t wrap_ring = local.ring_size;
    live_plan plan;
    if (live_build_plan(local.frame_size, wrap_ring,
                        ds4_transport_max_oob_bytes(transport), &plan) != 0) {
        live_send_error_best_effort(fd, errno);
        goto cleanup;
    }
    live_print_plan(options.server ? "server" : "client", transport,
                    peer.ring_size, wrap_ring, &plan);

    tx = malloc(plan.max_payload);
    rx = malloc(plan.max_payload);
    if (!tx || !rx) {
        live_error("allocate %zu-byte verification buffers: %s",
                   plan.max_payload, strerror(errno));
        live_send_error_best_effort(fd, ENOMEM);
        goto cleanup;
    }
    uint64_t elapsed_ns = 0;
    uint64_t rtt_min_ns = UINT64_MAX;
    uint64_t rtt_max_ns = 0;
    uint64_t rtt_sum_ns = 0;
    live_path_counts paths;
    memset(&paths, 0, sizeof(paths));
    int run_rc = options.server
        ? live_run_server(transport, fd, &options, &plan, tx, rx, &elapsed_ns,
                          &paths)
        : live_run_client(transport, fd, &options, &plan, tx, rx, &elapsed_ns,
                          &rtt_min_ns, &rtt_max_ns, &rtt_sum_ns,
                          &paths);
    if (run_rc != 0) goto cleanup;
    live_print_result(options.server ? "server" : "client", &plan,
                      wrap_ring, elapsed_ns, rtt_min_ns,
                      rtt_max_ns, rtt_sum_ns, &paths);
    result = 0;

cleanup:
    free(rx);
    free(tx);
    ds4_transport_release(transport);
    (void)shutdown(fd, SHUT_RDWR);
    close(fd);
    return result;
}
