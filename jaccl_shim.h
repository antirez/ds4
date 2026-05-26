#ifndef JACCL_SHIM_H
#define JACCL_SHIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *jaccl_group_t;

/* Dtype enum matching jaccl::Dtype (group.h). */
enum jaccl_dtype {
    JACCL_FLOAT32 = 11, /* jaccl::Dtype::Float32 */
    JACCL_FLOAT16 = 9,  /* jaccl::Dtype::Float16 */
};

bool          jaccl_is_available(void);
jaccl_group_t jaccl_init_from_env(bool strict);
void          jaccl_group_free(jaccl_group_t g);
int           jaccl_group_rank(jaccl_group_t g);
int           jaccl_group_size(jaccl_group_t g);
void          jaccl_group_all_sum(jaccl_group_t g, const void *in, void *out,
                                  size_t n_bytes, int dtype);
void          jaccl_group_barrier(jaccl_group_t g);
void          jaccl_group_send(jaccl_group_t g, const void *buf,
                               size_t n_bytes, int dst);
void          jaccl_group_recv(jaccl_group_t g, void *buf, size_t n_bytes,
                               int src);

#ifdef __cplusplus
}
#endif

#endif
