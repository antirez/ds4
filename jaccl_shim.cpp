#include "jaccl_shim.h"
#include <jaccl/jaccl.h>
#include <memory>

static std::shared_ptr<jaccl::Group> unwrap(jaccl_group_t g) {
    return *reinterpret_cast<std::shared_ptr<jaccl::Group>*>(g);
}

extern "C" {

bool jaccl_is_available(void) {
    return jaccl::is_available();
}

jaccl_group_t jaccl_init_from_env(bool strict) {
    auto group = jaccl::init(strict);
    if (!group) return nullptr;
    auto *p = new std::shared_ptr<jaccl::Group>(std::move(group));
    return reinterpret_cast<jaccl_group_t>(p);
}

void jaccl_group_free(jaccl_group_t g) {
    if (!g) return;
    delete reinterpret_cast<std::shared_ptr<jaccl::Group>*>(g);
}

int jaccl_group_rank(jaccl_group_t g) { return unwrap(g)->rank(); }
int jaccl_group_size(jaccl_group_t g) { return unwrap(g)->size(); }

void jaccl_group_all_sum(jaccl_group_t g, const void *in, void *out,
                         size_t n_bytes, int dtype) {
    unwrap(g)->all_sum(in, out, n_bytes, dtype);
}

void jaccl_group_barrier(jaccl_group_t g) { unwrap(g)->barrier(); }

void jaccl_group_send(jaccl_group_t g, const void *buf, size_t n_bytes,
                      int dst) {
    unwrap(g)->send(buf, n_bytes, dst);
}

void jaccl_group_recv(jaccl_group_t g, void *buf, size_t n_bytes, int src) {
    unwrap(g)->recv(buf, n_bytes, src);
}

} /* extern "C" */
