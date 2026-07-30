#pragma once

#include <pthread.h>

/* RAII scoped lock for a pthread_mutex_t (C++ port). Locks on construction,
 * unlocks on scope exit — so an early return cannot leak the lock (the
 * missed-unlock-on-early-return deadlock). Replaces the hand-paired
 *
 *     pthread_mutex_lock(&m);
 *     ... if (x) { pthread_mutex_unlock(&m); return; } ...
 *     pthread_mutex_unlock(&m);
 *
 * idiom, which is correct today but one forgotten unlock away from a hang.
 *
 * ONLY valid where the lock brackets a clean lexical scope: the mutex must be
 * held for the whole remaining scope. Do NOT use it where the original
 * deliberately unlocks EARLY to do work outside the critical section, or where
 * the same mutex is unlocked and re-locked mid-scope — a scope-end guard would
 * wrongly widen the critical section. Compatible with pthread_cond_wait, which
 * atomically releases and re-acquires the held mutex.
 *
 * Non-copyable, non-movable: the lock lifetime IS the enclosing scope. */

namespace pulsar {

class ScopedLock {
public:
    explicit ScopedLock(pthread_mutex_t *m) : m_(m) { pthread_mutex_lock(m_); }
    ~ScopedLock() { pthread_mutex_unlock(m_); }

    ScopedLock(const ScopedLock &) = delete;
    ScopedLock &operator=(const ScopedLock &) = delete;

private:
    pthread_mutex_t *m_;
};

} // namespace pulsar
