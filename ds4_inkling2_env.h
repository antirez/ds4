/* Shared parsing of the Inkling 2 environment switches.  Used by ds4.c (host code) and ds4_metal.m (kernel choice) so
 * both always agree on what a variable means. */
#ifndef DS4_INKLING2_ENV_H
#define DS4_INKLING2_ENV_H
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* A boolean switch is ON for any value except unset, empty or blank, "0", "false", "off" and "no" (any case, surrounding blanks
 * ignored). */
static inline int ds4_inkling2_env_flag(const char *name) {
    const char *e = getenv(name);
    if (!e) return 0;
    while (*e == ' ' || *e == '\t') e++;
    size_t n = strlen(e);
    while (n > 0 && (e[n - 1] == ' ' || e[n - 1] == '\t' || e[n - 1] == '\n')) n--;
    if (n == 0) return 0;
    static const char *const off[] = {"0", "false", "off", "no"};
    for (size_t i = 0; i < sizeof(off) / sizeof(off[0]); i++)
        if (strlen(off[i]) == n && strncasecmp(e, off[i], n) == 0) return 0;
    return 1;
}

/* DS4_INKLING2_EXACT: only the fixed-order kernels (a diagnostic mode; results do not depend on the token count of a dispatch). */
static inline int ds4_inkling2_exact_mode(void) { return ds4_inkling2_env_flag("DS4_INKLING2_EXACT"); }

/* Start check for the memory an Inkling server needs (sessions plus the mapped model).  need, ram and budget are bytes; free_pct is
 * the system's free percentage right now (kern.memorystatus_level, the number memory_pressure prints), negative when unknown.
 * Returns 0 when it fits, 1 when it exceeds the fixed budget, 2 when starting would leave less than floor_pct percent of RAM free
 * (the machine also runs other jobs; the floor is the safety margin). */
static inline int ds4_inkling2_memory_verdict(double need, double ram, double free_pct, double floor_pct, double budget) {
    if (budget > 0.0 && need > budget) return 1;
    if (ram > 0.0 && free_pct >= 0.0 && free_pct / 100.0 * ram - need < floor_pct / 100.0 * ram) return 2;
    return 0;
}

#endif
