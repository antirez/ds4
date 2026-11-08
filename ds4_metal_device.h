#ifndef DS4_METAL_DEVICE_H
#define DS4_METAL_DEVICE_H

#include <string.h>

/* Share the Qwen M1 Max defaults with M2-M4 while keeping other M1 devices
 * and newer generations on their existing paths. Bound the generation token
 * so names such as M20 do not inherit these defaults. */
static inline int ds4_metal_device_name_uses_qwen_legacy_tuning(const char *name) {
    return name && (strcmp(name, "Apple M1 Max") == 0 ||
                   (strncmp(name, "Apple M", 7) == 0 &&
                    name[7] >= '2' && name[7] <= '4' &&
                    (name[8] == '\0' || name[8] == ' ')));
}

/* M3 Ultra already has dedicated MoE defaults; preserve those selections. */
static inline int ds4_metal_device_name_uses_qwen_legacy_moe_tuning(const char *name) {
    return ds4_metal_device_name_uses_qwen_legacy_tuning(name) &&
           strcmp(name, "Apple M3 Ultra") != 0;
}

/* Share the M5 tuning policy with M6. Keep the generation token bounded so
 * untested future names such as M50 and M60 do not inherit these defaults.
 * Callers must still check runtime and kernel capabilities where required. */
static inline int ds4_metal_device_name_is_m5_or_m6(const char *name) {
    return name && strncmp(name, "Apple M", 7) == 0 &&
           (name[7] == '5' || name[7] == '6') &&
           (name[8] == '\0' || name[8] == ' ');
}

#endif
