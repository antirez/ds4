#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <mach/mach.h>

#include "ds4.h"
#include "ds4_gpu.h"

/*
 * Objective-C Metal glue for the C engine.
 *
 * The C code owns model semantics and graph scheduling.  This file owns only
 * Metal objects: device/queue/library setup, mmap-backed weight views, command
 * batching, persistent tensors, scratch buffers, and thin wrappers around the
 * kernel files in the metal directory.  Keeping this boundary narrow makes the
 * inference path readable from C while still using Objective-C where Metal
 * requires it.
 */

#include "metal/runtime.inc"
#include "metal/embedding.inc"
#include "metal/model_io.inc"
#include "metal/expert_streaming.inc"
#include "models/deepseek/metal/host/indexer.inc"
#include "metal/dense_norm.inc"
#include "models/deepseek/metal/host/attention.inc"
#include "metal/elementwise.inc"
#include "metal/moe_dispatch.inc"
#include "models/glm/metal/host/kernels.inc"
#include "models/deepseek/metal/host/moe.inc"
#include "models/deepseek/metal/host/hc.inc"
#include "metal/compat.inc"
