#ifndef DS4_VULKAN_H
#define DS4_VULKAN_H

#include <stdint.h>
#include <stdbool.h>

/* =========================================================================
 * Vulkan Backend for DS4 — Strix Halo (Radeon 8060S / gfx1151)
 * =========================================================================
 *
 * This backend replaces the ROCm/HIP implementation with Vulkan compute
 * shaders.  It uses volk (dynamic Vulkan loader) and VMA (Vulkan Memory
 * Allocator) for portability.  All compute kernels are written in GLSL 460
 * and compiled to SPIR-V at build time via glslangValidator.
 *
 * Build target:  make vulkan
 * Architecture:  Vulkan 1.3 + VK_KHR_* extensions (shader_float16, subgroup,
 *                16bit_storage, shader_atomic_int64, shader_int8)
 */

/* -----------------------------------------------------------------------
 * Forward declarations
 * ----------------------------------------------------------------------- */
struct ds4_vulkan_device;
struct ds4_vulkan_tensor;
struct ds4_vulkan_shader;
struct ds4_vulkan_shader_cache;
struct ds4_vulkan_pipeline_cache;
struct ds4_vulkan_descriptor_cache;
struct ds4_vulkan_submission;

/* -----------------------------------------------------------------------
 * Backend capability flags set during ds4_gpu_init()
 * ----------------------------------------------------------------------- */
typedef struct ds4_vulkan_caps {
    uint32_t subgroup_size;                /* e.g. 64 or 32 */
    uint32_t min_uniform_offset;
    uint32_t max_push_constants_size;      /* bytes */
    uint32_t max_compute_work_group_invocations;
    uint32_t max_shared_memory_size;       /* bytes */
    uint64_t device_memory_total;          /* bytes */
    uint64_t device_memory_available;      /* bytes at init */
    bool     has_float16;                  /* VK_KHR_shader_float16_int8 */
    bool     has_int8;                     /* VK_KHR_shader_int8 */
    bool     has_16bit_storage;            /* VK_KHR_16bit_storage */
    bool     has_subgroup_basic;
    bool     has_subgroup_arithmetic;
    bool     has_subgroup_ballot;
    bool     has_subgroup_shuffle;
    bool     has_atomic_int64;
    bool     has_timeline_semaphore;       /* VK_KHR_timeline_semaphore */
    bool     has_host_query_reset;         /* VK_EXT_host_query_reset */
} ds4_vulkan_caps;

/* -----------------------------------------------------------------------
 * Command submission tracking  (timeline semaphore based)
 * -----------------------------------------------------------------------
 *
 * Each submitted batch of work gets a monotonically increasing timeline
 * value.  The submission record lets us synchronize readback and tensor
 * lifecycle with precise GPU-side ordering.
 */
typedef struct ds4_vulkan_submission {
    uint64_t timeline_value;   /* value on the timeline semaphore */
    uint64_t fence_value;      /* value on the binary-fence ring slot */
} ds4_vulkan_submission;

/* -----------------------------------------------------------------------
 * Tensor backend extension  (internal, used by ds4_gpu_tensor owners)
 * -----------------------------------------------------------------------
 *
 * The C struct ds4_gpu_tensor (from ds4_gpu.h) holds {ptr, bytes, owner}.
 * Our C++ wrapper attaches a Vulkan buffer + allocation.  Tensor views
 * share the parent's allocation and record a buffer offset.
 */
typedef struct ds4_vulkan_tensor {
    /* ds4_gpu_tensor fields — kept in sync with the opaque C type */
    void     *ptr;             /* host-mapped pointer (if host-visible) */
    uint64_t  bytes;           /* total allocation size */
    int       owner;           /* 1 = this struct owns the allocation */

    /* Vulkan buffer state */
    VkBuffer              buffer;        /* VkBuffer handle */
    VmaAllocation         allocation;    /* VMA allocation (NULL for views) */
    VmaAllocationInfo     alloc_info;    /* allocation details */
    uint64_t              buffer_offset; /* byte offset into VkBuffer (views) */

    /* Lifecycle tracking */
    uint64_t              last_submit_value;  /* timeline value of last use */
    int                   is_managed;         /* 1 = managed memory */
    int                   is_host_visible;    /* 1 = HOST_VISIBLE | HOST_COHERENT */
    int                   is_mapped;          /* 1 = persistently mapped */
} ds4_vulkan_tensor;

/* -----------------------------------------------------------------------
 * Shader and pipeline cache entry
 * -----------------------------------------------------------------------
 */
typedef struct ds4_vulkan_shader {
    const char     *name;            /* human-readable shader name */
    VkShaderModule  module;          /* compiled SPIR-V module */
    VkPipelineLayout layout;         /* pipeline layout */
    VkPipeline      pipeline;        /* compute pipeline */
    uint32_t        push_constant_size; /* bytes */
    uint32_t        descriptor_set_count;
    uint32_t        ref_count;       /* for cache eviction */
} ds4_vulkan_shader;

/* -----------------------------------------------------------------------
 * Descriptor cache — pools and set layouts for compute dispatch
 * -----------------------------------------------------------------------
 *
 * The backend pre-allocates a descriptor pool large enough for the
 * max number of tensor bindings per dispatch.  Cached set layouts
 * are shared across pipelines with identical binding patterns.
 */
typedef struct ds4_vulkan_descriptor_cache {
    VkDescriptorPool         pool;
    VkDescriptorSetLayout    storage_buffer_layout;   /* single SSBO */
    VkDescriptorSetLayout    multi_storage_layout;    /* N SSBOs (max 16) */
    uint32_t                 max_sets;
    uint32_t                 allocated_sets;
} ds4_vulkan_descriptor_cache;

/* -----------------------------------------------------------------------
 * Expert cache state  (for MoE streaming SSD offload)
 * -----------------------------------------------------------------------
 *
 * Mirrors the ROCm backend's stream_expert_cache but uses staging
 * buffers and copy commands instead of HIP direct-mapped memory.
 */
typedef struct ds4_vulkan_expert_cache {
    /* Configured budget */
    uint32_t max_experts;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;

    /* Currently resident expert tensors (device-side) */
    ds4_vulkan_tensor *gate_buffer;
    ds4_vulkan_tensor *down_buffer;
    uint32_t           current_count;

    /* Layer load tracking */
    uint32_t current_layer;
    uint64_t last_load_timeline;

    /* Staging transfer state */
    VkBuffer     staging_buffer;
    VmaAllocation staging_allocation;
    uint64_t     staging_bytes;
} ds4_vulkan_expert_cache;

/* -----------------------------------------------------------------------
 * Transfer queue state  (for async host<->device copies)
 * -----------------------------------------------------------------------
 */
typedef struct ds4_vulkan_transfer_state {
    VkCommandPool    pool;
    VkCommandBuffer  cmd;
    VkFence          fence;
    int              has_dedicated_queue;

    /* Queue family info */
    uint32_t         queue_family;
    VkQueue          queue;
} ds4_vulkan_transfer_state;

/* -----------------------------------------------------------------------
 * Main Vulkan device singleton
 * -----------------------------------------------------------------------
 */
typedef struct ds4_vulkan_device {
    /* Instance / Device */
    VkInstance            instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkPhysicalDevice      physical_device;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    VkPhysicalDeviceSubgroupProperties subgroup_properties;
    VkDevice              device;

    /* Queues */
    uint32_t              compute_queue_family;
    VkQueue               compute_queue;

    /* Capabilities */
    ds4_vulkan_caps       caps;

    /* VMA allocator */
    VmaAllocator          allocator;

    /* Compute command buffer ring (double-buffered) */
    VkCommandPool         cmd_pool;
    VkCommandBuffer       cmd_buffers[2];
    VkFence               cmd_fences[2];
    int                   cmd_ring_index;
    uint64_t              cmd_ring_fence_value[2];

    /* Synchronization */
    VkSemaphore           timeline_semaphore;
    uint64_t              timeline_value;        /* monotonically increasing */
    VkSemaphore           submit_semaphore;      /* binary for submission order */

    /* Timeline semaphore import/export for multi-queue ordering */
    int                   has_timeline_semaphore;

    /* Shader / Pipeline cache */
    struct ds4_vulkan_shader_cache   *shader_cache;
    struct ds4_vulkan_pipeline_cache *pipeline_cache;
    ds4_vulkan_descriptor_cache       descriptor_cache;

    /* Descriptor set ring (pre-allocated, recycled per submission) */
    VkDescriptorSet       descriptor_sets[64];
    uint32_t              descriptor_set_count;
    uint32_t              descriptor_set_next;

    /* Transfer state */
    ds4_vulkan_transfer_state transfer;

    /* Expert cache */
    ds4_vulkan_expert_cache expert_cache;

    /* Model memory (mapped from file/pipe) */
    const void           *model_map;
    uint64_t              model_size;
    uint64_t              model_map_offset;
    VkDeviceMemory        model_device_memory;
    VkBuffer              model_buffer;
    int                   model_is_mapped;

    /* Staging buffer for model weight uploads */
    VkBuffer              staging_buffer;
    VmaAllocation         staging_allocation;
    uint64_t              staging_size;

    /* GPU name / driver info */
    char                  gpu_name[256];
    char                  driver_version[64];

    /* Memory pressure tracking */
    uint64_t              allocated_bytes;
    uint64_t              peak_allocated_bytes;
    int                   quality_mode;          /* true = high quality */
    int                   ssd_streaming_enabled;

    /* Error state */
    int                   init_done;
    int                   last_vk_result;
} ds4_vulkan_device;

/* -----------------------------------------------------------------------
 * Public C API  (declared for C callers; implemented in ds4_vulkan.cpp)
 * ----------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/* GPU name reported by ds4_gpu_init */
extern const char *ds4_vulkan_gpu_name;
extern const char *ds4_vulkan_driver_version;

/* Per-backend state query (used by ds4_gpu_init / ds4_gpu_print_memory_report) */
void ds4_vulkan_get_caps(ds4_vulkan_caps *caps);
int  ds4_vulkan_init(void);
void ds4_vulkan_cleanup(void);

/* Shader compilation helper (called during init for each kernel) */
int  ds4_vulkan_compile_shaders(void);

/* Internal device accessor (for .cpp implementation) */
struct ds4_vulkan_device *ds4_vulkan_get_device(void);

/* Tensor allocation helpers */
struct ds4_vulkan_tensor *ds4_vulkan_tensor_from_gpu(struct ds4_gpu_tensor *t);
struct ds4_gpu_tensor    *ds4_vulkan_tensor_to_gpu(struct ds4_vulkan_tensor *vt);

/* Debug / reporting */
void ds4_vulkan_print_memory_report(const char *label);
void ds4_vulkan_report_caps(void);

#ifdef __cplusplus
}
#endif

/* -----------------------------------------------------------------------
 * C++ internal helpers  (inline utilities used by .cpp implementation)
 * ----------------------------------------------------------------------- */

#ifdef __cplusplus

/* Convenience: convert between ds4_gpu_tensor* and ds4_vulkan_tensor*.
 *
 * The tensor's owner flag determines whether the pointer is to a
 * ds4_gpu_tensor (owner=0) or a ds4_vulkan_tensor (owner=1).  When a
 * tensor is allocated by the Vulkan backend it is always a full
 * ds4_vulkan_tensor; tensors created by the C graph driver as stack
 * temporaries use ds4_gpu_tensor and are promoted at allocation time.
 */
static inline ds4_vulkan_tensor *vk_tensor(ds4_gpu_tensor *t) {
    return reinterpret_cast<ds4_vulkan_tensor *>(t);
}

static inline const ds4_vulkan_tensor *vk_tensor_c(const ds4_gpu_tensor *t) {
    return reinterpret_cast<const ds4_vulkan_tensor *>(t);
}

/* Descriptor write helper: bind one storage buffer to a descriptor set */
static inline VkDescriptorBufferInfo vk_buffer_info(
    VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range)
{
    VkDescriptorBufferInfo info = {};
    info.buffer = buffer;
    info.offset = offset;
    info.range  = range;
    return info;
}

/* Push constant range helper */
static inline VkPushConstantRange vk_push_range(
    VkShaderStageFlags stages, uint32_t offset, uint32_t size)
{
    VkPushConstantRange range = {};
    range.stageFlags = stages;
    range.offset     = offset;
    range.size       = size;
    return range;
}

#endif /* __cplusplus */

#endif /* DS4_VULKAN_H */
