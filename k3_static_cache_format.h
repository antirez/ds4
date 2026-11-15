#ifndef K3_STATIC_CACHE_FORMAT_H
#define K3_STATIC_CACHE_FORMAT_H
#include <stdint.h>
#define K3SC_MAGIC "K3SC001"
#define K3SC_VERSION 1u
#define K3SC_ALIGNMENT 16384u
#define K3SC_BLOCK_BYTES (16u * 1024u * 1024u)
#define K3SC_MAX_LAYERS 256u
#define K3SC_MAX_BLOCKS 16384u
#define K3SC_RAW 0u
#define K3SC_BF16_PLANES_ZSTD 1u
/* Version1 is little-endian, fixed sizes, CRC32 IEEE (0xedb88320) with
 * initial/final xor 0xffffffff. Header CRC zeros header_crc32, includes table
 * CRC. All reserved fields must be zero. Data padding is zero on build. */
typedef struct {
    uint8_t magic[8];
    uint32_t version, header_bytes;
    uint64_t model_dev, model_ino, model_bytes;
    int64_t model_mtime_sec;
    uint64_t model_mtime_nsec;
    int64_t model_ctime_sec;
    uint64_t model_ctime_nsec;
    uint32_t layer_count, block_count, block_bytes, alignment;
    uint64_t table_bytes, data_offset, file_bytes;
    uint32_t header_crc32, table_crc32;
    uint64_t reserved;
} k3sc_header;
typedef struct {
    uint32_t layer, first_block, block_count, reserved;
    uint64_t offset, bytes;
} k3sc_layer;
typedef struct {
    uint64_t file_offset, decoded_offset;
    uint32_t stored_bytes, padded_bytes, decoded_bytes, crc32;
    uint32_t codec, reserved;
} k3sc_block;
_Static_assert(sizeof(k3sc_header) == 128, "cache header ABI");
_Static_assert(sizeof(k3sc_layer) == 32, "cache layer ABI");
_Static_assert(sizeof(k3sc_block) == 40, "cache block ABI");
#endif
