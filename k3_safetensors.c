#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "k3_safetensors.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef enum {
    JSON_OBJECT,
    JSON_ARRAY,
    JSON_STRING,
    JSON_PRIMITIVE,
} json_type;

typedef struct {
    json_type type;
    int start;
    int end;
    int parent;
} json_token;

typedef struct {
    json_token *tokens;
    int count;
    int cap;
    const char *text;
    int text_len;
} json_doc;

typedef struct {
    const char *key;
    uint32_t value;
} hash_slot;

struct k3_st_shard {
    char *path;
    int fd;
    bool gguf;
    uint64_t file_size;
    uint64_t data_offset;
    void *map;
    k3_st_tensor *tensors;
    uint32_t tensor_count;
    hash_slot *hash;
    uint32_t hash_cap;
};

static void set_error(char *err, size_t cap, const char *fmt, ...) {
    if (!err || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

static void *grow_array(void *old, size_t count, size_t elem) {
    if (elem != 0 && count > SIZE_MAX / elem) return NULL;
    return realloc(old, count * elem);
}

static char *dup_slice(const char *s, size_t n) {
    if (n == SIZE_MAX) return NULL;
    char *out = malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static uint64_t load_u64_le(const uint8_t bytes[8]) {
    uint64_t value = 0;
    for (uint32_t i = 0; i < 8; i++) {
        value |= (uint64_t)bytes[i] << (8u * i);
    }
    return value;
}

static uint32_t load_u32_le(const uint8_t bytes[4]) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < 4; i++) {
        value |= (uint32_t)bytes[i] << (8u * i);
    }
    return value;
}

static int pread_exact(int fd,
                       void *buf,
                       size_t size,
                       uint64_t offset,
                       char *err,
                       size_t errcap) {
    if (offset > (uint64_t)LLONG_MAX) {
        set_error(err, errcap, "file offset is too large");
        return 0;
    }
    uint8_t *dst = buf;
    size_t done = 0;
    while (done < size) {
        ssize_t n;
        do {
            n = pread(fd, dst + done, size - done, (off_t)(offset + done));
        } while (n < 0 && errno == EINTR);
        if (n < 0) {
            set_error(err, errcap, "pread: %s", strerror(errno));
            return 0;
        }
        if (n == 0) {
            set_error(err, errcap, "unexpected end of file");
            return 0;
        }
        done += (size_t)n;
    }
    return 1;
}

static int json_add(json_doc *doc,
                    json_type type,
                    int start,
                    int end,
                    int parent) {
    if (doc->count == doc->cap) {
        int cap = doc->cap ? doc->cap * 2 : 4096;
        json_token *tokens = grow_array(doc->tokens,
                                        (size_t)cap,
                                        sizeof(tokens[0]));
        if (!tokens) return -1;
        doc->tokens = tokens;
        doc->cap = cap;
    }
    const int index = doc->count++;
    doc->tokens[index] = (json_token){
        .type = type,
        .start = start,
        .end = end,
        .parent = parent,
    };
    return index;
}

static int json_parse(json_doc *doc,
                      const char *text,
                      size_t len,
                      char *err,
                      size_t errcap) {
    if (len > INT_MAX) {
        set_error(err, errcap, "SafeTensors JSON header is too large");
        return 0;
    }
    memset(doc, 0, sizeof(*doc));
    doc->text = text;
    doc->text_len = (int)len;
    int parent = -1;
    for (int i = 0; i < (int)len; i++) {
        const unsigned char c = (unsigned char)text[i];
        if (isspace(c) || c == ':' || c == ',') continue;
        if (c == '{' || c == '[') {
            const int token = json_add(doc,
                                       c == '{' ? JSON_OBJECT : JSON_ARRAY,
                                       i,
                                       -1,
                                       parent);
            if (token < 0) goto oom;
            parent = token;
            continue;
        }
        if (c == '}' || c == ']') {
            if (parent < 0 ||
                (c == '}' && doc->tokens[parent].type != JSON_OBJECT) ||
                (c == ']' && doc->tokens[parent].type != JSON_ARRAY)) {
                set_error(err, errcap, "malformed SafeTensors JSON");
                free(doc->tokens);
                doc->tokens = NULL;
                return 0;
            }
            doc->tokens[parent].end = i + 1;
            parent = doc->tokens[parent].parent;
            continue;
        }
        if (c == '"') {
            const int start = ++i;
            bool escaped = false;
            while (i < (int)len) {
                if (escaped) {
                    escaped = false;
                } else if (text[i] == '\\') {
                    escaped = true;
                } else if (text[i] == '"') {
                    break;
                }
                i++;
            }
            if (i >= (int)len) {
                set_error(err, errcap, "unterminated SafeTensors JSON string");
                free(doc->tokens);
                doc->tokens = NULL;
                return 0;
            }
            if (json_add(doc, JSON_STRING, start, i, parent) < 0) goto oom;
            continue;
        }
        const int start = i;
        while (i + 1 < (int)len) {
            const char next = text[i + 1];
            if (isspace((unsigned char)next) ||
                next == ',' || next == ']' || next == '}') {
                break;
            }
            i++;
        }
        if (json_add(doc, JSON_PRIMITIVE, start, i + 1, parent) < 0) goto oom;
    }
    if (parent != -1 || doc->count == 0) {
        set_error(err, errcap, "incomplete SafeTensors JSON");
        free(doc->tokens);
        doc->tokens = NULL;
        return 0;
    }
    return 1;

oom:
    set_error(err, errcap, "out of memory parsing SafeTensors JSON");
    free(doc->tokens);
    doc->tokens = NULL;
    return 0;
}

static void json_free(json_doc *doc) {
    free(doc->tokens);
    memset(doc, 0, sizeof(*doc));
}

static int json_skip(const json_doc *doc, int index) {
    const int end = doc->tokens[index].end;
    index++;
    while (index < doc->count && doc->tokens[index].start < end) index++;
    return index;
}

static bool json_equal(const json_doc *doc, int index, const char *value) {
    const json_token *token = &doc->tokens[index];
    const size_t len = (size_t)(token->end - token->start);
    return strlen(value) == len &&
           memcmp(doc->text + token->start, value, len) == 0;
}

static int json_object_get(const json_doc *doc, int object, const char *key) {
    if (object < 0 || object >= doc->count ||
        doc->tokens[object].type != JSON_OBJECT) {
        return -1;
    }
    for (int i = object + 1;
         i < doc->count && doc->tokens[i].parent == object;) {
        const int value = i + 1;
        if (value >= doc->count || doc->tokens[value].parent != object) {
            return -1;
        }
        if (json_equal(doc, i, key)) return value;
        i = json_skip(doc, value);
    }
    return -1;
}

static int json_u64(const json_doc *doc, int index, uint64_t *out) {
    if (index < 0 || index >= doc->count ||
        doc->tokens[index].type != JSON_PRIMITIVE) {
        return 0;
    }
    const json_token *token = &doc->tokens[index];
    const char *p = doc->text + token->start;
    const char *end = doc->text + token->end;
    if (p == end) return 0;
    uint64_t value = 0;
    while (p < end) {
        if (*p < '0' || *p > '9') return 0;
        const uint32_t digit = (uint32_t)(*p++ - '0');
        if (value > (UINT64_MAX - digit) / 10u) return 0;
        value = value * 10u + digit;
    }
    *out = value;
    return 1;
}

static k3_st_dtype parse_dtype(const json_doc *doc, int index) {
    if (json_equal(doc, index, "U8")) return K3_ST_DTYPE_U8;
    if (json_equal(doc, index, "F32")) return K3_ST_DTYPE_F32;
    if (json_equal(doc, index, "F16")) return K3_ST_DTYPE_F16;
    if (json_equal(doc, index, "BF16")) return K3_ST_DTYPE_BF16;
    return K3_ST_DTYPE_INVALID;
}

uint32_t k3_st_dtype_size(k3_st_dtype dtype) {
    switch (dtype) {
    case K3_ST_DTYPE_U8: return 1;
    case K3_ST_DTYPE_F16:
    case K3_ST_DTYPE_BF16: return 2;
    case K3_ST_DTYPE_F32: return 4;
    default: return 0;
    }
}

const char *k3_st_dtype_name(k3_st_dtype dtype) {
    switch (dtype) {
    case K3_ST_DTYPE_U8: return "U8";
    case K3_ST_DTYPE_F16: return "F16";
    case K3_ST_DTYPE_BF16: return "BF16";
    case K3_ST_DTYPE_F32: return "F32";
    case K3_ST_DTYPE_Q8_0: return "Q8_0";
    case K3_ST_DTYPE_Q2_K: return "Q2_K";
    case K3_ST_DTYPE_IQ2_XXS: return "IQ2_XXS";
    case K3_ST_DTYPE_MXFP4: return "MXFP4";
    default: return "INVALID";
    }
}

uint64_t k3_st_nelements(const k3_st_tensor *tensor) {
    if (!tensor) return 0;
    uint64_t count = 1;
    for (uint32_t i = 0; i < tensor->n_dims; i++) {
        if (tensor->shape[i] != 0 && count > UINT64_MAX / tensor->shape[i]) {
            return 0;
        }
        count *= tensor->shape[i];
    }
    return count;
}

static uint64_t hash_name(const char *name) {
    uint64_t hash = UINT64_C(1469598103934665603);
    while (*name) {
        hash ^= (uint8_t)*name++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int build_hash(k3_st_shard *shard, char *err, size_t errcap) {
    uint32_t cap = 2;
    while (cap < shard->tensor_count * 3u) {
        if (cap > UINT32_MAX / 2u) {
            set_error(err, errcap, "too many tensors in shard");
            return 0;
        }
        cap *= 2u;
    }
    shard->hash = calloc(cap, sizeof(shard->hash[0]));
    if (!shard->hash) {
        set_error(err, errcap, "out of memory indexing SafeTensors shard");
        return 0;
    }
    shard->hash_cap = cap;
    for (uint32_t i = 0; i < shard->tensor_count; i++) {
        uint32_t slot = (uint32_t)hash_name(shard->tensors[i].name) & (cap - 1u);
        while (shard->hash[slot].key) slot = (slot + 1u) & (cap - 1u);
        shard->hash[slot] = (hash_slot){
            .key = shard->tensors[i].name,
            .value = i,
        };
    }
    return 1;
}

static int append_tensor(k3_st_shard *shard,
                         const json_doc *doc,
                         int name_token,
                         int value_token,
                         char *err,
                         size_t errcap) {
    const int dtype_token = json_object_get(doc, value_token, "dtype");
    const int shape_token = json_object_get(doc, value_token, "shape");
    const int offsets_token = json_object_get(doc, value_token, "data_offsets");
    if (dtype_token < 0 || shape_token < 0 || offsets_token < 0 ||
        doc->tokens[shape_token].type != JSON_ARRAY ||
        doc->tokens[offsets_token].type != JSON_ARRAY) {
        set_error(err, errcap, "malformed tensor entry in SafeTensors header");
        return 0;
    }

    const json_token *name = &doc->tokens[name_token];
    k3_st_tensor tensor = {
        .name = dup_slice(doc->text + name->start,
                          (size_t)(name->end - name->start)),
        .dtype = parse_dtype(doc, dtype_token),
    };
    if (!tensor.name) {
        set_error(err, errcap, "out of memory reading tensor name");
        return 0;
    }
    if (tensor.dtype == K3_ST_DTYPE_INVALID) {
        set_error(err, errcap, "unsupported dtype for %s", tensor.name);
        free((char *)tensor.name);
        return 0;
    }

    for (int i = shape_token + 1;
         i < doc->count && doc->tokens[i].parent == shape_token;
         i = json_skip(doc, i)) {
        if (tensor.n_dims == K3_ST_MAX_DIMS ||
            !json_u64(doc, i, &tensor.shape[tensor.n_dims])) {
            set_error(err, errcap, "invalid shape for %s", tensor.name);
            free((char *)tensor.name);
            return 0;
        }
        tensor.n_dims++;
    }

    uint64_t offsets[2] = {0};
    uint32_t n_offsets = 0;
    for (int i = offsets_token + 1;
         i < doc->count && doc->tokens[i].parent == offsets_token;
         i = json_skip(doc, i)) {
        if (n_offsets == 2 || !json_u64(doc, i, &offsets[n_offsets])) {
            set_error(err, errcap, "invalid data offsets for %s", tensor.name);
            free((char *)tensor.name);
            return 0;
        }
        n_offsets++;
    }
    if (n_offsets != 2 || offsets[1] < offsets[0] ||
        offsets[0] > UINT64_MAX - shard->data_offset) {
        set_error(err, errcap, "invalid data range for %s", tensor.name);
        free((char *)tensor.name);
        return 0;
    }
    tensor.offset = shard->data_offset + offsets[0];
    tensor.size = offsets[1] - offsets[0];
    if (tensor.offset > shard->file_size ||
        tensor.size > shard->file_size - tensor.offset) {
        set_error(err, errcap, "tensor %s extends past end of shard", tensor.name);
        free((char *)tensor.name);
        return 0;
    }
    const uint64_t count = k3_st_nelements(&tensor);
    const uint32_t element_size = k3_st_dtype_size(tensor.dtype);
    if (element_size == 0 ||
        (count == 0 && tensor.size != 0) ||
        count > UINT64_MAX / element_size ||
        count * element_size != tensor.size) {
        set_error(err,
                  errcap,
                  "tensor %s has inconsistent shape and byte size",
                  tensor.name);
        free((char *)tensor.name);
        return 0;
    }

    if (shard->tensor_count == UINT32_MAX) {
        set_error(err, errcap, "too many tensors in shard");
        free((char *)tensor.name);
        return 0;
    }
    k3_st_tensor *tensors = grow_array(shard->tensors,
                                       shard->tensor_count + 1u,
                                       sizeof(tensors[0]));
    if (!tensors) {
        set_error(err, errcap, "out of memory reading tensor table");
        free((char *)tensor.name);
        return 0;
    }
    shard->tensors = tensors;
    shard->tensors[shard->tensor_count++] = tensor;
    return 1;
}

typedef struct {
    int fd;
    uint64_t size;
    uint64_t offset;
} gguf_reader;

static int gguf_read(gguf_reader *reader,
                     void *value,
                     size_t bytes,
                     char *err,
                     size_t errcap) {
    if (!reader || reader->offset > reader->size ||
        bytes > reader->size - reader->offset ||
        !pread_exact(reader->fd, value, bytes, reader->offset, err, errcap)) {
        if (reader && reader->offset <= reader->size &&
            bytes > reader->size - reader->offset) {
            set_error(err, errcap, "truncated GGUF metadata");
        }
        return 0;
    }
    reader->offset += bytes;
    return 1;
}

static int gguf_read_u32(gguf_reader *reader,
                         uint32_t *value,
                         char *err,
                         size_t errcap) {
    uint8_t bytes[4];
    if (!gguf_read(reader, bytes, sizeof(bytes), err, errcap)) return 0;
    *value = load_u32_le(bytes);
    return 1;
}

static int gguf_read_u64(gguf_reader *reader,
                         uint64_t *value,
                         char *err,
                         size_t errcap) {
    uint8_t bytes[8];
    if (!gguf_read(reader, bytes, sizeof(bytes), err, errcap)) return 0;
    *value = load_u64_le(bytes);
    return 1;
}

static int gguf_read_string(gguf_reader *reader,
                            char **value,
                            uint64_t *length_out,
                            char *err,
                            size_t errcap) {
    uint64_t length = 0;
    if (!gguf_read_u64(reader, &length, err, errcap) ||
        length > reader->size - reader->offset ||
        (value && length > SIZE_MAX - 1u)) {
        set_error(err, errcap, "invalid GGUF string length");
        return 0;
    }
    if (value) {
        char *text = malloc((size_t)length + 1u);
        if (!text) {
            set_error(err, errcap, "out of memory reading GGUF string");
            return 0;
        }
        if (!gguf_read(reader, text, (size_t)length, err, errcap)) {
            free(text);
            return 0;
        }
        text[length] = '\0';
        *value = text;
    } else {
        reader->offset += length;
    }
    if (length_out) *length_out = length;
    return 1;
}

static uint32_t gguf_scalar_size(uint32_t type) {
    switch (type) {
    case 0:  /* UINT8 */
    case 1:  /* INT8 */
    case 7:  /* BOOL */
        return 1;
    case 2:  /* UINT16 */
    case 3:  /* INT16 */
        return 2;
    case 4:  /* UINT32 */
    case 5:  /* INT32 */
    case 6:  /* FLOAT32 */
        return 4;
    case 10: /* UINT64 */
    case 11: /* INT64 */
    case 12: /* FLOAT64 */
        return 8;
    default:
        return 0;
    }
}

static int gguf_skip_value(gguf_reader *reader,
                           uint32_t type,
                           char *err,
                           size_t errcap) {
    if (type == 8) return gguf_read_string(reader, NULL, NULL, err, errcap);
    if (type == 9) {
        uint32_t element_type = 0;
        uint64_t count = 0;
        if (!gguf_read_u32(reader, &element_type, err, errcap) ||
            !gguf_read_u64(reader, &count, err, errcap)) {
            return 0;
        }
        const uint32_t scalar = gguf_scalar_size(element_type);
        if (scalar) {
            if (count > (reader->size - reader->offset) / scalar) {
                set_error(err, errcap, "truncated GGUF array");
                return 0;
            }
            reader->offset += count * scalar;
            return 1;
        }
        if (element_type != 8 || count > UINT32_MAX) {
            set_error(err, errcap, "unsupported GGUF array element type %u",
                      element_type);
            return 0;
        }
        for (uint64_t i = 0; i < count; i++) {
            if (!gguf_read_string(reader, NULL, NULL, err, errcap)) return 0;
        }
        return 1;
    }
    const uint32_t scalar = gguf_scalar_size(type);
    if (!scalar || scalar > reader->size - reader->offset) {
        set_error(err, errcap, "unsupported or truncated GGUF value type %u",
                  type);
        return 0;
    }
    reader->offset += scalar;
    return 1;
}

static k3_st_dtype gguf_tensor_dtype(uint32_t type) {
    switch (type) {
    case 0:  return K3_ST_DTYPE_F32;
    case 1:  return K3_ST_DTYPE_F16;
    case 8:  return K3_ST_DTYPE_Q8_0;
    case 10: return K3_ST_DTYPE_Q2_K;
    case 16: return K3_ST_DTYPE_IQ2_XXS;
    case 30: return K3_ST_DTYPE_BF16;
    case 39: return K3_ST_DTYPE_MXFP4;
    default: return K3_ST_DTYPE_INVALID;
    }
}

static uint64_t gguf_row_size(k3_st_dtype type, uint64_t elements) {
    uint64_t block = 0;
    uint64_t bytes = 0;
    switch (type) {
    case K3_ST_DTYPE_F32: block = 1; bytes = 4; break;
    case K3_ST_DTYPE_F16:
    case K3_ST_DTYPE_BF16: block = 1; bytes = 2; break;
    case K3_ST_DTYPE_Q8_0: block = 32; bytes = 34; break;
    case K3_ST_DTYPE_Q2_K: block = 256; bytes = 84; break;
    case K3_ST_DTYPE_IQ2_XXS: block = 256; bytes = 66; break;
    case K3_ST_DTYPE_MXFP4: block = 32; bytes = 17; break;
    default: return 0;
    }
    if (elements == 0 || elements % block != 0 ||
        elements / block > UINT64_MAX / bytes) {
        return 0;
    }
    return elements / block * bytes;
}

static int append_gguf_tensor(k3_st_shard *shard,
                              gguf_reader *reader,
                              char *err,
                              size_t errcap) {
    k3_st_tensor tensor = {0};
    uint64_t ne[K3_ST_MAX_DIMS] = {0};
    uint32_t ggml_type = 0;
    if (!gguf_read_string(reader, (char **)&tensor.name, NULL, err, errcap) ||
        !gguf_read_u32(reader, &tensor.n_dims, err, errcap) ||
        tensor.n_dims == 0 || tensor.n_dims > K3_ST_MAX_DIMS) {
        set_error(err, errcap, "invalid GGUF tensor name or rank");
        free((char *)tensor.name);
        return 0;
    }
    for (uint32_t i = 0; i < tensor.n_dims; i++) {
        if (!gguf_read_u64(reader, &ne[i], err, errcap) || ne[i] == 0) {
            set_error(err, errcap, "invalid shape for GGUF tensor %s",
                      tensor.name);
            free((char *)tensor.name);
            return 0;
        }
        tensor.shape[tensor.n_dims - 1u - i] = ne[i];
    }
    if (!gguf_read_u32(reader, &ggml_type, err, errcap) ||
        !gguf_read_u64(reader, &tensor.offset, err, errcap)) {
        free((char *)tensor.name);
        return 0;
    }
    tensor.dtype = gguf_tensor_dtype(ggml_type);
    const uint64_t row = gguf_row_size(tensor.dtype, ne[0]);
    uint64_t rows = 1;
    for (uint32_t i = 1; row && i < tensor.n_dims; i++) {
        if (rows > UINT64_MAX / ne[i]) {
            rows = 0;
            break;
        }
        rows *= ne[i];
    }
    if (tensor.dtype == K3_ST_DTYPE_INVALID || row == 0 ||
        rows == 0 || rows > UINT64_MAX / row) {
        set_error(err, errcap,
                  "unsupported GGUF type %u or shape for tensor %s",
                  ggml_type, tensor.name);
        free((char *)tensor.name);
        return 0;
    }
    tensor.size = rows * row;

    if (shard->tensor_count == UINT32_MAX) {
        set_error(err, errcap, "too many tensors in GGUF");
        free((char *)tensor.name);
        return 0;
    }
    k3_st_tensor *tensors = grow_array(
        shard->tensors, shard->tensor_count + 1u, sizeof(tensors[0]));
    if (!tensors) {
        set_error(err, errcap, "out of memory reading GGUF tensor table");
        free((char *)tensor.name);
        return 0;
    }
    shard->tensors = tensors;
    shard->tensors[shard->tensor_count++] = tensor;
    return 1;
}

static uint64_t align_u64(uint64_t value, uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1u)) != 0 ||
        value > UINT64_MAX - (alignment - 1u)) {
        return 0;
    }
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static int parse_gguf(k3_st_shard *shard,
                      char *err,
                      size_t errcap) {
    gguf_reader reader = {
        .fd = shard->fd,
        .size = shard->file_size,
        .offset = 4,
    };
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    uint32_t alignment = 32;
    if (!gguf_read_u32(&reader, &version, err, errcap) ||
        version != 3 ||
        !gguf_read_u64(&reader, &tensor_count, err, errcap) ||
        !gguf_read_u64(&reader, &kv_count, err, errcap) ||
        tensor_count == 0 || tensor_count > UINT32_MAX ||
        kv_count > UINT32_MAX) {
        set_error(err, errcap, "invalid or unsupported GGUF v%u header",
                  version);
        return 0;
    }
    for (uint64_t i = 0; i < kv_count; i++) {
        char *key = NULL;
        uint32_t type = 0;
        if (!gguf_read_string(&reader, &key, NULL, err, errcap) ||
            !gguf_read_u32(&reader, &type, err, errcap)) {
            free(key);
            return 0;
        }
        if (strcmp(key, "general.alignment") == 0 && type == 4) {
            if (!gguf_read_u32(&reader, &alignment, err, errcap)) {
                free(key);
                return 0;
            }
        } else if (!gguf_skip_value(&reader, type, err, errcap)) {
            free(key);
            return 0;
        }
        free(key);
    }
    for (uint64_t i = 0; i < tensor_count; i++) {
        if (!append_gguf_tensor(shard, &reader, err, errcap)) return 0;
    }
    shard->data_offset = align_u64(reader.offset, alignment);
    if (shard->data_offset == 0 || shard->data_offset > shard->file_size) {
        set_error(err, errcap, "invalid GGUF tensor data offset");
        return 0;
    }
    for (uint32_t i = 0; i < shard->tensor_count; i++) {
        k3_st_tensor *tensor = &shard->tensors[i];
        if (tensor->offset > UINT64_MAX - shard->data_offset) {
            set_error(err, errcap, "GGUF tensor offset overflow");
            return 0;
        }
        tensor->offset += shard->data_offset;
        if (tensor->offset > shard->file_size ||
            tensor->size > shard->file_size - tensor->offset) {
            set_error(err, errcap, "GGUF tensor %s extends past end of file",
                      tensor->name);
            return 0;
        }
    }
    shard->gguf = true;
    return build_hash(shard, err, errcap);
}

int k3_st_gguf_read_string(const char *path,
                           const char *wanted_key,
                           uint8_t **value,
                           size_t *value_size,
                           char *err,
                           size_t errcap) {
    if (value) *value = NULL;
    if (value_size) *value_size = 0;
    if (!path || !wanted_key || !value || !value_size) {
        set_error(err, errcap, "invalid GGUF metadata string request");
        return 0;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_error(err, errcap, "open %s: %s", path, strerror(errno));
        return 0;
    }
    struct stat st;
    uint8_t magic[4];
    if (fstat(fd, &st) != 0 || st.st_size < 24 ||
        !pread_exact(fd, magic, sizeof(magic), 0, err, errcap) ||
        memcmp(magic, "GGUF", 4) != 0) {
        set_error(err, errcap, "%s is not a GGUF file", path);
        close(fd);
        return 0;
    }
    gguf_reader reader = {
        .fd = fd,
        .size = (uint64_t)st.st_size,
        .offset = 4,
    };
    uint32_t version = 0;
    uint64_t tensors = 0;
    uint64_t kv_count = 0;
    int ok =
        gguf_read_u32(&reader, &version, err, errcap) &&
        version == 3 &&
        gguf_read_u64(&reader, &tensors, err, errcap) &&
        gguf_read_u64(&reader, &kv_count, err, errcap);
    (void)tensors;
    for (uint64_t i = 0; ok && i < kv_count; i++) {
        char *key = NULL;
        uint32_t type = 0;
        ok = gguf_read_string(&reader, &key, NULL, err, errcap) &&
             gguf_read_u32(&reader, &type, err, errcap);
        if (ok && strcmp(key, wanted_key) == 0) {
            char *string = NULL;
            uint64_t length = 0;
            if (type != 8 ||
                !gguf_read_string(
                    &reader, &string, &length, err, errcap) ||
                length > SIZE_MAX) {
                set_error(err, errcap,
                          "GGUF metadata %s is not a valid string",
                          wanted_key);
                ok = 0;
            } else {
                *value = (uint8_t *)string;
                *value_size = (size_t)length;
                free(key);
                close(fd);
                return 1;
            }
        } else if (ok) {
            ok = gguf_skip_value(&reader, type, err, errcap);
        }
        free(key);
    }
    if (ok) {
        set_error(err, errcap, "GGUF metadata string %s is missing",
                  wanted_key);
    }
    close(fd);
    return 0;
}

int k3_st_open(k3_st_shard **out, const char *path, char *err, size_t errcap) {
    if (out) *out = NULL;
    if (!out || !path || !path[0]) {
        set_error(err, errcap, "missing SafeTensors shard path");
        return 0;
    }

    k3_st_shard *shard = calloc(1, sizeof(*shard));
    if (!shard) {
        set_error(err, errcap, "out of memory opening SafeTensors shard");
        return 0;
    }
    shard->fd = -1;
    shard->path = dup_slice(path, strlen(path));
    if (!shard->path) {
        set_error(err, errcap, "out of memory copying shard path");
        goto fail;
    }
    shard->fd = open(path, O_RDONLY);
    if (shard->fd < 0) {
        set_error(err, errcap, "open %s: %s", path, strerror(errno));
        goto fail;
    }
    struct stat st;
    if (fstat(shard->fd, &st) != 0) {
        set_error(err,
                  errcap,
                  "stat %s: %s",
                  path,
                  strerror(errno));
        goto fail;
    }
    if (st.st_size < 8) {
        set_error(err, errcap, "%s is too short for SafeTensors", path);
        goto fail;
    }
    shard->file_size = (uint64_t)st.st_size;

    uint8_t header_size_bytes[8];
    if (!pread_exact(shard->fd,
                     header_size_bytes,
                     sizeof(header_size_bytes),
                     0,
                     err,
                     errcap)) {
        goto fail;
    }
    if (memcmp(header_size_bytes, "GGUF", 4) == 0) {
        if (!parse_gguf(shard, err, errcap)) goto fail;
        *out = shard;
        return 1;
    }
    const uint64_t header_size = load_u64_le(header_size_bytes);
    if (header_size == 0 ||
        header_size > SIZE_MAX - 1u ||
        header_size > shard->file_size - 8u) {
        set_error(err, errcap, "invalid SafeTensors header size");
        goto fail;
    }
    shard->data_offset = 8u + header_size;
    char *header = malloc((size_t)header_size + 1u);
    if (!header) {
        set_error(err, errcap, "out of memory reading SafeTensors header");
        goto fail;
    }
    if (!pread_exact(shard->fd,
                     header,
                     (size_t)header_size,
                     8,
                     err,
                     errcap)) {
        free(header);
        goto fail;
    }
    header[header_size] = '\0';

    json_doc doc;
    if (!json_parse(&doc, header, (size_t)header_size, err, errcap)) {
        free(header);
        goto fail;
    }
    if (doc.tokens[0].type != JSON_OBJECT) {
        set_error(err, errcap, "SafeTensors header root is not an object");
        json_free(&doc);
        free(header);
        goto fail;
    }
    int ok = 1;
    for (int i = 1;
         ok && i < doc.count && doc.tokens[i].parent == 0;) {
        const int value = i + 1;
        if (value >= doc.count || doc.tokens[value].parent != 0) {
            set_error(err, errcap, "malformed SafeTensors header object");
            ok = 0;
            break;
        }
        if (!json_equal(&doc, i, "__metadata__")) {
            ok = append_tensor(shard, &doc, i, value, err, errcap);
        }
        i = json_skip(&doc, value);
    }
    json_free(&doc);
    free(header);
    if (!ok || !build_hash(shard, err, errcap)) goto fail;

    *out = shard;
    return 1;

fail:
    k3_st_close(shard);
    return 0;
}

void k3_st_close(k3_st_shard *shard) {
    if (!shard) return;
    k3_st_unmap(shard);
    if (shard->fd >= 0) close(shard->fd);
    for (uint32_t i = 0; i < shard->tensor_count; i++) {
        free((char *)shard->tensors[i].name);
    }
    free(shard->tensors);
    free(shard->hash);
    free(shard->path);
    free(shard);
}

const char *k3_st_path(const k3_st_shard *shard) {
    return shard ? shard->path : NULL;
}

int k3_st_fd(const k3_st_shard *shard) {
    return shard ? shard->fd : -1;
}

bool k3_st_is_gguf(const k3_st_shard *shard) {
    return shard && shard->gguf;
}

uint64_t k3_st_file_size(const k3_st_shard *shard) {
    return shard ? shard->file_size : 0;
}

uint64_t k3_st_data_offset(const k3_st_shard *shard) {
    return shard ? shard->data_offset : 0;
}

uint32_t k3_st_tensor_count(const k3_st_shard *shard) {
    return shard ? shard->tensor_count : 0;
}

const k3_st_tensor *k3_st_tensor_at(const k3_st_shard *shard, uint32_t index) {
    if (!shard || index >= shard->tensor_count) return NULL;
    return &shard->tensors[index];
}

const k3_st_tensor *k3_st_find(const k3_st_shard *shard, const char *name) {
    if (!shard || !name || !shard->hash_cap) return NULL;
    uint32_t slot = (uint32_t)hash_name(name) & (shard->hash_cap - 1u);
    while (shard->hash[slot].key) {
        if (strcmp(shard->hash[slot].key, name) == 0) {
            return &shard->tensors[shard->hash[slot].value];
        }
        slot = (slot + 1u) & (shard->hash_cap - 1u);
    }
    return NULL;
}

int k3_st_map(k3_st_shard *shard, char *err, size_t errcap) {
    if (!shard) {
        set_error(err, errcap, "missing SafeTensors shard");
        return 0;
    }
    if (shard->map) return 1;
    if (shard->file_size > SIZE_MAX) {
        set_error(err, errcap, "SafeTensors shard is too large to mmap");
        return 0;
    }
    void *map = mmap(NULL,
                     (size_t)shard->file_size,
                     PROT_READ,
                     MAP_PRIVATE,
                     shard->fd,
                     0);
    if (map == MAP_FAILED) {
        set_error(err, errcap, "mmap %s: %s", shard->path, strerror(errno));
        return 0;
    }
    shard->map = map;
    return 1;
}

void k3_st_unmap(k3_st_shard *shard) {
    if (!shard || !shard->map) return;
    munmap(shard->map, (size_t)shard->file_size);
    shard->map = NULL;
}

const void *k3_st_map_base(const k3_st_shard *shard) {
    return shard ? shard->map : NULL;
}

int k3_st_prefetch(k3_st_shard *shard,
                   uint64_t offset,
                   uint64_t size,
                   bool touch,
                   char *err,
                   size_t errcap) {
    if (!shard || offset > shard->file_size ||
        size == 0 || size > shard->file_size - offset) {
        set_error(err, errcap, "prefetch range is outside SafeTensors shard");
        return 0;
    }

#if defined(F_RDADVISE)
    uint64_t pos = offset;
    uint64_t remaining = size;
    while (remaining != 0) {
        const uint64_t chunk = remaining > (uint64_t)INT_MAX ?
            (uint64_t)INT_MAX : remaining;
        if (pos > (uint64_t)LLONG_MAX) break;
        struct radvisory advisory = {
            .ra_offset = (off_t)pos,
            .ra_count = (int)chunk,
        };
        (void)fcntl(shard->fd, F_RDADVISE, &advisory);
        pos += chunk;
        remaining -= chunk;
    }
#endif

    if (!touch) return 1;
    if (!k3_st_map(shard, err, errcap)) return 0;

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        set_error(err, errcap, "cannot determine VM page size");
        return 0;
    }
    const uint64_t page = (uint64_t)page_size;
    const uint64_t page_offset = offset & ~(page - 1u);
    const uint64_t leading = offset - page_offset;
    if (leading > UINT64_MAX - size) {
        set_error(err, errcap, "prefetch range overflow");
        return 0;
    }
    uint64_t bytes = leading + size;
    if (bytes > UINT64_MAX - (page - 1u)) {
        set_error(err, errcap, "prefetch alignment overflow");
        return 0;
    }
    bytes = (bytes + page - 1u) & ~(page - 1u);
    if (bytes > shard->file_size - page_offset) {
        bytes = shard->file_size - page_offset;
    }
#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((uint8_t *)shard->map + page_offset,
                        (size_t)bytes,
                        POSIX_MADV_WILLNEED);
#endif
    const volatile uint8_t *p =
        (const volatile uint8_t *)shard->map + page_offset;
    uint8_t sink = 0;
    for (uint64_t i = 0; i < bytes; i += page) sink ^= p[i];
    sink ^= p[bytes - 1u];
    (void)sink;
    return 1;
}
