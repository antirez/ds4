#ifndef DS4_ACP_H
#define DS4_ACP_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    DS4_ACP_PARSE_OK,
    DS4_ACP_PARSE_JSON,
    DS4_ACP_PARSE_REQUEST,
} ds4_acp_parse_result;

typedef struct {
    bool has_id;
    char *id_json;
    char *method;
    bool has_params;
    char *params_json;
} ds4_acp_request;

void ds4_acp_request_free(ds4_acp_request *r);
ds4_acp_parse_result ds4_acp_parse_request(const char *json,
                                           ds4_acp_request *out,
                                           char *err, size_t err_len);

void ds4_acp_json_ws(const char **p);
bool ds4_acp_json_string(const char **p, char **out);
bool ds4_acp_json_skip_value(const char **p);
bool ds4_acp_json_raw_value(const char **p, char **out);

bool ds4_acp_object_get_string(const char *json, const char *key, char **out);
bool ds4_acp_object_get_raw(const char *json, const char *key, char **out);

char *ds4_acp_json_escape(const char *s, size_t n);

#endif
