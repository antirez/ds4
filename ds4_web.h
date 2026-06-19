#ifndef DS4_WEB_H
#define DS4_WEB_H

#include <stddef.h>
#include <stdbool.h>

typedef int (*ds4_web_confirm_fn)(void *privdata, const char *message,
                                  char *err, size_t err_len);
typedef void (*ds4_web_log_fn)(void *privdata, const char *message);
typedef bool (*ds4_web_cancel_fn)(void *privdata);

typedef struct {
    const char *home_dir;
    int port;
    ds4_web_confirm_fn confirm;
    void *confirm_privdata;
    ds4_web_log_fn log;
    void *log_privdata;
    ds4_web_cancel_fn cancel;
    void *cancel_privdata;
} ds4_web_config;

typedef struct ds4_web ds4_web;

ds4_web *ds4_web_create(const ds4_web_config *cfg);
void ds4_web_free(ds4_web *web);

char *ds4_web_google_search(ds4_web *web, const char *query,
                            char *err, size_t err_len);
char *ds4_web_visit_page(ds4_web *web, const char *url,
                         char *err, size_t err_len);

#endif
