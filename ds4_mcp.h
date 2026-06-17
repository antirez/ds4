#ifndef DS4_MCP_H
#define DS4_MCP_H

#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>

typedef bool (*ds4_mcp_cancel_fn)(void *privdata);

typedef struct {
    char *name;
    char *title;
    char *description;
    char *input_schema;
    char *dsml_name;
    int server_index;
} ds4_mcp_tool;

typedef struct {
    char *name;
    char *command;
    char **args;
    int argc;
    char **env_names;
    char **env_values;
    int envc;
    pid_t pid;
    int in_fd;
    int out_fd;
    int next_id;
} ds4_mcp_server;

typedef struct {
    ds4_mcp_server *servers;
    int servers_len;
    int servers_cap;
    ds4_mcp_tool *tools;
    int tools_len;
    int tools_cap;
    int connect_timeout_ms;
    int call_timeout_ms;
} ds4_mcp;

void ds4_mcp_init(ds4_mcp *mcp);
void ds4_mcp_close(ds4_mcp *mcp);
bool ds4_mcp_connect_json(ds4_mcp *mcp, const char *servers_json,
                          ds4_mcp_cancel_fn cancel, void *cancel_privdata,
                          char *err, size_t err_len);
char *ds4_mcp_tools_prompt(const ds4_mcp *mcp);
const ds4_mcp_tool *ds4_mcp_find_tool(const ds4_mcp *mcp, const char *dsml_name);
char *ds4_mcp_call_tool(ds4_mcp *mcp, const char *dsml_name,
                        const char *arguments_json,
                        ds4_mcp_cancel_fn cancel, void *cancel_privdata,
                        char *err, size_t err_len);

#endif
