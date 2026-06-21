# Project Journal

## Branch: api-key-auth

### Goal
Add API key authentication to ds4-server, allowing the server to require an API key for all HTTP requests.

### Changes Made
- **ds4_server.c**: Added `api_key` field to `server_config`, `server`, and `http_request` structs
- **ds4_server.c**: Extended `read_http_request` to parse `Authorization: Bearer <key>` and `x-api-key: <key>` headers
- **ds4_server.c**: Added `--api-key` CLI argument handling in `parse_options`
- **ds4_server.c**: Added API key validation check in `client_main` before dispatching requests (returns 401 on mismatch)
- **ds4_server.c**: Added `free(s->api_key)` in `server_close_resources`
- **ds4_help.c**: Added `--api-key KEY` documentation in `print_server_api`
- **setup_ds4_service.sh**: Added `API_KEY` config variable and conditional `--api-key` in service ExecStart

### Status
All changes committed and pushed to `origin/api-key-auth`.
Remote origin changed from `antirez/ds4` to `mooming/ds4`.

Branch `api-key-auth` is ready for review.
GitHub: https://github.com/mooming/ds4/pull/new/api-key-auth

### Debug Logging Added
- Startup: logs when API key auth is enabled (shows first 8 chars of key)
- Per-request: logs method, path, received auth value, and expected key prefix
- Rejection: logs whether no header was present or key mismatch with lengths
