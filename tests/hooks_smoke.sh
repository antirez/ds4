#!/bin/sh
set -eu
make test-hooks
printf '%s\n' 'hook smoke: ok'
