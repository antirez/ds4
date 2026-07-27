#ifndef PULSAR_HELP_H
#define PULSAR_HELP_H

#include <stdio.h>


typedef enum {
    PULSAR_HELP_DS4,
    PULSAR_HELP_SERVER,
    PULSAR_HELP_AGENT,
    PULSAR_HELP_BENCH,
    PULSAR_HELP_EVAL,
} pulsar_help_tool;

void pulsar_help_print(FILE *fp, pulsar_help_tool tool, const char *topic);


#endif
