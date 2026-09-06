#ifndef DS4_EVAL_CASES_H
#define DS4_EVAL_CASES_H

#include <stddef.h>
#include <stdint.h>

#define EVAL_MAX_CHOICES 10
#define EVAL_MAX_ALIASES 4

typedef enum {
    EVAL_ANSWER_AUTO = 0,
    EVAL_ANSWER_CHOICE,
    EVAL_ANSWER_INTEGER,
    EVAL_ANSWER_RATIONAL,
    EVAL_ANSWER_EXACT_TEXT,
    EVAL_ANSWER_ORDERED_SEQUENCE,
    EVAL_ANSWER_LINE_SET,
} eval_answer_kind;

enum {
    EVAL_SUITE_CORE = 1u << 0,
    EVAL_SUITE_HARD = 1u << 1,
    EVAL_SUITE_HARD_SMOKE = 1u << 2,
};

typedef struct {
    const char *source;
    const char *id;
    const char *domain;
    const char *title;
    const char *question;
    const char *choice[EVAL_MAX_CHOICES];
    const char *answer;
    const char *alias[EVAL_MAX_ALIASES];
    eval_answer_kind answer_kind;
    uint32_t suites;
    uint32_t max_tokens;
} eval_case;

typedef struct {
    const char *name;
    const char *url;
    const char *license;
} eval_source;

extern const eval_case eval_hard_cases[];
extern const size_t eval_hard_case_count;
extern const eval_source eval_hard_sources[];
extern const size_t eval_hard_source_count;

#endif
