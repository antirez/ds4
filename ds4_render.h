/* Streaming Markdown renderer for terminal chat output.
 *
 * The renderer consumes model output one chunk at a time and writes terminal
 * text to a FILE*.  It is a streaming parser: ambiguous bytes (marker runs,
 * incomplete UTF-8 sequences, table rows) are buffered only for as long as the
 * parser needs them to decide what they are.
 *
 * With color disabled the renderer is a byte-for-byte passthrough except for
 * the <think> handling, which keeps the historical CLI semantics: the tags are
 * removed and a newline is forced after the closing tag. */

#ifndef DS4_RENDER_H
#define DS4_RENDER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define DS4R_LINE_BUF   96      /* line-start scratch: indent plus markers */
#define DS4R_FENCE_LANG 32

typedef struct ds4r_syntax ds4r_syntax;

typedef enum {
    DS4R_MARK_NONE = 0,
    DS4R_MARK_STAR,
    DS4R_MARK_BACKTICK,
} ds4r_mark;

typedef enum {
    DS4R_LINE_OFF = 0,      /* mid line: bytes go to the inline parser */
    DS4R_LINE_SCAN,         /* classifying the start of a logical line */
    DS4R_LINE_TABLE,        /* buffering consecutive table rows */
} ds4r_line_mode;

typedef enum {
    DS4R_SCAN_INDENT = 0,
    DS4R_SCAN_HASH,
    DS4R_SCAN_MARKER,       /* run of '-', '*', '_' or '+' */
    DS4R_SCAN_MARKER_TAIL,  /* spaces after a rule-length marker run */
    DS4R_SCAN_NUMBER,
    DS4R_SCAN_NUMBER_DOT,
    DS4R_SCAN_QUOTE,        /* just emitted a '>' prefix */
} ds4r_scan_state;

typedef struct {
    FILE *fp;
    bool use_color;         /* ANSI attributes may be emitted */
    bool format_markdown;   /* markdown constructs are rendered */
    bool format_tables;     /* pipe tables are laid out as boxes */
    bool think_markdown;    /* thinking uses the muted markdown palette */
    bool format_thinking;   /* <think> blocks are filtered */
    bool in_think;
    bool color_open;        /* a non-default attribute is currently set */
    bool last_output_newline;
    unsigned attr_key;      /* attribute set the open escape stands for */
    int cols_override;      /* 0: ask the terminal */

    /* <think> marker reassembly across writes. */
    char pending[16];
    size_t pending_len;

    /* UTF-8 accumulation, so escapes never land inside a codepoint. */
    char utf8_pending[4];
    size_t utf8_pending_len;
    size_t utf8_pending_need;

    /* Inline markdown. */
    ds4r_mark md_mark;
    size_t md_mark_len;
    bool md_bold;
    bool md_italic;
    bool md_inline_code;
    bool md_heading;
    bool md_heading_underline;

    /* Fenced code blocks. */
    bool md_code_block;
    bool md_fence_info;
    bool md_code_line_start;
    bool md_code_in_ml_comment;
    const ds4r_syntax *md_syntax;
    char md_fence_lang[DS4R_FENCE_LANG];
    size_t md_fence_lang_len;
    char *code_line;
    size_t code_line_len;
    size_t code_line_cap;

    /* Line-start classification. */
    ds4r_line_mode line_mode;
    ds4r_scan_state scan;
    char line_buf[DS4R_LINE_BUF];
    size_t line_len;
    size_t indent_len;      /* bytes of line_buf that precede the marker run */
    size_t marker_run;
    char marker_ch;

    /* Table buffering. */
    char *tbl;
    size_t tbl_len;
    size_t tbl_cap;
    size_t tbl_lines;
    bool tbl_scan;          /* between rows, deciding whether the table goes on */
    bool tbl_raw;           /* buffer cap hit: pass the rest of the table through */
} ds4r;

/* color enables ANSI attributes, markdown rendering, table layout, and the
 * muted thinking palette; the setters below narrow that down.
 * ds4r_set_markdown(false) keeps colors but prints markdown source verbatim,
 * ds4r_set_tables(false) lets table lines flow through as ordinary text, and
 * ds4r_set_think_markdown(false) restores plain single-grey thinking. */
void ds4r_init(ds4r *r, FILE *fp, bool color, bool format_thinking);
void ds4r_set_markdown(ds4r *r, bool enabled);
void ds4r_set_tables(ds4r *r, bool enabled);
void ds4r_set_think_markdown(ds4r *r, bool enabled);
void ds4r_set_in_think(ds4r *r, bool in_think);
void ds4r_set_columns(ds4r *r, int cols);   /* 0 restores terminal detection */
void ds4r_write(ds4r *r, const char *text, size_t len);
void ds4r_finish(ds4r *r);
void ds4r_free(ds4r *r);
void ds4r_newline(ds4r *r);
bool ds4r_at_line_start(const ds4r *r);

/* Locale independent character metrics, also used by the table layout. */
int ds4r_wcwidth(uint32_t cp);
int ds4r_visible_width(const char *s, size_t len);

#endif /* DS4_RENDER_H */
