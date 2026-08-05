/* Streaming Markdown renderer for terminal chat output.  See ds4_render.h.
 *
 * The parser is a byte pump with three layers.  ds4r_write() strips <think>
 * markers, ds4r_render_byte() classifies logical line starts (headings, lists,
 * quotes, rules, tables), and ds4r_md_feed() handles the inline constructs
 * (bold, italic, inline code, fenced blocks).  Everything that cannot be
 * decided from the bytes seen so far is held in a small buffer: marker runs,
 * partial UTF-8 sequences, and whole table rows.
 *
 * The inline machine, the UTF-8 accumulator, and the fenced-block highlighter
 * follow the agent renderer in ds4_agent.c; the line-start dispatch and the
 * table layout are new.
 *
 * <think> blocks run through the same pipeline with a muted palette, so
 * reasoning keeps its structure without ever using hue. */

#include "ds4_render.h"

#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define DS4R_TBL_MAX_BYTES (64u * 1024u)
#define DS4R_TBL_MAX_LINES 512u
#define DS4R_TBL_MAX_COLS  64

#define DS4R_GREY  "\x1b[38;5;244m"
#define DS4R_RESET "\x1b[0m"

/* Thinking text is rendered with the same markdown pipeline as the answer, but
 * every style maps to a grey variant: reasoning must stay in the background and
 * never compete with the answer for attention.  No hue is ever emitted inside a
 * think block, only SGR 90 plus bold/italic/underline/dim. */
#define DS4R_THINK      "\x1b[90m"
#define DS4R_THINK_DIM  "\x1b[2;90m"

static void ds4r_render_byte(ds4r *r, char c);
static void ds4r_md_feed(ds4r *r, char c);
static void ds4r_scan_byte(ds4r *r, char c);
static void ds4r_scan_flush(ds4r *r);
static void ds4r_table_byte(ds4r *r, char c);
static void ds4r_table_flush(ds4r *r);
static void ds4r_line_begin(ds4r *r);

/* ============================================================================
 * Character metrics
 * ============================================================================ */

typedef struct {
    uint32_t lo;
    uint32_t hi;
} ds4r_range;

/* Combining marks and other zero-advance codepoints.  The table is a compact
 * subset: it covers what shows up in model output rather than the full Unicode
 * combining property. */
static const ds4r_range ds4r_zero_ranges[] = {
    {0x0300, 0x036F}, {0x0483, 0x0489}, {0x0591, 0x05BD}, {0x05BF, 0x05BF},
    {0x05C1, 0x05C2}, {0x05C4, 0x05C5}, {0x05C7, 0x05C7}, {0x0610, 0x061A},
    {0x064B, 0x065F}, {0x0670, 0x0670}, {0x06D6, 0x06DC}, {0x06DF, 0x06E4},
    {0x06E7, 0x06E8}, {0x06EA, 0x06ED}, {0x0711, 0x0711}, {0x0730, 0x074A},
    {0x07A6, 0x07B0}, {0x07EB, 0x07F3}, {0x0816, 0x0819}, {0x081B, 0x0823},
    {0x0825, 0x0827}, {0x0829, 0x082D}, {0x0900, 0x0902}, {0x093A, 0x093A},
    {0x093C, 0x093C}, {0x0941, 0x0948}, {0x094D, 0x094D}, {0x0951, 0x0957},
    {0x0E31, 0x0E31}, {0x0E34, 0x0E3A}, {0x0E47, 0x0E4E}, {0x1AB0, 0x1AFF},
    {0x1DC0, 0x1DFF}, {0x200B, 0x200F}, {0x2060, 0x2064}, {0x20D0, 0x20F0},
    {0xFE00, 0xFE0F}, {0xFE20, 0xFE2F}, {0xFEFF, 0xFEFF}, {0xE0100, 0xE01EF},
};

/* East Asian Wide and Fullwidth blocks, plus the emoji ranges that terminals
 * render double width. */
static const ds4r_range ds4r_wide_ranges[] = {
    {0x1100, 0x115F}, {0x231A, 0x231B}, {0x2329, 0x232A}, {0x23E9, 0x23EC},
    {0x23F0, 0x23F0}, {0x23F3, 0x23F3}, {0x25FD, 0x25FE}, {0x2614, 0x2615},
    {0x2648, 0x2653}, {0x267F, 0x267F}, {0x2693, 0x2693}, {0x26A1, 0x26A1},
    {0x26AA, 0x26AB}, {0x26BD, 0x26BE}, {0x26C4, 0x26C5}, {0x26CE, 0x26CE},
    {0x26D4, 0x26D4}, {0x26EA, 0x26EA}, {0x26F2, 0x26F3}, {0x26F5, 0x26F5},
    {0x26FA, 0x26FA}, {0x26FD, 0x26FD}, {0x2705, 0x2705}, {0x270A, 0x270B},
    {0x2728, 0x2728}, {0x274C, 0x274C}, {0x274E, 0x274E}, {0x2753, 0x2755},
    {0x2757, 0x2757}, {0x2795, 0x2797}, {0x27B0, 0x27B0}, {0x27BF, 0x27BF},
    {0x2B1B, 0x2B1C}, {0x2B50, 0x2B50}, {0x2B55, 0x2B55}, {0x2E80, 0x303E},
    {0x3041, 0x33FF}, {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xA000, 0xA4CF},
    {0xA960, 0xA97F}, {0xAC00, 0xD7A3}, {0xF900, 0xFAFF}, {0xFE10, 0xFE19},
    {0xFE30, 0xFE52}, {0xFE54, 0xFE66}, {0xFE68, 0xFE6B}, {0xFF00, 0xFF60},
    {0xFFE0, 0xFFE6}, {0x16FE0, 0x16FE4}, {0x17000, 0x18AFF},
    {0x1B000, 0x1B2FF}, {0x1F004, 0x1F004}, {0x1F0CF, 0x1F0CF},
    {0x1F18E, 0x1F18E}, {0x1F191, 0x1F19A}, {0x1F200, 0x1F320},
    {0x1F32D, 0x1F335}, {0x1F337, 0x1F37C}, {0x1F37E, 0x1F393},
    {0x1F3A0, 0x1F3CA}, {0x1F3CF, 0x1F3D3}, {0x1F3E0, 0x1F3F0},
    {0x1F3F4, 0x1F3F4}, {0x1F3F8, 0x1F43E}, {0x1F440, 0x1F440},
    {0x1F442, 0x1F4FC}, {0x1F4FF, 0x1F53D}, {0x1F54B, 0x1F54E},
    {0x1F550, 0x1F567}, {0x1F57A, 0x1F57A}, {0x1F595, 0x1F596},
    {0x1F5A4, 0x1F5A4}, {0x1F5FB, 0x1F64F}, {0x1F680, 0x1F6C5},
    {0x1F6CC, 0x1F6CC}, {0x1F6D0, 0x1F6D2}, {0x1F6EB, 0x1F6EC},
    {0x1F6F4, 0x1F6FC}, {0x1F7E0, 0x1F7EB}, {0x1F90C, 0x1F9FF},
    {0x1FA70, 0x1FAFF}, {0x20000, 0x2FFFD}, {0x30000, 0x3FFFD},
};

static bool ds4r_in_ranges(const ds4r_range *t, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < t[mid].lo) hi = mid;
        else if (cp > t[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

int ds4r_wcwidth(uint32_t cp) {
    if (cp == 0) return 0;
    if (cp < 0x20 || (cp >= 0x7f && cp < 0xa0)) return 0;
    if (ds4r_in_ranges(ds4r_zero_ranges,
                       sizeof(ds4r_zero_ranges) / sizeof(ds4r_zero_ranges[0]),
                       cp))
        return 0;
    if (ds4r_in_ranges(ds4r_wide_ranges,
                       sizeof(ds4r_wide_ranges) / sizeof(ds4r_wide_ranges[0]),
                       cp))
        return 2;
    return 1;
}

static size_t ds4r_utf8_need(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}

/* Decode one codepoint and report the bytes it used.  Malformed input is
 * consumed one byte at a time and reported as U+FFFD. */
static size_t ds4r_utf8_decode(const char *s, size_t len, uint32_t *out) {
    unsigned char c = (unsigned char)s[0];
    size_t need = ds4r_utf8_need(c);
    if (need == 1 || need > len) {
        *out = c < 0x80 ? c : 0xFFFD;
        return 1;
    }
    uint32_t cp = (uint32_t)(c & (0xff >> (need + 1)));
    for (size_t i = 1; i < need; i++) {
        unsigned char cc = (unsigned char)s[i];
        if ((cc & 0xc0) != 0x80) {
            *out = 0xFFFD;
            return 1;
        }
        cp = (cp << 6) | (cc & 0x3f);
    }
    *out = cp;
    return need;
}

int ds4r_visible_width(const char *s, size_t len) {
    int w = 0;
    size_t i = 0;
    while (i < len) {
        if (s[i] == 0x1b) {
            size_t j = i + 1;
            if (j < len && s[j] == '[') {
                j++;
                while (j < len && (unsigned char)s[j] >= 0x20 &&
                       (unsigned char)s[j] <= 0x3f) j++;
                if (j < len) j++;
            } else if (j < len) {
                j++;
            }
            i = j;
            continue;
        }
        uint32_t cp = 0;
        i += ds4r_utf8_decode(s + i, len - i, &cp);
        w += ds4r_wcwidth(cp);
    }
    return w;
}

/* ============================================================================
 * Growable buffers
 * ============================================================================ */

typedef struct {
    char *p;
    size_t len;
    size_t cap;
    bool oom;
} ds4r_str;

static void ds4r_str_add(ds4r_str *s, const char *b, size_t n) {
    if (s->oom || !n) return;
    if (s->len + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap * 2 : 256;
        while (cap < s->len + n + 1) cap *= 2;
        char *np = realloc(s->p, cap);
        if (!np) {
            s->oom = true;
            return;
        }
        s->p = np;
        s->cap = cap;
    }
    memcpy(s->p + s->len, b, n);
    s->len += n;
    s->p[s->len] = '\0';
}

static void ds4r_str_addz(ds4r_str *s, const char *z) {
    ds4r_str_add(s, z, strlen(z));
}

static void ds4r_str_addc(ds4r_str *s, char c) {
    ds4r_str_add(s, &c, 1);
}

static void ds4r_str_pad(ds4r_str *s, int n) {
    for (int i = 0; i < n; i++) ds4r_str_addc(s, ' ');
}

static void ds4r_str_repeat(ds4r_str *s, const char *unit, int times) {
    for (int i = 0; i < times; i++) ds4r_str_addz(s, unit);
}

static void ds4r_str_free(ds4r_str *s) {
    free(s->p);
    memset(s, 0, sizeof(*s));
}

static bool ds4r_bytes_append(char **buf, size_t *len, size_t *cap,
                              const char *s, size_t n) {
    if (!n) return true;
    if (*len + n + 1 > *cap) {
        size_t want = *cap ? *cap * 2 : 1024;
        while (want < *len + n + 1) want *= 2;
        char *np = realloc(*buf, want);
        if (!np) return false;
        *buf = np;
        *cap = want;
    }
    memcpy(*buf + *len, s, n);
    *len += n;
    (*buf)[*len] = '\0';
    return true;
}

/* ============================================================================
 * Raw output and terminal attributes
 * ============================================================================ */

static void ds4r_out(ds4r *r, const char *s, size_t n) {
    if (!n) return;
    fwrite(s, 1, n, r->fp);
    r->last_output_newline = s[n - 1] == '\n';
}

static void ds4r_seq(ds4r *r, const char *s) {
    if (r->use_color) ds4r_out(r, s, strlen(s));
}

static void ds4r_reset_color(ds4r *r) {
    if (r->use_color && r->color_open) ds4r_out(r, DS4R_RESET, 4);
    r->color_open = false;
    r->attr_key = 0;
}

/* One bit per active attribute, so an attribute change in the middle of a line
 * repaints instead of silently keeping the previous escape. */
static unsigned ds4r_attr_key(const ds4r *r) {
    unsigned k = 0;
    if (r->in_think) k |= 1u;
    if (r->md_code_block) k |= 2u;
    if (r->md_inline_code) k |= 4u;
    if (r->md_bold || r->md_heading) k |= 8u;
    if (r->md_italic) k |= 16u;
    if (r->md_heading_underline) k |= 32u;
    return k;
}

/* Decorations the renderer draws itself: bullets, quote bars, rules, and table
 * frames. */
static const char *ds4r_deco_color(const ds4r *r) {
    return r->in_think ? DS4R_THINK_DIM : DS4R_GREY;
}

static void ds4r_set_attrs(ds4r *r) {
    if (r->in_think) {
        /* One combined sequence so the grey base survives every attribute. */
        char seq[16];
        size_t n = 0;
        seq[n++] = '\x1b';
        seq[n++] = '[';
        if (!r->md_code_block) {
            if (r->md_bold || r->md_heading) {
                seq[n++] = '1';
                seq[n++] = ';';
            }
            if (r->md_italic) {
                seq[n++] = '3';
                seq[n++] = ';';
            }
            if (r->md_heading_underline) {
                seq[n++] = '4';
                seq[n++] = ';';
            }
            if (r->md_inline_code) {
                seq[n++] = '2';
                seq[n++] = ';';
            }
        }
        seq[n++] = '9';
        seq[n++] = '0';
        seq[n++] = 'm';
        seq[n] = '\0';
        ds4r_seq(r, seq);
        return;
    }
    if (r->md_code_block) {
        ds4r_seq(r, "\x1b[38;5;75m");
        return;
    }
    if (r->md_inline_code) ds4r_seq(r, "\x1b[36m");
    if (r->md_bold || r->md_heading) ds4r_seq(r, "\x1b[1m");
    if (r->md_italic) ds4r_seq(r, "\x1b[3m");
    if (r->md_heading_underline) ds4r_seq(r, "\x1b[4m");
}

/* Emit one complete character with the attributes it must be shown in. */
static void ds4r_put(ds4r *r, const char *s, size_t n) {
    unsigned key = r->use_color ? ds4r_attr_key(r) : 0;
    if (key != r->attr_key) {
        ds4r_reset_color(r);
        if (key) {
            ds4r_set_attrs(r);
            r->color_open = true;
            r->attr_key = key;
        }
    }
    ds4r_out(r, s, n);
}

static void ds4r_flush_utf8(ds4r *r) {
    if (!r->utf8_pending_len) return;
    char buf[4];
    size_t n = r->utf8_pending_len;
    memcpy(buf, r->utf8_pending, n);
    r->utf8_pending_len = 0;
    r->utf8_pending_need = 0;
    ds4r_put(r, buf, n);
}

/* Hold incomplete UTF-8 sequences so an attribute change can never be written
 * between the bytes of one character. */
static void ds4r_write_char_raw(ds4r *r, char c) {
    if (!r->format_markdown) {
        ds4r_put(r, &c, 1);
        return;
    }
    unsigned char uc = (unsigned char)c;
    if (r->utf8_pending_len) {
        if ((uc & 0xc0) == 0x80 &&
            r->utf8_pending_len < sizeof(r->utf8_pending)) {
            r->utf8_pending[r->utf8_pending_len++] = c;
            if (r->utf8_pending_len == r->utf8_pending_need) ds4r_flush_utf8(r);
            return;
        }
        ds4r_flush_utf8(r);
    }
    size_t need = ds4r_utf8_need(uc);
    if (need == 1) {
        ds4r_put(r, &c, 1);
        return;
    }
    r->utf8_pending[0] = c;
    r->utf8_pending_len = 1;
    r->utf8_pending_need = need;
}

/* Bytes the renderer generates itself (rules, bullets, quote bars) carry their
 * own color, so any streaming attribute is closed first. */
static void ds4r_emit_styled(ds4r *r, const char *color, const char *text) {
    ds4r_flush_utf8(r);
    ds4r_reset_color(r);
    if (color) ds4r_seq(r, color);
    ds4r_out(r, text, strlen(text));
    if (color) ds4r_seq(r, DS4R_RESET);
}

/* Fence markers are shown without markdown attributes.  Inside a think block
 * they still carry the muted base color, so the fence does not flash at answer
 * brightness in the middle of dim text. */
static void ds4r_put_plain(ds4r *r, char c) {
    bool bold = r->md_bold;
    bool italic = r->md_italic;
    bool code = r->md_inline_code;
    bool block = r->md_code_block;
    bool heading = r->md_heading;
    bool underline = r->md_heading_underline;

    ds4r_flush_utf8(r);
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    r->md_heading = false;
    r->md_heading_underline = false;
    ds4r_put(r, &c, 1);
    r->md_bold = bold;
    r->md_italic = italic;
    r->md_inline_code = code;
    r->md_code_block = block;
    r->md_heading = heading;
    r->md_heading_underline = underline;
}

static int ds4r_columns(const ds4r *r) {
    if (r->cols_override > 0) return r->cols_override;
    struct winsize ws;
    int fd = r->fp ? fileno(r->fp) : -1;
    if (fd >= 0 && ioctl(fd, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return ws.ws_col;
    return 80;
}

/* ============================================================================
 * Fenced code block highlighting
 * ============================================================================
 *
 * Poor man's code highlighter inspired by antirez/kilo: a tiny language table
 * plus one line-oriented tokenizer for comments, strings, numbers, and
 * separator-bounded keywords.  Ported from the agent renderer. */

#define DS4R_HL_NORMAL   0
#define DS4R_HL_COMMENT  1
#define DS4R_HL_KEYWORD1 2
#define DS4R_HL_KEYWORD2 3
#define DS4R_HL_STRING   4
#define DS4R_HL_NUMBER   5

#define DS4R_SYNTAX_NUMBERS          (1u << 0)
#define DS4R_SYNTAX_STRINGS          (1u << 1)
#define DS4R_SYNTAX_BACKTICK_STRINGS (1u << 2)
#define DS4R_SYNTAX_CASE_INSENSITIVE (1u << 3)

struct ds4r_syntax {
    const char *name;
    const char *aliases;
    const char **keywords;
    const char *singleline_comments[3];
    const char *multiline_start;
    const char *multiline_end;
    unsigned flags;
};

static const char *ds4r_kw_generic[] = {
    "if","else","for","while","do","switch","case","default","break",
    "continue","return","try","catch","finally","throw","throws","class",
    "struct","enum","interface","trait","impl","fn","func","function",
    "def","lambda","let","var","const","static","public","private",
    "protected","import","include","from","export","package","module",
    "namespace","new","delete","async","await","yield","match","type",
    "true|","false|","null|","nil|","none|","None|","NULL|","void|",
    "int|","long|","float|","double|","char|","bool|","string|",
    "String|","usize|","isize|","u8|","u16|","u32|","u64|","i8|",
    "i16|","i32|","i64|",NULL
};

static const char *ds4r_kw_c[] = {
    "auto","break","case","continue","default","do","else","enum",
    "extern","for","goto","if","register","return","sizeof","static",
    "struct","switch","typedef","union","volatile","while",
    "alignas","alignof","and","and_eq","asm","bitand","bitor","class",
    "compl","constexpr","const_cast","decltype","delete","dynamic_cast",
    "explicit","export","false","friend","inline","mutable","namespace",
    "new","noexcept","not","not_eq","nullptr","operator","or","or_eq",
    "private","protected","public","reinterpret_cast","static_assert",
    "static_cast","template","this","thread_local","throw","true","try",
    "typeid","typename","virtual","xor","xor_eq",
    "NULL|","bool|","char|","const|","double|","float|","int|","long|",
    "short|","signed|","size_t|","ssize_t|","uint8_t|","uint16_t|",
    "uint32_t|","uint64_t|","unsigned|","void|",NULL
};

static const char *ds4r_kw_python[] = {
    "and","as","assert","async","await","break","case","class","continue",
    "def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","match","nonlocal","not","or","pass",
    "raise","return","try","while","with","yield",
    "False|","None|","True|","bool|","bytes|","dict|","float|","int|",
    "list|","object|","set|","str|","tuple|",NULL
};

static const char *ds4r_kw_js[] = {
    "async","await","break","case","catch","class","const","continue",
    "debugger","default","delete","do","else","export","extends",
    "finally","for","from","function","get","if","import","in",
    "instanceof","let","new","of","return","set","static","super",
    "switch","this","throw","try","typeof","var","void","while","with",
    "yield","abstract","as","declare","enum","implements","interface",
    "keyof","namespace","private","protected","public","readonly","type",
    "any|","boolean|","false|","never|","null|","number|","string|",
    "symbol|","true|","undefined|","unknown|","void|",NULL
};

static const char *ds4r_kw_java[] = {
    "abstract","assert","break","case","catch","class","const","continue",
    "default","do","else","enum","extends","final","finally","for","goto",
    "if","implements","import","instanceof","interface","native","new",
    "package","private","protected","public","return","static","strictfp",
    "super","switch","synchronized","this","throw","throws","transient",
    "try","volatile","while",
    "boolean|","byte|","char|","double|","false|","float|","int|","long|",
    "null|","short|","true|","void|",NULL
};

static const char *ds4r_kw_go[] = {
    "break","case","chan","const","continue","default","defer","else",
    "fallthrough","for","func","go","goto","if","import","interface",
    "map","package","range","return","select","struct","switch","type",
    "var","bool|","byte|","complex64|","complex128|","error|","false|",
    "float32|","float64|","int|","int8|","int16|","int32|","int64|",
    "nil|","rune|","string|","true|","uint|","uint8|","uint16|",
    "uint32|","uint64|","uintptr|",NULL
};

static const char *ds4r_kw_rust[] = {
    "as","async","await","break","const","continue","crate","dyn","else",
    "enum","extern","fn","for","if","impl","in","let","loop","match",
    "mod","move","mut","pub","ref","return","self","Self","static",
    "struct","super","trait","type","unsafe","use","where","while",
    "bool|","char|","false|","f32|","f64|","i8|","i16|","i32|","i64|",
    "i128|","isize|","str|","String|","true|","u8|","u16|","u32|",
    "u64|","u128|","usize|",NULL
};

static const char *ds4r_kw_shell[] = {
    "case","do","done","elif","else","esac","fi","for","function","if",
    "in","select","then","time","until","while","break","continue",
    "return","export","local","readonly","source","test","true|","false|",
    "echo|","printf|","cd|","pwd|","read|","set|","unset|","shift|",NULL
};

static const char *ds4r_kw_sql[] = {
    "add","alter","and","as","asc","between","by","case","check","column",
    "constraint","create","delete","desc","distinct","drop","else","end",
    "exists","foreign","from","group","having","in","index","insert",
    "into","is","join","key","left","like","limit","not","null","on",
    "or","order","outer","primary","references","right","select","set",
    "table","then","union","unique","update","values","view","where",
    "bigint|","boolean|","date|","decimal|","false|","int|","integer|",
    "numeric|","real|","text|","timestamp|","true|","varchar|",NULL
};

static const char *ds4r_kw_ruby[] = {
    "BEGIN","END","alias","and","begin","break","case","class","def",
    "defined?","do","else","elsif","end","ensure","for","if","in",
    "module","next","not","or","redo","rescue","retry","return","self",
    "super","then","undef","unless","until","when","while","yield",
    "false|","nil|","true|",NULL
};

static const char *ds4r_kw_swift[] = {
    "actor","as","associatedtype","async","await","break","case","catch",
    "class","continue","default","defer","do","else","enum","extension",
    "fallthrough","for","func","guard","if","import","in","init","inout",
    "is","let","nonisolated","operator","private","protocol","public",
    "repeat","return","self","Self","static","struct","subscript","super",
    "switch","throw","throws","try","typealias","var","where","while",
    "Any|","Bool|","Double|","false|","Float|","Int|","nil|","String|",
    "true|","Void|",NULL
};

static const char *ds4r_kw_kotlin[] = {
    "as","break","class","continue","do","else","false","for","fun","if",
    "in","interface","is","null","object","package","return","super",
    "this","throw","true","try","typealias","typeof","val","var","when",
    "while","actual","annotation","by","catch","companion","const",
    "constructor","crossinline","data","enum","expect","external","final",
    "finally","import","infix","init","inline","inner","internal","lateinit",
    "noinline","open","operator","out","override","private","protected",
    "public","reified","sealed","suspend","tailrec","vararg",
    "Any|","Boolean|","Byte|","Char|","Double|","Float|","Int|","Long|",
    "Short|","String|","Unit|",NULL
};

static const char *ds4r_kw_lua[] = {
    "and","break","do","else","elseif","end","false","for","function",
    "goto","if","in","local","nil","not","or","repeat","return","then",
    "true","until","while",NULL
};

static const char *ds4r_kw_html[] = {
    "a","body","button","div","doctype","form","h1","h2","h3","head",
    "html","input","label","li","link","main","meta","ol","option","p",
    "script","section","select","span","style","table","tbody","td","th",
    "thead","title","tr","ul","class|","href|","id|","name|","rel|",
    "src|","type|","value|",NULL
};

static const char *ds4r_kw_css[] = {
    "align-items","background","border","bottom","color","display","flex",
    "font","font-size","gap","grid","height","justify-content","left",
    "margin","max-width","min-width","padding","position","right","top",
    "transform","width","z-index","absolute|","auto|","block|","flex|",
    "grid|","hidden|","inline|","none|","relative|","solid|",NULL
};

static const ds4r_syntax ds4r_syntaxes[] = {
    {"generic", "", ds4r_kw_generic, {"//","#",NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS | DS4R_SYNTAX_BACKTICK_STRINGS},
    {"c", " c h cpp c++ cc cxx hpp hxx objc objective-c", ds4r_kw_c,
        {"//",NULL,NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"python", " py python py3", ds4r_kw_python, {"#",NULL,NULL}, NULL, NULL,
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"javascript", " js jsx javascript typescript ts tsx node mjs cjs",
        ds4r_kw_js, {"//",NULL,NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS | DS4R_SYNTAX_BACKTICK_STRINGS},
    {"java", " java", ds4r_kw_java, {"//",NULL,NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"go", " go golang", ds4r_kw_go, {"//",NULL,NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS | DS4R_SYNTAX_BACKTICK_STRINGS},
    {"rust", " rs rust", ds4r_kw_rust, {"//",NULL,NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"shell", " sh bash zsh shell fish ksh", ds4r_kw_shell, {"#",NULL,NULL},
        NULL, NULL,
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS | DS4R_SYNTAX_BACKTICK_STRINGS},
    {"sql", " sql postgres mysql sqlite", ds4r_kw_sql, {"--",NULL,NULL},
        "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS | DS4R_SYNTAX_CASE_INSENSITIVE},
    {"ruby", " rb ruby", ds4r_kw_ruby, {"#",NULL,NULL}, NULL, NULL,
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"swift", " swift", ds4r_kw_swift, {"//",NULL,NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"kotlin", " kt kts kotlin", ds4r_kw_kotlin, {"//",NULL,NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"lua", " lua", ds4r_kw_lua, {"--",NULL,NULL}, NULL, NULL,
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"html", " html htm xml svg", ds4r_kw_html, {NULL,NULL,NULL}, "<!--", "-->",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"css", " css scss sass", ds4r_kw_css, {NULL,NULL,NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"json", " json jsonc", NULL, {"//",NULL,NULL}, "/*", "*/",
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {"yaml", " yaml yml toml ini", NULL, {"#",NULL,NULL}, NULL, NULL,
        DS4R_SYNTAX_NUMBERS | DS4R_SYNTAX_STRINGS},
    {NULL, NULL, NULL, {NULL,NULL,NULL}, NULL, NULL, 0}
};

static bool ds4r_syntax_alias_match(const char *aliases, const char *lang) {
    if (!aliases || !lang || !lang[0]) return false;
    size_t llen = strlen(lang);
    const char *p = aliases;
    while (*p) {
        while (*p == ' ') p++;
        const char *start = p;
        while (*p && *p != ' ') p++;
        if ((size_t)(p - start) == llen && !strncasecmp(start, lang, llen))
            return true;
    }
    return false;
}

/* An empty or explicitly plain info string disables highlighting: those fences
 * usually carry ASCII diagrams or command output that must stay untouched. */
static const ds4r_syntax *ds4r_syntax_for_lang(const char *lang) {
    static const char *plain[] = {"text","txt","plain","none","ascii",
                                  "diagram","output","log",NULL};
    if (!lang || !lang[0]) return NULL;
    for (int i = 0; plain[i]; i++)
        if (!strcasecmp(plain[i], lang)) return NULL;
    for (const ds4r_syntax *s = ds4r_syntaxes; s->name; s++) {
        if (!strcasecmp(s->name, lang) ||
            ds4r_syntax_alias_match(s->aliases, lang))
            return s;
    }
    return &ds4r_syntaxes[0];
}

static bool ds4r_syntax_separator(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || isspace(uc) ||
           strchr(",.()+-/*=~%[]{}<>:;!&|^?", c) != NULL;
}

static bool ds4r_syntax_line_comment(const ds4r_syntax *syn, const char *p,
                                     const char *end) {
    for (int i = 0; i < 3 && syn->singleline_comments[i]; i++) {
        const char *m = syn->singleline_comments[i];
        size_t mlen = strlen(m);
        if (mlen && (size_t)(end - p) >= mlen && !strncmp(p, m, mlen))
            return true;
    }
    return false;
}

static int ds4r_syntax_color(int hl) {
    switch (hl) {
    case DS4R_HL_COMMENT: return 244;
    case DS4R_HL_KEYWORD1: return 214;
    case DS4R_HL_KEYWORD2: return 81;
    case DS4R_HL_STRING: return 150;
    case DS4R_HL_NUMBER: return 203;
    default: return 252;
    }
}

/* cur tracks the color already set, so a run of same-class tokens costs one
 * escape instead of one per character. */
static void ds4r_syntax_out(ds4r *r, ds4r_str *out, int *cur, int hl,
                            const char *s, size_t n) {
    if (!n) return;
    if (r->use_color && hl != *cur) {
        char seq[32];
        snprintf(seq, sizeof(seq), "\x1b[38;5;%dm", ds4r_syntax_color(hl));
        ds4r_str_addz(out, seq);
        *cur = hl;
    }
    ds4r_str_add(out, s, n);
}

static size_t ds4r_syntax_keyword_len(const char *kw, bool *secondary) {
    size_t len = strlen(kw);
    *secondary = len && kw[len - 1] == '|';
    return *secondary ? len - 1 : len;
}

static bool ds4r_syntax_match_keyword(const ds4r_syntax *syn, const char *p,
                                      const char *line_end, size_t *out_len,
                                      int *out_hl) {
    if (!syn->keywords) return false;
    for (int i = 0; syn->keywords[i]; i++) {
        bool secondary = false;
        size_t klen = ds4r_syntax_keyword_len(syn->keywords[i], &secondary);
        if ((size_t)(line_end - p) < klen) continue;
        bool match = (syn->flags & DS4R_SYNTAX_CASE_INSENSITIVE) ?
            !strncasecmp(p, syn->keywords[i], klen) :
            !strncmp(p, syn->keywords[i], klen);
        if (!match) continue;
        if (!ds4r_syntax_separator(p[klen])) continue;
        *out_len = klen;
        *out_hl = secondary ? DS4R_HL_KEYWORD2 : DS4R_HL_KEYWORD1;
        return true;
    }
    return false;
}

static bool ds4r_syntax_number_start(const char *p, const char *line,
                                     bool prev_sep, int prev_hl) {
    unsigned char c = (unsigned char)*p;
    if (isdigit(c) && (prev_sep || prev_hl == DS4R_HL_NUMBER)) return true;
    if (*p == '.' && p > line && prev_hl == DS4R_HL_NUMBER) return true;
    return false;
}

static size_t ds4r_syntax_number_len(const char *p, const char *line_end) {
    const char *q = p;
    while (q < line_end) {
        unsigned char c = (unsigned char)*q;
        if (isalnum(c) || *q == '_' || *q == '.' || *q == '+' || *q == '-') q++;
        else break;
    }
    return (size_t)(q - p);
}

/* The caller guarantees line[len] == '\0', so keyword lookahead may read the
 * terminator when a keyword ends the line. */
static void ds4r_syntax_emit_line(ds4r *r, ds4r_str *out,
                                  const char *line, size_t len) {
    const ds4r_syntax *syn = r->md_syntax;
    const char *p = line;
    const char *end = line + len;
    bool prev_sep = true;
    int prev_hl = DS4R_HL_NORMAL;
    int cur = -1;

    while (p < end) {
        if (r->md_code_in_ml_comment) {
            const char *mce = syn->multiline_end;
            if (mce && *mce) {
                size_t mlen = strlen(mce);
                const char *q = p;
                while (q < end && ((size_t)(end - q) < mlen ||
                       strncmp(q, mce, mlen))) q++;
                if (q < end) {
                    q += mlen;
                    ds4r_syntax_out(r, out, &cur, DS4R_HL_COMMENT, p, (size_t)(q - p));
                    p = q;
                    r->md_code_in_ml_comment = false;
                    prev_sep = true;
                    prev_hl = DS4R_HL_COMMENT;
                    continue;
                }
            }
            ds4r_syntax_out(r, out, &cur, DS4R_HL_COMMENT, p, (size_t)(end - p));
            break;
        }

        if (ds4r_syntax_line_comment(syn, p, end)) {
            ds4r_syntax_out(r, out, &cur, DS4R_HL_COMMENT, p, (size_t)(end - p));
            break;
        }

        if (syn->multiline_start && syn->multiline_end &&
            (size_t)(end - p) >= strlen(syn->multiline_start) &&
            !strncmp(p, syn->multiline_start, strlen(syn->multiline_start))) {
            size_t mlen = strlen(syn->multiline_start);
            const char *q = p + mlen;
            size_t elen = strlen(syn->multiline_end);
            while (q < end && ((size_t)(end - q) < elen ||
                   strncmp(q, syn->multiline_end, elen))) q++;
            if (q < end) q += elen;
            else r->md_code_in_ml_comment = true;
            ds4r_syntax_out(r, out, &cur, DS4R_HL_COMMENT, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = DS4R_HL_COMMENT;
            continue;
        }

        if ((syn->flags & DS4R_SYNTAX_STRINGS) &&
            (*p == '"' || *p == '\'' ||
             ((syn->flags & DS4R_SYNTAX_BACKTICK_STRINGS) && *p == '`'))) {
            int quote = *p;
            const char *q = p + 1;
            while (q < end) {
                if (*q == '\\' && q + 1 < end) {
                    q += 2;
                    continue;
                }
                q++;
                if (q[-1] == quote) break;
            }
            ds4r_syntax_out(r, out, &cur, DS4R_HL_STRING, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = DS4R_HL_STRING;
            continue;
        }

        if ((syn->flags & DS4R_SYNTAX_NUMBERS) &&
            ds4r_syntax_number_start(p, line, prev_sep, prev_hl)) {
            size_t nlen = ds4r_syntax_number_len(p, end);
            ds4r_syntax_out(r, out, &cur, DS4R_HL_NUMBER, p, nlen);
            p += nlen;
            prev_sep = false;
            prev_hl = DS4R_HL_NUMBER;
            continue;
        }

        if (prev_sep) {
            size_t klen = 0;
            int khl = DS4R_HL_NORMAL;
            if (ds4r_syntax_match_keyword(syn, p, end, &klen, &khl)) {
                ds4r_syntax_out(r, out, &cur, khl, p, klen);
                p += klen;
                prev_sep = false;
                prev_hl = khl;
                continue;
            }
        }

        ds4r_syntax_out(r, out, &cur, DS4R_HL_NORMAL, p, 1);
        prev_sep = ds4r_syntax_separator(*p);
        prev_hl = DS4R_HL_NORMAL;
        p++;
    }
    if (r->use_color && cur >= 0) ds4r_str_addz(out, DS4R_RESET);
}

/* ============================================================================
 * Code blocks
 * ============================================================================ */

static void ds4r_code_line_append(ds4r *r, char c) {
    if (!ds4r_bytes_append(&r->code_line, &r->code_line_len,
                           &r->code_line_cap, &c, 1))
        ds4r_put(r, &c, 1);      /* out of memory: stream the byte instead */
}

/* Code lines are held until their newline arrives so the highlighter sees a
 * whole line.  Nothing is rewritten afterwards, so the terminal never gets a
 * repaint and fence content stays byte exact apart from color. */
static void ds4r_code_emit_line(ds4r *r, bool with_newline) {
    ds4r_flush_utf8(r);
    ds4r_reset_color(r);
    if (r->code_line_len) {
        if (r->md_syntax && !r->in_think) {
            ds4r_str out = {0};
            ds4r_syntax_emit_line(r, &out, r->code_line, r->code_line_len);
            if (!out.oom && out.p) ds4r_out(r, out.p, out.len);
            else ds4r_out(r, r->code_line, r->code_line_len);
            ds4r_str_free(&out);
        } else {
            ds4r_seq(r, r->in_think ? DS4R_THINK : "\x1b[38;5;75m");
            ds4r_out(r, r->code_line, r->code_line_len);
            ds4r_seq(r, DS4R_RESET);
        }
    }
    r->code_line_len = 0;
    if (r->code_line) r->code_line[0] = '\0';
    if (with_newline) {
        ds4r_out(r, "\n", 1);
        r->md_code_line_start = true;
    }
}

static void ds4r_code_byte(ds4r *r, char c) {
    if (c == '\n') {
        ds4r_code_emit_line(r, true);
        return;
    }
    ds4r_code_line_append(r, c);
    if (c != ' ' && c != '\t' && c != '\r') r->md_code_line_start = false;
}

static void ds4r_code_begin(ds4r *r) {
    ds4r_flush_utf8(r);
    ds4r_reset_color(r);
    r->md_code_block = true;
    r->md_inline_code = false;
    r->md_fence_info = true;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->code_line_len = 0;
}

static void ds4r_code_end(ds4r *r) {
    bool only_space = true;
    for (size_t i = 0; i < r->code_line_len; i++) {
        if (r->code_line[i] != ' ' && r->code_line[i] != '\t' &&
            r->code_line[i] != '\r') {
            only_space = false;
            break;
        }
    }
    if (r->code_line_len && !only_space) ds4r_code_emit_line(r, false);
    else r->code_line_len = 0;
    r->md_code_block = false;
    r->md_inline_code = false;
    r->md_fence_info = false;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
}

/* ============================================================================
 * Inline markdown
 * ============================================================================ */

static void ds4r_md_clear_mark(ds4r *r) {
    r->md_mark = DS4R_MARK_NONE;
    r->md_mark_len = 0;
}

static void ds4r_md_emit_mark_literals(ds4r *r) {
    char c;
    if (r->md_mark == DS4R_MARK_STAR) c = '*';
    else if (r->md_mark == DS4R_MARK_BACKTICK) c = '`';
    else return;
    size_t count = r->md_mark_len;
    ds4r_md_clear_mark(r);
    for (size_t i = 0; i < count; i++) {
        if (r->md_code_block) ds4r_code_byte(r, c);
        else ds4r_write_char_raw(r, c);
    }
}

static void ds4r_md_commit_backticks(ds4r *r) {
    size_t count = r->md_mark_len;
    ds4r_md_clear_mark(r);
    if (count >= 3) {
        bool was_code = r->md_code_block;
        /* Flush the partial code line before the closing fence is printed,
         * otherwise its buffered indentation would follow the marker. */
        if (was_code) ds4r_code_end(r);
        for (size_t i = 0; i < count; i++) ds4r_put_plain(r, '`');
        if (!was_code) ds4r_code_begin(r);
        return;
    }
    if (r->md_code_block) {
        for (size_t i = 0; i < count; i++) ds4r_code_byte(r, '`');
        return;
    }
    /* Support both `code` and ``code``. */
    r->md_inline_code = !r->md_inline_code;
}

static bool ds4r_space_byte(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static void ds4r_md_end_line(ds4r *r) {
    r->md_heading = false;
    r->md_heading_underline = false;
    ds4r_write_char_raw(r, '\n');
    ds4r_flush_utf8(r);
    ds4r_line_begin(r);
}

/* Consume one byte of markdown-aware output.  Backticks and stars are held
 * until the parser knows whether they form a marker; ordinary text goes to the
 * raw writer with the current attributes. */
static void ds4r_md_feed(ds4r *r, char c) {
    if (r->md_fence_info) {
        if (c == '\n') {
            r->md_fence_lang[r->md_fence_lang_len] = '\0';
            r->md_syntax = r->in_think ? NULL :
                ds4r_syntax_for_lang(r->md_fence_lang);
            ds4r_put_plain(r, '\n');
            r->md_fence_info = false;
            r->md_code_line_start = true;
        } else {
            unsigned char uc = (unsigned char)c;
            if (r->md_fence_lang_len + 1 < sizeof(r->md_fence_lang) &&
                (isalnum(uc) || c == '_' || c == '-' || c == '+' || c == '#'))
                r->md_fence_lang[r->md_fence_lang_len++] = c;
            ds4r_put_plain(r, c);
        }
        return;
    }

    if (r->md_mark == DS4R_MARK_BACKTICK) {
        if (c == '`') {
            r->md_mark_len++;
            return;
        }
        ds4r_md_commit_backticks(r);
        ds4r_md_feed(r, c);
        return;
    }

    if (r->md_mark == DS4R_MARK_STAR) {
        ds4r_md_clear_mark(r);
        if (!r->md_inline_code && !r->md_code_block && c == '*') {
            r->md_bold = !r->md_bold;
            return;
        }
        /* "a * b" keeps its literal star: opening emphasis may not be followed
         * by whitespace. */
        if (!r->md_inline_code && !r->md_code_block &&
            (r->md_italic || !ds4r_space_byte(c))) {
            r->md_italic = !r->md_italic;
            ds4r_md_feed(r, c);
            return;
        }
        ds4r_write_char_raw(r, '*');
        ds4r_md_feed(r, c);
        return;
    }

    if (c == '`' && (!r->md_code_block || r->md_code_line_start)) {
        r->md_mark = DS4R_MARK_BACKTICK;
        r->md_mark_len = 1;
        return;
    }
    if (r->md_code_block) {
        ds4r_code_byte(r, c);
        return;
    }
    if (!r->md_inline_code && c == '*') {
        r->md_mark = DS4R_MARK_STAR;
        r->md_mark_len = 1;
        return;
    }
    if (c == '\n') {
        ds4r_md_end_line(r);
        return;
    }
    ds4r_write_char_raw(r, c);
}

/* ============================================================================
 * Tables
 * ============================================================================ */

typedef enum {
    DS4R_ALIGN_LEFT = 0,
    DS4R_ALIGN_CENTER,
    DS4R_ALIGN_RIGHT,
} ds4r_align;

/* Split one buffered row into trimmed cells.  Returns the cell count and, when
 * out is not NULL, stores freshly allocated unescaped strings. */
static int ds4r_row_cells(const char *line, size_t len, char **out, int max) {
    const char *p = line;
    const char *end = line + len;
    while (end > p && (end[-1] == '\n' || end[-1] == '\r' ||
                       end[-1] == ' ' || end[-1] == '\t')) end--;
    while (p < end && (*p == ' ' || *p == '\t')) p++;
    if (p < end && *p == '|') p++;
    if (end > p && end[-1] == '|') {
        int backslashes = 0;
        const char *q = end - 1;
        while (q > p && q[-1] == '\\') {
            backslashes++;
            q--;
        }
        if ((backslashes % 2) == 0) end--;      /* unescaped closing pipe */
    }

    int n = 0;
    const char *cell = p;
    bool escaped = false;
    for (const char *q = p; q <= end; q++) {
        bool split = q == end;
        if (!split) {
            if (escaped) {
                escaped = false;
                continue;
            }
            if (*q == '\\') {
                escaped = true;
                continue;
            }
            split = *q == '|';
        }
        if (!split) continue;
        if (n >= max) return -1;
        if (out) {
            const char *cs = cell;
            const char *ce = q;
            while (cs < ce && (*cs == ' ' || *cs == '\t')) cs++;
            while (ce > cs && (ce[-1] == ' ' || ce[-1] == '\t')) ce--;
            char *dst = malloc((size_t)(ce - cs) + 1);
            if (!dst) return -1;
            size_t k = 0;
            for (const char *s = cs; s < ce; s++) {
                if (*s == '\\' && s + 1 < ce && s[1] == '|') s++;
                dst[k++] = *s;
            }
            dst[k] = '\0';
            out[n] = dst;
        }
        n++;
        cell = q + 1;
    }
    return n;
}

static bool ds4r_is_align_cell(const char *s, ds4r_align *out) {
    const char *p = s;
    bool left = false, right = false;
    int dashes = 0;
    if (*p == ':') {
        left = true;
        p++;
    }
    while (*p == '-') {
        dashes++;
        p++;
    }
    if (*p == ':') {
        right = true;
        p++;
    }
    if (*p != '\0' || dashes < 1) return false;
    *out = left && right ? DS4R_ALIGN_CENTER :
           right ? DS4R_ALIGN_RIGHT : DS4R_ALIGN_LEFT;
    return true;
}

/* Render one cell with inline markdown, stopping after limit visible columns.
 * With out == NULL the function only measures.  bold_init keeps a header cell
 * bold across nested **markers**. */
static int ds4r_cell_scan(const char *s, int limit, bool ellipsis,
                          bool bold_init, bool color, bool muted,
                          ds4r_str *out) {
    bool bold = bold_init;
    bool italic = false;
    bool code = false;
    bool cut = false;
    int w = 0;
    size_t len = strlen(s);
    size_t i = 0;

    if (out && color && bold) ds4r_str_addz(out, "\x1b[1m");
    while (i < len) {
        if (s[i] == '`') {
            code = !code;
            if (out && color)
                ds4r_str_addz(out, muted ? (code ? "\x1b[2m" : "\x1b[22m") :
                                           (code ? "\x1b[36m" : "\x1b[39m"));
            i++;
            continue;
        }
        if (s[i] == '*' && !code) {
            if (i + 1 < len && s[i + 1] == '*') {
                bold = !bold;
                if (out && color)
                    ds4r_str_addz(out, bold ? "\x1b[1m" : "\x1b[22m");
                i += 2;
                continue;
            }
            italic = !italic;
            if (out && color)
                ds4r_str_addz(out, italic ? "\x1b[3m" : "\x1b[23m");
            i++;
            continue;
        }
        if (s[i] == '\\' && i + 1 < len && !isalnum((unsigned char)s[i + 1]))
            i++;
        uint32_t cp = 0;
        size_t n = ds4r_utf8_decode(s + i, len - i, &cp);
        int cw = ds4r_wcwidth(cp);
        if (w + cw > limit) {
            cut = true;
            break;
        }
        if (out) ds4r_str_add(out, s + i, n);
        w += cw;
        i += n;
    }
    if (out && color) {
        if (code) ds4r_str_addz(out, muted ? "\x1b[22m" : "\x1b[39m");
        if (italic) ds4r_str_addz(out, "\x1b[23m");
        if (bold) ds4r_str_addz(out, "\x1b[22m");
    }
    if (cut && ellipsis) {
        if (out) ds4r_str_addz(out, "\xe2\x80\xa6");
        w += 1;
    }
    return w;
}

static void ds4r_table_border(ds4r *r, ds4r_str *out, const int *colw,
                              int ncols, int pad, const char *left,
                              const char *mid, const char *right) {
    if (r->use_color) ds4r_str_addz(out, ds4r_deco_color(r));
    ds4r_str_addz(out, left);
    for (int c = 0; c < ncols; c++) {
        ds4r_str_repeat(out, "\xe2\x94\x80", colw[c] + 2 * pad);
        ds4r_str_addz(out, c + 1 == ncols ? right : mid);
    }
    if (r->use_color) ds4r_str_addz(out, DS4R_RESET);
    ds4r_str_addc(out, '\n');
}

static void ds4r_table_bar(ds4r *r, ds4r_str *out) {
    if (r->use_color) ds4r_str_addz(out, ds4r_deco_color(r));
    ds4r_str_addz(out, "\xe2\x94\x82");
    if (r->use_color) ds4r_str_addz(out, DS4R_RESET);
}

static void ds4r_table_row(ds4r *r, ds4r_str *out, char **cells, int ncells,
                           const int *colw, const ds4r_align *align, int ncols,
                           int pad, bool header) {
    for (int c = 0; c < ncols; c++) {
        const char *text = (c < ncells && cells[c]) ? cells[c] : "";
        ds4r_str cell = {0};
        bool muted = r->in_think;
        int want = ds4r_cell_scan(text, INT_MAX, false, header, false, muted,
                                  NULL);
        int w = want <= colw[c] ?
            ds4r_cell_scan(text, INT_MAX, false, header, r->use_color, muted,
                           &cell) :
            ds4r_cell_scan(text, colw[c] - 1, true, header, r->use_color, muted,
                           &cell);
        int slack = colw[c] - w;
        if (slack < 0) slack = 0;
        int before = align[c] == DS4R_ALIGN_RIGHT ? slack :
                     align[c] == DS4R_ALIGN_CENTER ? slack / 2 : 0;

        ds4r_table_bar(r, out);
        ds4r_str_pad(out, pad + before);
        if (muted && r->use_color) ds4r_str_addz(out, DS4R_THINK);
        if (cell.p) ds4r_str_add(out, cell.p, cell.len);
        if (muted && r->use_color) ds4r_str_addz(out, DS4R_RESET);
        ds4r_str_pad(out, slack - before + pad);
        ds4r_str_free(&cell);
    }
    ds4r_table_bar(r, out);
    ds4r_str_addc(out, '\n');
}

/* Lay out and print the buffered rows.  Returns false when the buffer is not a
 * usable table, in which case the caller prints the original text. */
static bool ds4r_table_render(ds4r *r) {
    const char *buf = r->tbl;
    size_t len = r->tbl_len;
    bool ok = false;
    size_t nrows = 0;
    size_t *line_off = NULL;
    size_t *line_len = NULL;
    int *rowcells = NULL;
    char **cells = NULL;
    int *colw = NULL;
    ds4r_align *align = NULL;
    int ncols = 0;

    for (size_t i = 0; i < len; i++)
        if (buf[i] == '\n') nrows++;
    if (len && buf[len - 1] != '\n') nrows++;
    if (nrows < 2) return false;

    line_off = malloc(nrows * sizeof(*line_off));
    line_len = malloc(nrows * sizeof(*line_len));
    rowcells = malloc(nrows * sizeof(*rowcells));
    if (!line_off || !line_len || !rowcells) goto done;

    size_t row = 0;
    size_t start = 0;
    for (size_t i = 0; i <= len; i++) {
        if (i == len || buf[i] == '\n') {
            if (i == len && i == start) break;
            line_off[row] = start;
            line_len[row] = i - start;
            row++;
            start = i + 1;
            if (row == nrows) break;
        }
    }
    nrows = row;
    if (nrows < 2) goto done;

    for (size_t i = 0; i < nrows; i++) {
        rowcells[i] = ds4r_row_cells(buf + line_off[i], line_len[i], NULL,
                                     DS4R_TBL_MAX_COLS);
        if (rowcells[i] < 1) goto done;
        if (rowcells[i] > ncols) ncols = rowcells[i];
    }

    cells = calloc(nrows * (size_t)ncols, sizeof(*cells));
    colw = calloc((size_t)ncols, sizeof(*colw));
    align = calloc((size_t)ncols, sizeof(*align));
    if (!cells || !colw || !align) goto done;
    for (size_t i = 0; i < nrows; i++) {
        if (ds4r_row_cells(buf + line_off[i], line_len[i],
                           cells + i * (size_t)ncols, ncols) < 1)
            goto done;
    }

    /* The second row must be the alignment row, as GitHub Markdown requires. */
    for (int c = 0; c < rowcells[1]; c++) {
        ds4r_align a = DS4R_ALIGN_LEFT;
        if (!ds4r_is_align_cell(cells[(size_t)ncols + (size_t)c], &a))
            goto done;
        if (c < ncols) align[c] = a;
    }

    for (size_t i = 0; i < nrows; i++) {
        if (i == 1) continue;
        for (int c = 0; c < ncols; c++) {
            const char *text = cells[i * (size_t)ncols + (size_t)c];
            if (!text) continue;
            int w = ds4r_cell_scan(text, INT_MAX, false, false, false, false,
                                   NULL);
            if (w > colw[c]) colw[c] = w;
        }
    }
    for (int c = 0; c < ncols; c++)
        if (colw[c] < 1) colw[c] = 1;

    /* Shrink to the terminal: first the cell padding, then the widest column. */
    int cols = ds4r_columns(r);
    int pad = 1;
    for (;;) {
        int total = ncols + 1;
        for (int c = 0; c < ncols; c++) total += colw[c] + 2 * pad;
        if (total <= cols) break;
        if (pad > 0) {
            pad = 0;
            continue;
        }
        int widest = 0;
        for (int c = 1; c < ncols; c++)
            if (colw[c] > colw[widest]) widest = c;
        if (colw[widest] <= 1) goto done;   /* cannot fit: print the source */
        colw[widest]--;
    }

    ds4r_str out = {0};
    ds4r_table_border(r, &out, colw, ncols, pad,
                      "\xe2\x94\x8c", "\xe2\x94\xac", "\xe2\x94\x90");
    ds4r_table_row(r, &out, cells, rowcells[0], colw, align, ncols, pad, true);
    ds4r_table_border(r, &out, colw, ncols, pad,
                      "\xe2\x94\x9c", "\xe2\x94\xbc", "\xe2\x94\xa4");
    for (size_t i = 2; i < nrows; i++)
        ds4r_table_row(r, &out, cells + i * (size_t)ncols, rowcells[i],
                       colw, align, ncols, pad, false);
    ds4r_table_border(r, &out, colw, ncols, pad,
                      "\xe2\x94\x94", "\xe2\x94\xb4", "\xe2\x94\x98");
    if (!out.oom && out.p) {
        ds4r_flush_utf8(r);
        ds4r_reset_color(r);
        ds4r_out(r, out.p, out.len);
        ok = true;
    }
    ds4r_str_free(&out);

done:
    if (cells) {
        for (size_t i = 0; i < nrows * (size_t)ncols; i++) free(cells[i]);
        free(cells);
    }
    free(line_off);
    free(line_len);
    free(rowcells);
    free(colw);
    free(align);
    return ok;
}

static void ds4r_table_reset(ds4r *r) {
    r->tbl_len = 0;
    r->tbl_lines = 0;
    r->tbl_scan = false;
    r->tbl_raw = false;
    if (r->tbl) r->tbl[0] = '\0';
}

static void ds4r_table_flush(ds4r *r) {
    if (!r->tbl_raw && r->tbl_len && !ds4r_table_render(r)) {
        ds4r_flush_utf8(r);
        ds4r_reset_color(r);
        ds4r_out(r, r->tbl, r->tbl_len);
    }
    ds4r_table_reset(r);
    r->line_mode = DS4R_LINE_OFF;
    r->line_len = 0;
    r->indent_len = 0;
    r->marker_run = 0;
    ds4r_line_begin(r);
}

static void ds4r_table_append(ds4r *r, const char *s, size_t n) {
    if (r->tbl_raw) {
        ds4r_out(r, s, n);
        return;
    }
    if (r->tbl_len + n > DS4R_TBL_MAX_BYTES ||
        r->tbl_lines >= DS4R_TBL_MAX_LINES ||
        !ds4r_bytes_append(&r->tbl, &r->tbl_len, &r->tbl_cap, s, n)) {
        /* Too large to lay out: print what was buffered and pass the rest of
         * the table through unchanged. */
        ds4r_flush_utf8(r);
        ds4r_reset_color(r);
        if (r->tbl_len) ds4r_out(r, r->tbl, r->tbl_len);
        r->tbl_len = 0;
        r->tbl_raw = true;
        ds4r_out(r, s, n);
    }
}

/* Buffer consecutive rows.  The mode ends on the first line that does not
 * start with a pipe, which is then re-fed through the normal path. */
static void ds4r_table_byte(ds4r *r, char c) {
    if (r->tbl_scan) {
        if ((c == ' ' || c == '\t') && r->line_len + 1 < DS4R_LINE_BUF) {
            r->line_buf[r->line_len++] = c;
            return;
        }
        if (c == '|') {
            r->tbl_scan = false;
            ds4r_table_append(r, r->line_buf, r->line_len);
            r->line_len = 0;
            ds4r_table_append(r, "|", 1);
            return;
        }
        char held[DS4R_LINE_BUF];
        size_t n = r->line_len;
        memcpy(held, r->line_buf, n);
        ds4r_table_flush(r);
        for (size_t i = 0; i < n; i++) ds4r_render_byte(r, held[i]);
        ds4r_render_byte(r, c);
        return;
    }
    ds4r_table_append(r, &c, 1);
    if (c == '\n') {
        r->tbl_lines++;
        r->tbl_scan = true;
        r->line_len = 0;
    }
}

/* ============================================================================
 * Line-start dispatch
 * ============================================================================ */

static void ds4r_line_begin(ds4r *r) {
    if (!r->format_markdown || r->md_code_block) {
        r->line_mode = DS4R_LINE_OFF;
        return;
    }
    r->line_mode = DS4R_LINE_SCAN;
    r->scan = DS4R_SCAN_INDENT;
    r->line_len = 0;
    r->indent_len = 0;
    r->marker_run = 0;
}

/* Give up on the classification: replay everything the scanner held back. */
static void ds4r_scan_flush(ds4r *r) {
    char head[DS4R_LINE_BUF];
    char tail[DS4R_LINE_BUF];
    size_t hn = r->indent_len;
    size_t tn = r->line_len - r->indent_len;
    size_t run = r->marker_run;
    char mc = r->marker_ch;

    memcpy(head, r->line_buf, hn);
    memcpy(tail, r->line_buf + hn, tn);
    r->line_mode = DS4R_LINE_OFF;
    r->line_len = 0;
    r->indent_len = 0;
    r->marker_run = 0;

    for (size_t i = 0; i < hn; i++) ds4r_render_byte(r, head[i]);
    for (size_t i = 0; i < run; i++) ds4r_render_byte(r, mc);
    for (size_t i = 0; i < tn; i++) ds4r_render_byte(r, tail[i]);
}

/* Emit the indentation of a line whose marker the renderer replaces. */
static void ds4r_scan_emit_indent(ds4r *r) {
    for (size_t i = 0; i < r->indent_len; i++)
        ds4r_write_char_raw(r, r->line_buf[i]);
}

static void ds4r_scan_done(ds4r *r) {
    r->line_mode = DS4R_LINE_OFF;
    r->line_len = 0;
    r->indent_len = 0;
    r->marker_run = 0;
}

static void ds4r_scan_rule(ds4r *r) {
    int cols = ds4r_columns(r);
    if (cols > 80) cols = 80;
    if (cols < 4) cols = 4;
    ds4r_str out = {0};
    ds4r_str_repeat(&out, "\xe2\x94\x80", cols);
    if (!out.oom && out.p) ds4r_emit_styled(r, ds4r_deco_color(r), out.p);
    ds4r_str_free(&out);
}

static void ds4r_scan_byte(ds4r *r, char c) {
    switch (r->scan) {
    case DS4R_SCAN_INDENT:
        if (c == ' ' || c == '\t') {
            if (r->line_len + 16 < DS4R_LINE_BUF) {
                r->line_buf[r->line_len++] = c;
                r->indent_len = r->line_len;
                return;
            }
            break;
        }
        if (c == '|') {
            r->line_mode = DS4R_LINE_TABLE;
            r->tbl_scan = false;
            ds4r_table_append(r, r->line_buf, r->line_len);
            r->line_len = 0;
            r->indent_len = 0;
            ds4r_table_append(r, "|", 1);
            return;
        }
        if (c == '#') {
            r->scan = DS4R_SCAN_HASH;
            r->line_buf[r->line_len++] = c;
            return;
        }
        if (c == '-' || c == '*' || c == '+' || c == '_') {
            r->scan = DS4R_SCAN_MARKER;
            r->marker_ch = c;
            r->marker_run = 1;
            return;
        }
        if (c == '>') {
            ds4r_scan_emit_indent(r);
            ds4r_emit_styled(r, ds4r_deco_color(r), "\xe2\x94\x82 ");
            r->line_len = 0;
            r->indent_len = 0;
            r->scan = DS4R_SCAN_QUOTE;
            return;
        }
        if (c >= '0' && c <= '9') {
            r->scan = DS4R_SCAN_NUMBER;
            r->line_buf[r->line_len++] = c;
            return;
        }
        break;

    case DS4R_SCAN_HASH:
        if (c == '#' && r->line_len - r->indent_len < 6) {
            r->line_buf[r->line_len++] = c;
            return;
        }
        if (c == ' ') {
            ds4r_scan_emit_indent(r);
            r->md_heading = true;
            r->md_heading_underline = r->line_len - r->indent_len <= 2;
            ds4r_scan_done(r);
            return;
        }
        break;

    case DS4R_SCAN_MARKER:
        if (c == r->marker_ch) {
            r->marker_run++;
            return;
        }
        if (c == ' ' || c == '\t') {
            if (r->marker_run == 1 && r->marker_ch != '_') {
                ds4r_scan_emit_indent(r);
                ds4r_emit_styled(r, ds4r_deco_color(r), "\xe2\x80\xa2");
                ds4r_out(r, " ", 1);
                ds4r_scan_done(r);
                return;
            }
            if (r->marker_run >= 3 && r->line_len + 4 < DS4R_LINE_BUF) {
                r->scan = DS4R_SCAN_MARKER_TAIL;
                r->line_buf[r->line_len++] = c;
                return;
            }
            break;
        }
        if (c == '\n' && r->marker_run >= 3) {
            ds4r_scan_rule(r);
            ds4r_scan_done(r);
            ds4r_render_byte(r, '\n');
            return;
        }
        break;

    case DS4R_SCAN_MARKER_TAIL:
        if ((c == ' ' || c == '\t') && r->line_len + 4 < DS4R_LINE_BUF) {
            r->line_buf[r->line_len++] = c;
            return;
        }
        if (c == '\n') {
            ds4r_scan_rule(r);
            ds4r_scan_done(r);
            ds4r_render_byte(r, '\n');
            return;
        }
        break;

    case DS4R_SCAN_NUMBER:
        if (c >= '0' && c <= '9' && r->line_len + 6 < DS4R_LINE_BUF) {
            r->line_buf[r->line_len++] = c;
            return;
        }
        if (c == '.' || c == ')') {
            r->scan = DS4R_SCAN_NUMBER_DOT;
            r->line_buf[r->line_len++] = c;
            return;
        }
        break;

    case DS4R_SCAN_NUMBER_DOT:
        if (c == ' ') {
            char marker[DS4R_LINE_BUF];
            size_t n = r->line_len - r->indent_len;
            memcpy(marker, r->line_buf + r->indent_len, n);
            marker[n] = '\0';
            ds4r_scan_emit_indent(r);
            ds4r_emit_styled(r, ds4r_deco_color(r), marker);
            ds4r_out(r, " ", 1);
            ds4r_scan_done(r);
            return;
        }
        break;

    case DS4R_SCAN_QUOTE:
        r->scan = DS4R_SCAN_INDENT;
        if (c == ' ') return;    /* the space after '>' belongs to the marker */
        ds4r_scan_byte(r, c);
        return;
    }

    ds4r_scan_flush(r);
    ds4r_render_byte(r, c);
}

static void ds4r_render_byte(ds4r *r, char c) {
    if (!r->format_markdown) {
        ds4r_md_emit_mark_literals(r);
        ds4r_write_char_raw(r, c);
        return;
    }
    if (r->line_mode == DS4R_LINE_TABLE) {
        ds4r_table_byte(r, c);
        return;
    }
    if (r->line_mode == DS4R_LINE_SCAN) {
        ds4r_scan_byte(r, c);
        return;
    }
    ds4r_md_feed(r, c);
}

/* ============================================================================
 * <think> filtering and public entry points
 * ============================================================================ */

static bool ds4r_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool ds4r_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

/* Close every construct that is still open and print what was held back.  Used
 * at both think boundaries and at the end of a reply, so no table, fence, or
 * attribute can leak from thinking into the answer or the other way round. */
static void ds4r_md_flush_state(ds4r *r) {
    if (!r->format_markdown) return;
    if (r->line_mode == DS4R_LINE_SCAN) ds4r_scan_flush(r);
    if (r->line_mode == DS4R_LINE_TABLE) ds4r_table_flush(r);
    /* A closing fence can be the last thing the model emits, with no following
     * byte to push the pending backticks through. */
    if (r->md_mark == DS4R_MARK_BACKTICK && r->md_mark_len >= 3)
        ds4r_md_commit_backticks(r);
    else
        ds4r_md_emit_mark_literals(r);
    if (r->md_code_block && r->code_line_len) ds4r_code_emit_line(r, false);
    ds4r_flush_utf8(r);
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_heading = false;
    r->md_heading_underline = false;
    r->md_code_block = false;
    r->md_fence_info = false;
    r->md_code_line_start = false;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    ds4r_md_clear_mark(r);
    ds4r_table_reset(r);
    r->line_mode = DS4R_LINE_OFF;
}

/* Hide the think markers, grey the thinking text, and never emit a control tag
 * that is still split across model tokens. */
static void ds4r_think_process(ds4r *r, const char *text, size_t len,
                               bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = r->pending_len + len;
    char *buf = malloc(total ? total : 1);
    if (!buf) return;
    if (r->pending_len) memcpy(buf, r->pending, r->pending_len);
    if (len) memcpy(buf + r->pending_len, text, len);
    r->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        const size_t rem = total - i;
        if (ds4r_has_prefix(cur, rem, think_open)) {
            ds4r_md_flush_state(r);
            ds4r_flush_utf8(r);
            ds4r_reset_color(r);
            r->in_think = true;
            ds4r_line_begin(r);
            i += strlen(think_open);
            continue;
        }
        if (ds4r_has_prefix(cur, rem, think_close)) {
            /* Flush while still inside the block so anything held back is
             * printed with the muted palette, then start the answer clean. */
            ds4r_md_flush_state(r);
            ds4r_flush_utf8(r);
            r->in_think = false;
            ds4r_reset_color(r);
            if (!r->last_output_newline) ds4r_out(r, "\n", 1);
            ds4r_line_begin(r);
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (ds4r_is_partial_prefix(cur, rem, think_open) ||
             ds4r_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(r->pending)) {
                memcpy(r->pending, cur, rem);
                r->pending_len = rem;
            }
            break;
        }
        ds4r_render_byte(r, cur[0]);
        i++;
    }

    free(buf);
}

void ds4r_init(ds4r *r, FILE *fp, bool color, bool format_thinking) {
    memset(r, 0, sizeof(*r));
    r->fp = fp;
    r->use_color = color;
    r->format_markdown = color;
    r->format_thinking = format_thinking;
    r->last_output_newline = true;
    ds4r_line_begin(r);
}

void ds4r_set_markdown(ds4r *r, bool enabled) {
    r->format_markdown = enabled && r->use_color;
    ds4r_line_begin(r);
}

void ds4r_set_in_think(ds4r *r, bool in_think) {
    r->in_think = in_think;
    ds4r_line_begin(r);
}

void ds4r_set_columns(ds4r *r, int cols) {
    r->cols_override = cols;
}

bool ds4r_at_line_start(const ds4r *r) {
    return r->last_output_newline;
}

void ds4r_newline(ds4r *r) {
    ds4r_flush_utf8(r);
    ds4r_reset_color(r);
    ds4r_out(r, "\n", 1);
    ds4r_line_begin(r);
}

void ds4r_write(ds4r *r, const char *text, size_t len) {
    if (r->format_thinking) {
        ds4r_think_process(r, text, len, false);
        return;
    }
    if (!r->format_markdown) {
        if (len) {
            fwrite(text, 1, len, r->fp);
            r->last_output_newline = text[len - 1] == '\n';
        }
        return;
    }
    for (size_t i = 0; i < len; i++) ds4r_render_byte(r, text[i]);
}

void ds4r_finish(ds4r *r) {
    if (r->format_thinking) ds4r_think_process(r, NULL, 0, true);
    ds4r_md_flush_state(r);
    ds4r_flush_utf8(r);
    ds4r_reset_color(r);
    fflush(r->fp);
}

void ds4r_free(ds4r *r) {
    free(r->code_line);
    r->code_line = NULL;
    r->code_line_len = 0;
    r->code_line_cap = 0;
    free(r->tbl);
    r->tbl = NULL;
    r->tbl_len = 0;
    r->tbl_cap = 0;
}
