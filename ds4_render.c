/* Streaming Markdown renderer for terminal chat output.  See ds4_render.h.
 *
 * ds4r_write() strips the <think> markers and feeds the rest to
 * ds4r_md_feed(), which handles the inline constructs: bold, italic, inline
 * code, and fenced code blocks.  Bytes that cannot be classified yet are held
 * back: marker runs and partial UTF-8 sequences.
 *
 * The inline machine, the UTF-8 accumulator, and the fenced-block highlighter
 * follow the agent renderer in ds4_agent.c. */

#include "ds4_render.h"

#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define DS4R_RESET "\x1b[0m"

static void ds4r_render_byte(ds4r *r, char c);
static void ds4r_md_feed(ds4r *r, char c);

static size_t ds4r_utf8_need(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
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
    if (r->md_bold) k |= 8u;
    if (r->md_italic) k |= 16u;
    return k;
}

static void ds4r_set_attrs(ds4r *r) {
    if (r->in_think) {
        ds4r_seq(r, "\x1b[90m");
        return;
    }
    if (r->md_code_block) {
        ds4r_seq(r, "\x1b[38;5;75m");
        return;
    }
    if (r->md_inline_code) ds4r_seq(r, "\x1b[36m");
    if (r->md_bold) ds4r_seq(r, "\x1b[1m");
    if (r->md_italic) ds4r_seq(r, "\x1b[3m");
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

/* Fence markers are shown without markdown attributes. */
static void ds4r_put_plain(ds4r *r, char c) {
    ds4r_flush_utf8(r);
    ds4r_reset_color(r);
    ds4r_out(r, &c, 1);
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
        if (r->md_syntax) {
            ds4r_str out = {0};
            ds4r_syntax_emit_line(r, &out, r->code_line, r->code_line_len);
            if (!out.oom && out.p) ds4r_out(r, out.p, out.len);
            else ds4r_out(r, r->code_line, r->code_line_len);
            ds4r_str_free(&out);
        } else {
            ds4r_seq(r, "\x1b[38;5;75m");
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

/* Consume one byte of markdown-aware output.  Backticks and stars are held
 * until the parser knows whether they form a marker; ordinary text goes to the
 * raw writer with the current attributes. */
static void ds4r_md_feed(ds4r *r, char c) {
    if (r->md_fence_info) {
        if (c == '\n') {
            r->md_fence_lang[r->md_fence_lang_len] = '\0';
            r->md_syntax = ds4r_syntax_for_lang(r->md_fence_lang);
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
    ds4r_write_char_raw(r, c);
}

static void ds4r_render_byte(ds4r *r, char c) {
    if (!r->format_markdown || r->in_think) {
        ds4r_md_emit_mark_literals(r);
        ds4r_write_char_raw(r, c);
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
            ds4r_flush_utf8(r);
            r->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (ds4r_has_prefix(cur, rem, think_close)) {
            ds4r_flush_utf8(r);
            r->in_think = false;
            ds4r_reset_color(r);
            if (!r->last_output_newline) ds4r_out(r, "\n", 1);
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
}

void ds4r_set_markdown(ds4r *r, bool enabled) {
    r->format_markdown = enabled && r->use_color;
}

void ds4r_set_in_think(ds4r *r, bool in_think) {
    r->in_think = in_think;
}

bool ds4r_at_line_start(const ds4r *r) {
    return r->last_output_newline;
}

void ds4r_newline(ds4r *r) {
    ds4r_flush_utf8(r);
    ds4r_reset_color(r);
    ds4r_out(r, "\n", 1);
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
    if (r->format_markdown) {
        /* A closing fence can be the last thing the model emits, with no
         * following byte to push the pending backticks through. */
        if (r->md_mark == DS4R_MARK_BACKTICK && r->md_mark_len >= 3)
            ds4r_md_commit_backticks(r);
        else
            ds4r_md_emit_mark_literals(r);
        if (r->md_code_block && r->code_line_len) ds4r_code_emit_line(r, false);
        ds4r_flush_utf8(r);
        r->md_bold = false;
        r->md_italic = false;
        r->md_inline_code = false;
        r->md_code_block = false;
        r->md_fence_info = false;
        r->md_code_line_start = false;
        r->md_code_in_ml_comment = false;
        r->md_syntax = NULL;
        r->md_fence_lang_len = 0;
        r->md_fence_lang[0] = '\0';
        ds4r_md_clear_mark(r);
    }
    ds4r_flush_utf8(r);
    ds4r_reset_color(r);
    fflush(r->fp);
}

void ds4r_free(ds4r *r) {
    free(r->code_line);
    r->code_line = NULL;
    r->code_line_len = 0;
    r->code_line_cap = 0;
}
