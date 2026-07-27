#include "ds4_agent_internal.h"



/* ============================================================================
 * Assistant Markdown Rendering
 * ============================================================================
 *
 * This renderer handles only the cheap markdown cues that make terminal output
 * readable: **bold**, *italic*, inline code, and fenced code blocks.  It is a
 * streaming parser, so it buffers only ambiguous marker bytes long enough to
 * decide whether they are formatting or literal text.
 */

static void agent_tail_capture_append(agent_tail_capture *t,
                                      const char *s, size_t n) {
    if (!t || !n) return;
    if (!t->cap) return;
    if (!t->buf) t->buf = (char *)agent_xmalloc(t->cap);
    t->total += n;

    if (n >= t->cap) {
        memcpy(t->buf, s + n - t->cap, t->cap);
        t->start = 0;
        t->len = t->cap;
        return;
    }

    if (t->len < t->cap) {
        size_t free_tail = t->cap - t->len;
        size_t first = n < free_tail ? n : free_tail;
        size_t pos = (t->start + t->len) % t->cap;
        size_t right = t->cap - pos;
        size_t chunk = first < right ? first : right;
        memcpy(t->buf + pos, s, chunk);
        if (first > chunk) memcpy(t->buf, s + chunk, first - chunk);
        t->len += first;
        s += first;
        n -= first;
    }

    while (n) {
        size_t pos = (t->start + t->len) % t->cap;
        size_t right = t->cap - pos;
        size_t chunk = n < right ? n : right;
        memcpy(t->buf + pos, s, chunk);
        t->start = (t->start + chunk) % t->cap;
        s += chunk;
        n -= chunk;
    }
}



char *agent_tail_capture_take(agent_tail_capture *t, size_t *len) {
    size_t n = t ? t->len : 0;
    char *out = (char *)agent_xmalloc(n + 1);
    if (n) {
        size_t right = t->cap - t->start;
        size_t first = n < right ? n : right;
        memcpy(out, t->buf + t->start, first);
        if (n > first) memcpy(out + first, t->buf, n - first);
    }
    out[n] = '\0';
    if (len) *len = n;
    free(t->buf);
    memset(t, 0, sizeof(*t));
    return out;
}



void renderer_write(agent_token_renderer *r, const char *s, size_t n) {
    if (r->capture) agent_tail_capture_append(r->capture, s, n);
    else agent_publish(r->worker, s, n);
}



static void renderer_set_grey(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[38;5;245m", 11);
}



void renderer_reset_color(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[0m", 4);
    r->color_open = false;
}



static size_t renderer_utf8_need(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}



static bool renderer_has_text_attrs(agent_token_renderer *r) {
    return r->in_think || r->md_bold || r->md_italic ||
           r->md_inline_code || r->md_code_block;
}



static void renderer_set_text_attrs(agent_token_renderer *r) {
    if (!r->use_color) return;
    if (r->in_think) {
        renderer_set_grey(r);
        return;
    }
    if (r->md_code_block) {
        renderer_write(r, "\x1b[38;5;75m", 10);
        return;
    } else if (r->md_inline_code) {
        renderer_write(r, "\x1b[36m", 5);
    }
    if (r->md_bold) renderer_write(r, "\x1b[1m", 4);
    if (r->md_italic) renderer_write(r, "\x1b[3m", 4);
}



void renderer_restore_text_attrs(agent_token_renderer *r) {
    if (!r->use_color || !r->color_open || !renderer_has_text_attrs(r)) return;
    renderer_set_text_attrs(r);
}



static void renderer_write_complete_char_raw(agent_token_renderer *r, const char *s, size_t n) {
    bool styled = r->use_color && renderer_has_text_attrs(r);
    if (styled && !r->color_open) {
        renderer_set_text_attrs(r);
        r->color_open = true;
    } else if (!styled && r->color_open) {
        renderer_reset_color(r);
    }
    renderer_write(r, s, n);
    if (n) r->wrote_visible_output = true;
    r->last_output_newline = n == 1 && s[0] == '\n';
}



static void renderer_flush_utf8(agent_token_renderer *r) {
    if (!r->utf8_pending_len) return;
    renderer_write_complete_char_raw(r, r->utf8_pending, r->utf8_pending_len);
    r->utf8_pending_len = 0;
    r->utf8_pending_need = 0;
}



static void renderer_write_char_raw(agent_token_renderer *r, char c) {
    unsigned char uc = (unsigned char)c;

    if (r->utf8_pending_len) {
        if ((uc & 0xc0) == 0x80 && r->utf8_pending_len < sizeof(r->utf8_pending)) {
            r->utf8_pending[r->utf8_pending_len++] = c;
            if (r->utf8_pending_len == r->utf8_pending_need) renderer_flush_utf8(r);
            return;
        }
        renderer_flush_utf8(r);
    }

    size_t need = renderer_utf8_need(uc);
    if (need == 1) {
        renderer_write_complete_char_raw(r, &c, 1);
        return;
    }
    r->utf8_pending[0] = c;
    r->utf8_pending_len = 1;
    r->utf8_pending_need = need;
}



static void renderer_write_plain_byte(agent_token_renderer *r, char c) {
    bool old_bold = r->md_bold;
    bool old_italic = r->md_italic;
    bool old_inline_code = r->md_inline_code;
    bool old_code_block = r->md_code_block;

    /* Code blocks are streamed immediately in plain text, then repainted with
     * syntax colors when a complete terminal-safe line is available.  Disable
     * markdown attributes only for this byte; renderer_write_char_raw() will
     * reset any tracked manual color once if needed. */
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    renderer_write_char_raw(r, c);
    r->md_bold = old_bold;
    r->md_italic = old_italic;
    r->md_inline_code = old_inline_code;
    r->md_code_block = old_code_block;
}



static const char *agent_kw_generic[] = {
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



static const char *agent_kw_c[] = {
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



static const char *agent_kw_python[] = {
    "and","as","assert","async","await","break","case","class","continue",
    "def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","match","nonlocal","not","or","pass",
    "raise","return","try","while","with","yield",
    "False|","None|","True|","bool|","bytes|","dict|","float|","int|",
    "list|","object|","set|","str|","tuple|",NULL
};



static const char *agent_kw_js[] = {
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



static const char *agent_kw_java[] = {
    "abstract","assert","break","case","catch","class","const","continue",
    "default","do","else","enum","extends","final","finally","for","goto",
    "if","implements","import","instanceof","interface","native","new",
    "package","private","protected","public","return","static","strictfp",
    "super","switch","synchronized","this","throw","throws","transient",
    "try","volatile","while",
    "boolean|","byte|","char|","double|","false|","float|","int|","long|",
    "null|","short|","true|","void|",NULL
};



static const char *agent_kw_csharp[] = {
    "abstract","as","base","break","case","catch","checked","class","const",
    "continue","default","delegate","do","else","enum","event","explicit",
    "extern","finally","fixed","for","foreach","goto","if","implicit","in",
    "interface","internal","is","lock","namespace","new","operator","out",
    "override","params","private","protected","public","readonly","ref",
    "return","sealed","sizeof","stackalloc","static","struct","switch",
    "this","throw","try","typeof","unchecked","unsafe","using","virtual",
    "volatile","while","async","await","get","init","record","set","var",
    "bool|","byte|","char|","decimal|","double|","false|","float|","int|",
    "long|","null|","object|","sbyte|","short|","string|","true|","uint|",
    "ulong|","ushort|","void|",NULL
};



static const char *agent_kw_go[] = {
    "break","case","chan","const","continue","default","defer","else",
    "fallthrough","for","func","go","goto","if","import","interface",
    "map","package","range","return","select","struct","switch","type",
    "var","bool|","byte|","complex64|","complex128|","error|","false|",
    "float32|","float64|","int|","int8|","int16|","int32|","int64|",
    "nil|","rune|","string|","true|","uint|","uint8|","uint16|",
    "uint32|","uint64|","uintptr|",NULL
};



static const char *agent_kw_rust[] = {
    "as","async","await","break","const","continue","crate","dyn","else",
    "enum","extern","fn","for","if","impl","in","let","loop","match",
    "mod","move","mut","pub","ref","return","self","Self","static",
    "struct","super","trait","type","unsafe","use","where","while",
    "bool|","char|","false|","f32|","f64|","i8|","i16|","i32|","i64|",
    "i128|","isize|","str|","String|","true|","u8|","u16|","u32|",
    "u64|","u128|","usize|",NULL
};



static const char *agent_kw_shell[] = {
    "case","do","done","elif","else","esac","fi","for","function","if",
    "in","select","then","time","until","while","break","continue",
    "return","export","local","readonly","source","test","true|","false|",
    "echo|","printf|","cd|","pwd|","read|","set|","unset|","shift|",NULL
};



static const char *agent_kw_sql[] = {
    "add","alter","and","as","asc","between","by","case","check","column",
    "constraint","create","delete","desc","distinct","drop","else","end",
    "exists","foreign","from","group","having","in","index","insert",
    "into","is","join","key","left","like","limit","not","null","on",
    "or","order","outer","primary","references","right","select","set",
    "table","then","union","unique","update","values","view","where",
    "bigint|","boolean|","date|","decimal|","false|","int|","integer|",
    "numeric|","real|","text|","timestamp|","true|","varchar|",NULL
};



static const char *agent_kw_ruby[] = {
    "BEGIN","END","alias","and","begin","break","case","class","def",
    "defined?","do","else","elsif","end","ensure","for","if","in",
    "module","next","not","or","redo","rescue","retry","return","self",
    "super","then","undef","unless","until","when","while","yield",
    "false|","nil|","true|",NULL
};



static const char *agent_kw_php[] = {
    "abstract","and","array","as","break","callable","case","catch","class",
    "clone","const","continue","declare","default","die","do","echo","else",
    "elseif","empty","enddeclare","endfor","endforeach","endif","endswitch",
    "endwhile","eval","exit","extends","final","finally","fn","for",
    "foreach","function","global","goto","if","implements","include",
    "include_once","instanceof","insteadof","interface","isset","list",
    "match","namespace","new","or","print","private","protected","public",
    "readonly","require","require_once","return","static","switch","throw",
    "trait","try","unset","use","var","while","xor","bool|","false|",
    "float|","int|","null|","string|","true|","void|",NULL
};



static const char *agent_kw_swift[] = {
    "actor","as","associatedtype","async","await","break","case","catch",
    "class","continue","default","defer","do","else","enum","extension",
    "fallthrough","for","func","guard","if","import","in","init","inout",
    "is","let","nonisolated","operator","private","protocol","public",
    "repeat","return","self","Self","static","struct","subscript","super",
    "switch","throw","throws","try","typealias","var","where","while",
    "Any|","Bool|","Double|","false|","Float|","Int|","nil|","String|",
    "true|","Void|",NULL
};



static const char *agent_kw_kotlin[] = {
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



static const char *agent_kw_zig[] = {
    "addrspace","align","allowzero","and","anyframe","anytype","asm",
    "async","await","break","callconv","catch","comptime","const",
    "continue","defer","else","enum","errdefer","error","export","extern",
    "fn","for","if","inline","linksection","noalias","noinline","nosuspend",
    "opaque","or","orelse","packed","pub","resume","return","struct",
    "suspend","switch","test","threadlocal","try","union","unreachable",
    "usingnamespace","var","volatile","while",
    "bool|","false|","f32|","f64|","i32|","i64|","null|","true|","u8|",
    "u16|","u32|","u64|","usize|","void|",NULL
};



static const char *agent_kw_lua[] = {
    "and","break","do","else","elseif","end","false","for","function",
    "goto","if","in","local","nil","not","or","repeat","return","then",
    "true","until","while",NULL
};



static const char *agent_kw_html[] = {
    "a","body","button","div","doctype","form","h1","h2","h3","head",
    "html","input","label","li","link","main","meta","ol","option","p",
    "script","section","select","span","style","table","tbody","td","th",
    "thead","title","tr","ul","class|","href|","id|","name|","rel|",
    "src|","type|","value|",NULL
};



static const char *agent_kw_css[] = {
    "align-items","background","border","bottom","color","display","flex",
    "font","font-size","gap","grid","height","justify-content","left",
    "margin","max-width","min-width","padding","position","right","top",
    "transform","width","z-index","absolute|","auto|","block|","flex|",
    "grid|","hidden|","inline|","none|","relative|","solid|",NULL
};



static const agent_syntax agent_syntaxes[] = {
    {"generic", " text txt", agent_kw_generic, {"//","#",NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"c", " c h cpp c++ cc cxx hpp hxx objc objective-c", agent_kw_c, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"python", " py python py3", agent_kw_python, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"javascript", " js jsx javascript typescript ts tsx node mjs cjs", agent_kw_js, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"java", " java", agent_kw_java, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"csharp", " cs c# csharp dotnet", agent_kw_csharp, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"go", " go golang", agent_kw_go, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"rust", " rs rust", agent_kw_rust, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"shell", " sh bash zsh shell fish ksh", agent_kw_shell, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"sql", " sql postgres mysql sqlite", agent_kw_sql, {"--",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_CASE_INSENSITIVE},
    {"ruby", " rb ruby", agent_kw_ruby, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"php", " php", agent_kw_php, {"//","#",NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"swift", " swift", agent_kw_swift, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"kotlin", " kt kts kotlin", agent_kw_kotlin, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"zig", " zig", agent_kw_zig, {"//",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"lua", " lua", agent_kw_lua, {"--",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"html", " html htm xml svg", agent_kw_html, {NULL,NULL,NULL}, "<!--", "-->",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"css", " css scss sass", agent_kw_css, {NULL,NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"json", " json jsonc", NULL, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"yaml", " yaml yml toml ini", NULL, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"markdown", " md markdown", agent_kw_generic, {NULL,NULL,NULL}, "<!--", "-->",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {NULL, NULL, NULL, {NULL,NULL,NULL}, NULL, NULL, 0}
};



static bool agent_syntax_alias_match(const char *aliases, const char *lang) {
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



static const agent_syntax *agent_syntax_for_lang(const char *lang) {
    if (lang && lang[0]) {
        for (const agent_syntax *s = agent_syntaxes; s->name; s++) {
            if (!strcasecmp(s->name, lang) ||
                agent_syntax_alias_match(s->aliases, lang))
                return s;
        }
    }
    return &agent_syntaxes[0];
}



const agent_syntax *agent_syntax_for_path(const char *path) {
    if (!path || !path[0]) return agent_syntax_for_lang(NULL);
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if (!strcasecmp(base, "Dockerfile")) return agent_syntax_for_lang("sh");
    if (!strcasecmp(base, "Makefile") || !strcasecmp(base, "makefile"))
        return agent_syntax_for_lang("sh");
    const char *dot = strrchr(base, '.');
    if (!dot || !dot[1]) return agent_syntax_for_lang(NULL);
    return agent_syntax_for_lang(dot + 1);
}



static bool agent_syntax_separator(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || isspace(uc) || strchr(",.()+-/*=~%[]{}<>:;!&|^?", c) != NULL;
}



static const char *agent_syntax_line_comment(const agent_syntax *syn,
                                             const char *p) {
    if (!syn) return NULL;
    for (int i = 0; i < 3 && syn->singleline_comments[i]; i++) {
        const char *m = syn->singleline_comments[i];
        size_t mlen = strlen(m);
        if (mlen && !strncmp(p, m, mlen)) return m;
    }
    return NULL;
}



static int agent_syntax_color(int hl) {
    switch (hl) {
    case AGENT_HL_COMMENT: return 244;
    case AGENT_HL_KEYWORD1: return 214;
    case AGENT_HL_KEYWORD2: return 81;
    case AGENT_HL_STRING: return 150;
    case AGENT_HL_NUMBER: return 203;
    default: return 252;
    }
}



static void renderer_syntax_write(agent_token_renderer *r, int hl,
                                  const char *s, size_t n) {
    if (!n) return;
    if (hl != AGENT_HL_NORMAL) r->md_syntax_has_highlight = true;
    if (r->md_syntax_silent) return;
    if (r->use_color && hl != AGENT_HL_NORMAL) {
        char seq[32];
        snprintf(seq, sizeof(seq), "\x1b[38;5;%dm", agent_syntax_color(hl));
        renderer_write(r, seq, strlen(seq));
    }
    renderer_write(r, s, n);
    if (r->use_color && hl != AGENT_HL_NORMAL) renderer_write(r, "\x1b[0m", 4);
    r->wrote_visible_output = true;
    r->last_output_newline = false;
}



static void renderer_syntax_write_upto_marker(agent_token_renderer *r) {
    static const char marker[] = "[upto]";
    r->md_syntax_has_highlight = true;
    if (r->md_syntax_silent) return;
    if (r->use_color) {
        renderer_write(r, "\x1b[38;5;244m[", strlen("\x1b[38;5;244m["));
        renderer_write(r, "\x1b[1;38;5;177mupto",
                       strlen("\x1b[1;38;5;177mupto"));
        renderer_write(r, "\x1b[38;5;244m]\x1b[0m",
                       strlen("\x1b[38;5;244m]\x1b[0m"));
    } else {
        renderer_write(r, marker, sizeof(marker) - 1);
    }
    r->wrote_visible_output = true;
    r->last_output_newline = false;
}



static size_t agent_syntax_keyword_len(const char *kw, bool *secondary) {
    size_t len = strlen(kw);
    *secondary = len && kw[len - 1] == '|';
    return *secondary ? len - 1 : len;
}



static bool agent_syntax_match_keyword(const agent_syntax *syn,
                                       const char *p,
                                       const char *line_end,
                                       size_t *out_len,
                                       int *out_hl) {
    if (!syn || !syn->keywords) return false;
    for (int i = 0; syn->keywords[i]; i++) {
        bool secondary = false;
        size_t klen = agent_syntax_keyword_len(syn->keywords[i], &secondary);
        if ((size_t)(line_end - p) < klen) continue;
        bool match = (syn->flags & AGENT_SYNTAX_CASE_INSENSITIVE) ?
            !strncasecmp(p, syn->keywords[i], klen) :
            !strncmp(p, syn->keywords[i], klen);
        if (!match) continue;
        if (!agent_syntax_separator(p[klen])) continue;
        *out_len = klen;
        *out_hl = secondary ? AGENT_HL_KEYWORD2 : AGENT_HL_KEYWORD1;
        return true;
    }
    return false;
}



static bool agent_syntax_number_start(const char *p, const char *line,
                                      bool prev_sep, int prev_hl) {
    unsigned char c = (unsigned char)*p;
    if (isdigit(c) && (prev_sep || prev_hl == AGENT_HL_NUMBER)) return true;
    if (*p == '.' && p > line && prev_hl == AGENT_HL_NUMBER) return true;
    return false;
}



static size_t agent_syntax_number_len(const char *p, const char *line_end) {
    const char *q = p;
    while (q < line_end) {
        unsigned char c = (unsigned char)*q;
        if (isalnum(c) || *q == '_' || *q == '.' || *q == '+' || *q == '-') q++;
        else break;
    }
    return (size_t)(q - p);
}



static void renderer_syntax_emit_line(agent_token_renderer *r,
                                      const char *line, size_t len) {
    const agent_syntax *syn = r->md_syntax ? r->md_syntax : agent_syntax_for_lang(NULL);
    const char *p = line;
    const char *end = line + len;
    bool prev_sep = true;
    int prev_hl = AGENT_HL_NORMAL;
    int in_string = 0;

    while (p < end) {
        if (r->md_code_highlight_upto &&
            (size_t)(end - p) >= strlen("[upto]") &&
            !strncmp(p, "[upto]", strlen("[upto]")))
        {
            renderer_syntax_write_upto_marker(r);
            p += strlen("[upto]");
            prev_sep = true;
            prev_hl = AGENT_HL_NORMAL;
            continue;
        }

        if (r->md_code_in_ml_comment) {
            const char *mce = syn->multiline_end;
            if (mce && *mce) {
                size_t mlen = strlen(mce);
                const char *q = p;
                while (q < end && ((size_t)(end - q) < mlen ||
                       strncmp(q, mce, mlen))) q++;
                if (q < end) {
                    q += mlen;
                    renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(q - p));
                    p = q;
                    r->md_code_in_ml_comment = false;
                    prev_sep = true;
                    prev_hl = AGENT_HL_COMMENT;
                    continue;
                }
            }
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(end - p));
            return;
        }

        const char *scs = agent_syntax_line_comment(syn, p);
        if (!in_string && scs) {
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(end - p));
            return;
        }

        if (!in_string && syn->multiline_start && syn->multiline_end &&
            !strncmp(p, syn->multiline_start, strlen(syn->multiline_start))) {
            size_t mlen = strlen(syn->multiline_start);
            const char *q = p + mlen;
            size_t elen = strlen(syn->multiline_end);
            while (q < end && ((size_t)(end - q) < elen ||
                   strncmp(q, syn->multiline_end, elen))) q++;
            if (q < end) q += elen;
            else r->md_code_in_ml_comment = true;
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_COMMENT;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_STRINGS) && in_string) {
            const char *q = p;
            while (q < end) {
                if (*q == '\\' && q + 1 < end) {
                    q += 2;
                    continue;
                }
                q++;
                if (q[-1] == in_string) {
                    in_string = 0;
                    break;
                }
            }
            renderer_syntax_write(r, AGENT_HL_STRING, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_STRING;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_STRINGS) &&
            (*p == '"' || *p == '\'' ||
             ((syn->flags & AGENT_SYNTAX_BACKTICK_STRINGS) && *p == '`'))) {
            int quote = *p;
            const char *q = p + 1;
            while (q < end) {
                if (*q == '\\' && q + 1 < end) {
                    q += 2;
                    continue;
                }
                q++;
                if (q[-1] == quote) {
                    break;
                }
            }
            renderer_syntax_write(r, AGENT_HL_STRING, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_STRING;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_NUMBERS) &&
            agent_syntax_number_start(p, line, prev_sep, prev_hl)) {
            size_t nlen = agent_syntax_number_len(p, end);
            renderer_syntax_write(r, AGENT_HL_NUMBER, p, nlen);
            p += nlen;
            prev_sep = false;
            prev_hl = AGENT_HL_NUMBER;
            continue;
        }

        if (prev_sep) {
            size_t klen = 0;
            int khl = AGENT_HL_NORMAL;
            if (agent_syntax_match_keyword(syn, p, end, &klen, &khl)) {
                renderer_syntax_write(r, khl, p, klen);
                p += klen;
                prev_sep = false;
                prev_hl = khl;
                continue;
            }
        }

        renderer_syntax_write(r, AGENT_HL_NORMAL, p, 1);
        prev_sep = agent_syntax_separator(*p);
        prev_hl = AGENT_HL_NORMAL;
        p++;
    }
}



static void renderer_code_line_append(agent_token_renderer *r,
                                      const char *s, size_t n) {
    if (!n) return;
    if (r->md_code_line_len + n + 1 > r->md_code_line_cap) {
        size_t cap = r->md_code_line_cap ? r->md_code_line_cap * 2 : 256;
        while (cap < r->md_code_line_len + n + 1) cap *= 2;
        r->md_code_line = (char *)agent_xrealloc(r->md_code_line, cap);
        r->md_code_line_cap = cap;
    }
    memcpy(r->md_code_line + r->md_code_line_len, s, n);
    r->md_code_line_len += n;
    r->md_code_line[r->md_code_line_len] = '\0';
}



int renderer_terminal_cols(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return ws.ws_col;
    return 80;
}



static bool renderer_code_line_can_repaint(agent_token_renderer *r) {
    if (!r->use_color || r->capture || r->md_code_line_len == 0) return false;
    int cols = renderer_terminal_cols();
    size_t prefix_len = r->md_code_line_prefix ?
                        strlen(r->md_code_line_prefix) : 0;
    if (cols <= 1 || prefix_len + r->md_code_line_len >= (size_t)cols)
        return false;
    for (size_t i = 0; i < r->md_code_line_len; i++) {
        unsigned char c = (unsigned char)r->md_code_line[i];
        if (c == '\t' || c == 0x1b || c >= 0x80 || (c < 0x20 && c != '\r'))
            return false;
    }
    return true;
}



static void renderer_code_write_line_prefix(agent_token_renderer *r) {
    if (!r->md_code_line_prefix) return;
    if (r->use_color && r->md_code_line_prefix_color)
        renderer_write(r, r->md_code_line_prefix_color,
                       strlen(r->md_code_line_prefix_color));
    renderer_write(r, r->md_code_line_prefix,
                   strlen(r->md_code_line_prefix));
    if (r->use_color && r->md_code_line_prefix_color)
        renderer_write(r, "\x1b[0m", 4);
    r->color_open = false;
}



/* Run the syntax highlighter in silent mode to learn whether the already
 * streamed line would change if repainted, while preserving the multiline
 * comment state until the caller decides whether repaint is safe. */
static bool renderer_code_scan_line(agent_token_renderer *r,
                                    bool *final_ml_comment) {
    bool old_silent = r->md_syntax_silent;
    bool old_highlight = r->md_syntax_has_highlight;
    bool old_ml_comment = r->md_code_in_ml_comment;

    r->md_syntax_silent = true;
    r->md_syntax_has_highlight = false;
    renderer_syntax_emit_line(r, r->md_code_line, r->md_code_line_len);
    bool changed = r->md_syntax_has_highlight;
    *final_ml_comment = r->md_code_in_ml_comment;

    r->md_code_in_ml_comment = old_ml_comment;
    r->md_syntax_silent = old_silent;
    r->md_syntax_has_highlight = old_highlight;
    return changed;
}



/* Code is shown as soon as bytes arrive.  At end-of-line we can cheaply
 * replace only that terminal row with syntax-highlighted text, but only for
 * simple one-row ASCII lines; long, tabbed, escaped, or UTF-8 lines are left
 * as streamed and only advance the highlighter state. */
static void renderer_code_emit_buffered_line(agent_token_renderer *r,
                                             bool with_newline) {
    bool final_ml_comment = r->md_code_in_ml_comment;
    bool changed = renderer_code_scan_line(r, &final_ml_comment);
    bool repaint = changed && renderer_code_line_can_repaint(r);
    if (repaint) {
        renderer_reset_color(r);
        renderer_write(r, "\r\x1b[0K", 5);
        renderer_code_write_line_prefix(r);
        renderer_syntax_emit_line(r, r->md_code_line, r->md_code_line_len);
    } else {
        r->md_code_in_ml_comment = final_ml_comment;
    }
    r->md_code_line_len = 0;
    if (with_newline) {
        renderer_write_plain_byte(r, '\n');
        r->wrote_visible_output = true;
        r->last_output_newline = true;
        r->md_code_line_start = true;
    }
}



void renderer_code_byte(agent_token_renderer *r, char c) {
    if (c == '\n') {
        renderer_code_emit_buffered_line(r, true);
        return;
    }
    renderer_code_line_append(r, &c, 1);
    renderer_write_plain_byte(r, c);
    if (c != ' ' && c != '\t' && c != '\r') r->md_code_line_start = false;
}



static void renderer_code_emit_backtick_literals(agent_token_renderer *r,
                                                 size_t count) {
    for (size_t i = 0; i < count; i++) renderer_code_byte(r, '`');
}



static void renderer_code_begin(agent_token_renderer *r) {
    renderer_reset_color(r);
    r->md_code_block = true;
    r->md_inline_code = false;
    r->md_fence_info = true;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = agent_syntax_for_lang(NULL);
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_highlight_upto = false;
    r->md_code_line_len = 0;
}



void renderer_code_stream_begin(agent_token_renderer *r,
                                       const agent_syntax *syntax) {
    renderer_reset_color(r);
    r->md_code_block = true;
    r->md_inline_code = false;
    r->md_fence_info = false;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = syntax ? syntax : agent_syntax_for_lang(NULL);
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_highlight_upto = false;
    r->md_code_line_len = 0;
}



void renderer_code_stream_set_prefix(agent_token_renderer *r,
                                            const char *prefix,
                                            const char *color) {
    r->md_code_line_prefix = prefix;
    r->md_code_line_prefix_color = color;
}



void renderer_code_stream_set_upto_marker(agent_token_renderer *r,
                                                 bool enabled) {
    r->md_code_highlight_upto = enabled;
}



void renderer_code_end(agent_token_renderer *r) {
    bool only_space = true;
    for (size_t i = 0; i < r->md_code_line_len; i++) {
        if (r->md_code_line[i] != ' ' && r->md_code_line[i] != '\t' &&
            r->md_code_line[i] != '\r') {
            only_space = false;
            break;
        }
    }
    if (r->md_code_line_len && !only_space)
        renderer_code_emit_buffered_line(r, false);
    else
        r->md_code_line_len = 0;
    r->md_code_block = false;
    r->md_inline_code = false;
    r->md_fence_info = false;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
}



/* Tiny streaming Markdown highlighter for assistant prose.  It deliberately
 * recognizes only delimiters that the model commonly emits in short answers:
 * **bold**, *italic*, `inline code`, ``inline code`` and fenced code blocks.
 * The state machine holds only possible delimiter bytes; once a byte is known
 * to be ordinary text it is sent to the raw UTF-8 writer above.  Tool
 * visualization and redirected output bypass this layer. */
static void renderer_markdown_clear_pending(agent_token_renderer *r) {
    r->md_pending = AGENT_MD_PENDING_NONE;
    r->md_pending_len = 0;
}



static void renderer_markdown_emit_pending_literals(agent_token_renderer *r) {
    char c;
    if (r->md_pending == AGENT_MD_PENDING_STAR) {
        c = '*';
    } else if (r->md_pending == AGENT_MD_PENDING_BACKTICK) {
        c = '`';
    } else {
        return;
    }
    size_t count = r->md_pending_len;
    renderer_markdown_clear_pending(r);
    if (r->md_code_block) {
        if (c == '`') renderer_code_emit_backtick_literals(r, count);
        else for (size_t i = 0; i < count; i++) renderer_code_byte(r, c);
        return;
    }
    for (size_t i = 0; i < count; i++) renderer_write_char_raw(r, c);
}



static void renderer_markdown_commit_backticks(agent_token_renderer *r) {
    size_t count = r->md_pending_len;
    renderer_markdown_clear_pending(r);
    if (count >= 3) {
        for (size_t i = 0; i < count; i++) renderer_write_plain_byte(r, '`');
        if (r->md_code_block) renderer_code_end(r);
        else renderer_code_begin(r);
        return;
    }
    if (r->md_code_block) {
        renderer_code_emit_backtick_literals(r, count);
        return;
    }
    /* Support both `code` and ``code``.  The latter is uncommon in model
     * replies, but accepting it costs nothing and avoids leaking delimiters. */
    r->md_inline_code = !r->md_inline_code;
}



static bool renderer_space_byte(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}



/* Consume one byte of markdown-aware assistant output.  Backticks and stars
 * are held in r->pending until the parser knows whether they form a marker;
 * all ordinary text is emitted with the current terminal attributes. */
static void renderer_markdown_feed(agent_token_renderer *r, char c) {
    if (r->md_fence_info) {
        if (c == '\n') {
            if (r->md_code_block) {
                r->md_fence_lang[r->md_fence_lang_len] = '\0';
                r->md_syntax = agent_syntax_for_lang(r->md_fence_lang);
            }
            renderer_write_plain_byte(r, '\n');
            r->md_fence_info = false;
        } else if (r->md_code_block) {
            unsigned char uc = (unsigned char)c;
            if (r->md_fence_lang_len + 1 < sizeof(r->md_fence_lang) &&
                (isalnum(uc) || c == '_' || c == '-' || c == '+' || c == '#'))
            {
                r->md_fence_lang[r->md_fence_lang_len++] = c;
            }
            renderer_write_plain_byte(r, c);
        }
        return;
    }

    if (r->md_pending == AGENT_MD_PENDING_BACKTICK) {
        if (c == '`') {
            r->md_pending_len++;
            return;
        }
        renderer_markdown_commit_backticks(r);
        renderer_markdown_feed(r, c);
        return;
    }

    if (r->md_pending == AGENT_MD_PENDING_STAR) {
        renderer_markdown_clear_pending(r);
        if (!r->md_inline_code && !r->md_code_block && c == '*') {
            r->md_bold = !r->md_bold;
            return;
        }
        if (!r->md_inline_code && !r->md_code_block &&
            (r->md_italic || !renderer_space_byte(c)))
        {
            r->md_italic = !r->md_italic;
            renderer_markdown_feed(r, c);
            return;
        }
        renderer_write_char_raw(r, '*');
        renderer_markdown_feed(r, c);
        return;
    }

    if (c == '`' && (!r->md_code_block || r->md_code_line_start)) {
        r->md_pending = AGENT_MD_PENDING_BACKTICK;
        r->md_pending_len = 1;
        return;
    }
    if (r->md_code_block) {
        renderer_code_byte(r, c);
        return;
    }
    if (!r->md_inline_code && !r->md_code_block && c == '*') {
        r->md_pending = AGENT_MD_PENDING_STAR;
        r->md_pending_len = 1;
        return;
    }
    renderer_write_char_raw(r, c);
}



static void renderer_markdown_finish(agent_token_renderer *r) {
    /* A closing code fence can be the final bytes of the assistant reply.  In
     * that case no following character arrives to force the pending backticks
     * through the normal streaming path, so commit a full fence here instead of
     * leaking the literal ``` marker to the terminal. */
    if (r->md_pending == AGENT_MD_PENDING_BACKTICK && r->md_pending_len >= 3)
        renderer_markdown_commit_backticks(r);
    else
        renderer_markdown_emit_pending_literals(r);
    if (r->md_code_block && r->md_code_line_len)
        renderer_code_emit_buffered_line(r, false);
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
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_highlight_upto = false;
    free(r->md_code_line);
    r->md_code_line = NULL;
    r->md_code_line_len = 0;
    r->md_code_line_cap = 0;
}



void renderer_write_char(agent_token_renderer *r, char c) {
    if (!r->format_markdown || r->in_think) {
        renderer_markdown_emit_pending_literals(r);
        renderer_write_char_raw(r, c);
        return;
    }
    renderer_markdown_feed(r, c);
}



/* Render assistant text while hiding <think> tags and dimming thinking text.
 * The function is also responsible for not prematurely emitting a partial
 * control tag split across model tokens. */
static void renderer_process(agent_token_renderer *r, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = r->pending_len + len;
    char *buf = (char *)agent_xmalloc(total ? total : 1);
    if (r->pending_len) memcpy(buf, r->pending, r->pending_len);
    if (len) memcpy(buf + r->pending_len, text, len);
    r->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            r->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            r->in_think = false;
            renderer_reset_color(r);
            if (!r->last_output_newline) renderer_write(r, "\n", 1);
            renderer_write(r, "\n", 1);
            r->last_output_newline = true;
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(r->pending)) {
                memcpy(r->pending, cur, rem);
                r->pending_len = rem;
            }
            break;
        }
        renderer_write_char(r, cur[0]);
        i++;
    }
    free(buf);
}



void renderer_finish(agent_token_renderer *r) {
    if (r->format_thinking) {
        renderer_process(r, NULL, 0, true);
    }
    renderer_markdown_finish(r);
    renderer_flush_utf8(r);
    renderer_reset_color(r);
    if (r->wrote_visible_output) {
        if (!r->last_output_newline) renderer_write(r, "\n", 1);
        renderer_write(r, "\n", 1);
        r->last_output_newline = true;
    }
}



void renderer_color(agent_token_renderer *r, const char *seq) {
    renderer_markdown_emit_pending_literals(r);
    renderer_flush_utf8(r);
    bool reset = !seq || !seq[0] || !strcmp(seq, "\x1b[0m");
    if (r->use_color && seq && seq[0]) renderer_write(r, seq, strlen(seq));
    r->color_open = r->use_color && !reset;
}



void renderer_plain(agent_token_renderer *r, const char *s, size_t n) {
    renderer_markdown_emit_pending_literals(r);
    renderer_flush_utf8(r);
    renderer_write(r, s, n);
    if (n) r->wrote_visible_output = true;
    if (n) r->last_output_newline = s[n - 1] == '\n';
}

