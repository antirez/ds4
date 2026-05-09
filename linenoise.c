/* linenoise.c -- guerrilla line editing library against the idea that a
 * line editing lib needs to be 20,000 lines of C code.
 *
 * You can find the latest source code at:
 *
 *   http://github.com/antirez/linenoise
 *
 * Does a number of crazy assumptions that happen to be true in 99.9999% of
 * the 2010 UNIX computers around.
 *
 * ------------------------------------------------------------------------
 *
 * Copyright (c) 2010-2023, Salvatore Sanfilippo <antirez at gmail dot com>
 * Copyright (c) 2010-2013, Pieter Noordhuis <pcnoordhuis at gmail dot com>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *  *  Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *  *  Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ------------------------------------------------------------------------
 *
 * References:
 * - http://invisible-island.net/xterm/ctlseqs/ctlseqs.html
 * - http://www.3waylabs.com/nw/WWW/products/wizcon/vt220.html
 *
 * Todo list:
 * - Filter bogus Ctrl+<char> combinations.
 * - Win32 support
 *
 * Bloat:
 * - History search like Ctrl+r in readline?
 *
 * List of escape sequences used by this program, we do everything just
 * with three sequences. In order to be so cheap we may have some
 * flickering effect with some slow terminal, but the lesser sequences
 * the more compatible.
 *
 * EL (Erase Line)
 *    Sequence: ESC [ n K
 *    Effect: if n is 0 or missing, clear from cursor to end of line
 *    Effect: if n is 1, clear from beginning of line to cursor
 *    Effect: if n is 2, clear entire line
 *
 * CUF (CUrsor Forward)
 *    Sequence: ESC [ n C
 *    Effect: moves cursor forward n chars
 *
 * CUB (CUrsor Backward)
 *    Sequence: ESC [ n D
 *    Effect: moves cursor backward n chars
 *
 * The following is used to get the terminal width if getting
 * the width with the TIOCGWINSZ ioctl fails
 *
 * DSR (Device Status Report)
 *    Sequence: ESC [ 6 n
 *    Effect: reports the current cursor position as ESC [ n ; m R
 *            where n is the row and m is the column
 *
 * When multi line mode is enabled, we also use an additional escape
 * sequence. However multi line editing is disabled by default.
 *
 * CUU (Cursor Up)
 *    Sequence: ESC [ n A
 *    Effect: moves cursor up of n chars.
 *
 * CUD (Cursor Down)
 *    Sequence: ESC [ n B
 *    Effect: moves cursor down of n chars.
 *
 * When linenoiseClearScreen() is called, two additional escape sequences
 * are used in order to clear the screen and position the cursor at home
 * position.
 *
 * CUP (Cursor position)
 *    Sequence: ESC [ H
 *    Effect: moves the cursor to upper left corner
 *
 * ED (Erase display)
 *    Sequence: ESC [ 2 J
 *    Effect: clear the whole screen
 *
 */

#include <termios.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>
#include "linenoise.h"

#define LINENOISE_DEFAULT_HISTORY_MAX_LEN 100
#define LINENOISE_MAX_LINE (1024*1024)      // That will get dynamically allocated
#define LINENOISE_INITIAL_BUFLEN 4096
#define PASTE_FOLD_THRESHOLD 200            // Min bytes to fold a single-line paste.
#define PASTE_FOLD_CONTEXT 8                // Context chars kept around generic folds.
#define PASTE_MAX_BYTES LINENOISE_MAX_LINE
static char *unsupported_term[] = {"dumb","cons25","emacs",NULL};
static linenoiseCompletionCallback *completionCallback = NULL;
static linenoiseHintsCallback *hintsCallback = NULL;
static linenoiseFreeHintsCallback *freeHintsCallback = NULL;
static char *linenoiseReadLine(FILE *fp, int *err);
static char *linenoiseNoTTY(void);
static void refreshLineWithCompletion(struct linenoiseState *ls, linenoiseCompletions *lc, int flags);
static void refreshLineWithFlags(struct linenoiseState *l, int flags);
static void linenoiseFoldClear(struct linenoiseState *l);

static struct termios orig_termios; /* In order to restore at exit.*/
static int maskmode = 0; /* Show "***" instead of input. For passwords. */
static int rawmode = 0; /* For atexit() function to check if restore is needed*/
static int rawmode_output = STDOUT_FILENO; /* fd used for terminal escapes. */
static int mlmode = 0;  /* Multi line mode. Default is single line. */
static int atexit_registered = 0; /* Register atexit just 1 time. */
static int history_max_len = LINENOISE_DEFAULT_HISTORY_MAX_LEN;
static int history_len = 0;
static char **history = NULL;

/* =========================== UTF-8 support ================================ */

/* Return the number of bytes that compose the UTF-8 character starting at
 * 'c'. This function assumes a valid UTF-8 encoding and handles the four
 * standard byte patterns:
 *   0xxxxxxx -> 1 byte (ASCII)
 *   110xxxxx -> 2 bytes
 *   1110xxxx -> 3 bytes
 *   11110xxx -> 4 bytes */
static int utf8ByteLen(char c) {
    unsigned char uc = (unsigned char)c;
    if ((uc & 0x80) == 0)    return 1;   /* 0xxxxxxx: ASCII */
    if ((uc & 0xE0) == 0xC0) return 2;   /* 110xxxxx: 2-byte seq */
    if ((uc & 0xF0) == 0xE0) return 3;   /* 1110xxxx: 3-byte seq */
    if ((uc & 0xF8) == 0xF0) return 4;   /* 11110xxx: 4-byte seq */
    return 1; /* Fallback for invalid encoding, treat as single byte. */
}

/* Decode a UTF-8 sequence starting at 's' into a Unicode codepoint.
 * Returns the codepoint value. Assumes valid UTF-8 encoding. */
static uint32_t utf8DecodeChar(const char *s, size_t *len) {
    unsigned char *p = (unsigned char *)s;
    uint32_t cp;

    if ((*p & 0x80) == 0) {
        *len = 1;
        return *p;
    } else if ((*p & 0xE0) == 0xC0) {
        *len = 2;
        cp = (*p & 0x1F) << 6;
        cp |= (p[1] & 0x3F);
        return cp;
    } else if ((*p & 0xF0) == 0xE0) {
        *len = 3;
        cp = (*p & 0x0F) << 12;
        cp |= (p[1] & 0x3F) << 6;
        cp |= (p[2] & 0x3F);
        return cp;
    } else if ((*p & 0xF8) == 0xF0) {
        *len = 4;
        cp = (*p & 0x07) << 18;
        cp |= (p[1] & 0x3F) << 12;
        cp |= (p[2] & 0x3F) << 6;
        cp |= (p[3] & 0x3F);
        return cp;
    }
    *len = 1;
    return *p; /* Fallback for invalid sequences. */
}

/* Check if codepoint is a variation selector (emoji style modifiers). */
static int isVariationSelector(uint32_t cp) {
    return cp == 0xFE0E || cp == 0xFE0F;  /* Text/emoji style */
}

/* Check if codepoint is a skin tone modifier. */
static int isSkinToneModifier(uint32_t cp) {
    return cp >= 0x1F3FB && cp <= 0x1F3FF;
}

/* Check if codepoint is Zero Width Joiner. */
static int isZWJ(uint32_t cp) {
    return cp == 0x200D;
}

/* Check if codepoint is a Regional Indicator (for flag emoji). */
static int isRegionalIndicator(uint32_t cp) {
    return cp >= 0x1F1E6 && cp <= 0x1F1FF;
}

/* Check if codepoint is a combining mark or other zero-width character. */
static int isCombiningMark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F) ||   /* Combining Diacriticals */
           (cp >= 0x1AB0 && cp <= 0x1AFF) ||   /* Combining Diacriticals Extended */
           (cp >= 0x1DC0 && cp <= 0x1DFF) ||   /* Combining Diacriticals Supplement */
           (cp >= 0x20D0 && cp <= 0x20FF) ||   /* Combining Diacriticals for Symbols */
           (cp >= 0xFE20 && cp <= 0xFE2F);     /* Combining Half Marks */
}

/* Check if codepoint extends the previous character (doesn't start a new grapheme). */
static int isGraphemeExtend(uint32_t cp) {
    return isVariationSelector(cp) || isSkinToneModifier(cp) ||
           isZWJ(cp) || isCombiningMark(cp);
}

/* Decode the UTF-8 codepoint ending at position 'pos' (exclusive) and
 * return its value. Also sets *cplen to the byte length of the codepoint. */
static uint32_t utf8DecodePrev(const char *buf, size_t pos, size_t *cplen) {
    if (pos == 0) {
        *cplen = 0;
        return 0;
    }
    /* Scan backwards to find the start byte. */
    size_t i = pos;
    do {
        i--;
    } while (i > 0 && (pos - i) < 4 && ((unsigned char)buf[i] & 0xC0) == 0x80);
    *cplen = pos - i;
    size_t dummy;
    return utf8DecodeChar(buf + i, &dummy);
}

/* Given a buffer and a position, return the byte length of the grapheme
 * cluster before that position. A grapheme cluster includes:
 * - The base character
 * - Any following variation selectors, skin tone modifiers
 * - ZWJ sequences (emoji joined by Zero Width Joiner)
 * - Regional indicator pairs (flag emoji) */
static size_t utf8PrevCharLen(const char *buf, size_t pos) {
    if (pos == 0) return 0;

    size_t total = 0;
    size_t curpos = pos;

    /* First, get the last codepoint. */
    size_t cplen;
    uint32_t cp = utf8DecodePrev(buf, curpos, &cplen);
    if (cplen == 0) return 0;
    total += cplen;
    curpos -= cplen;

    /* If we're at an extending character, we need to find what it extends.
     * Keep going back through the grapheme cluster. */
    while (curpos > 0) {
        size_t prevlen;
        uint32_t prevcp = utf8DecodePrev(buf, curpos, &prevlen);
        if (prevlen == 0) break;

        if (isZWJ(prevcp)) {
            /* ZWJ joins two emoji. Include the ZWJ and continue to get
             * the preceding character. */
            total += prevlen;
            curpos -= prevlen;
            /* Now get the character before ZWJ. */
            prevcp = utf8DecodePrev(buf, curpos, &prevlen);
            if (prevlen == 0) break;
            total += prevlen;
            curpos -= prevlen;
            cp = prevcp;
            continue;  /* Check if there's more extending before this. */
        } else if (isGraphemeExtend(cp)) {
            /* Current cp is an extending character; include previous. */
            total += prevlen;
            curpos -= prevlen;
            cp = prevcp;
            continue;
        } else if (isRegionalIndicator(cp) && isRegionalIndicator(prevcp)) {
            /* Two regional indicators form a flag. But we need to be careful:
             * flags are always pairs, so only join if we're at an even boundary.
             * For simplicity, just join one pair. */
            total += prevlen;
            curpos -= prevlen;
            break;
        } else {
            /* No more extending; we've found the start of the cluster. */
            break;
        }
    }

    return total;
}

/* Given a buffer, position and total length, return the byte length of the
 * grapheme cluster at the current position. */
static size_t utf8NextCharLen(const char *buf, size_t pos, size_t len) {
    if (pos >= len) return 0;

    size_t total = 0;
    size_t curpos = pos;

    /* Get the first codepoint. */
    size_t cplen;
    uint32_t cp = utf8DecodeChar(buf + curpos, &cplen);
    total += cplen;
    curpos += cplen;

    int isRI = isRegionalIndicator(cp);

    /* Consume any extending characters that follow. */
    while (curpos < len) {
        size_t nextlen;
        uint32_t nextcp = utf8DecodeChar(buf + curpos, &nextlen);

        if (isZWJ(nextcp) && curpos + nextlen < len) {
            /* ZWJ: include it and the following character. */
            total += nextlen;
            curpos += nextlen;
            /* Get the character after ZWJ. */
            nextcp = utf8DecodeChar(buf + curpos, &nextlen);
            total += nextlen;
            curpos += nextlen;
            continue;  /* Check for more extending after the joined char. */
        } else if (isGraphemeExtend(nextcp)) {
            /* Variation selector, skin tone, combining mark, etc. */
            total += nextlen;
            curpos += nextlen;
            continue;
        } else if (isRI && isRegionalIndicator(nextcp)) {
            /* Second regional indicator for a flag pair. */
            total += nextlen;
            curpos += nextlen;
            isRI = 0;  /* Only pair once. */
            continue;
        } else {
            break;
        }
    }

    return total;
}

/* Return the display width of a Unicode codepoint. This is a heuristic
 * that works for most common cases:
 * - Control chars and zero-width: 0 columns
 * - Grapheme-extending chars (VS, skin tone, ZWJ): 0 columns
 * - ASCII printable: 1 column
 * - Wide chars (CJK, emoji, fullwidth): 2 columns
 * - Everything else: 1 column
 *
 * This is not a full wcwidth() implementation, but a minimal heuristic
 * that handles emoji and CJK characters reasonably well. */
static int utf8CharWidth(uint32_t cp) {
    /* Control characters and combining marks: zero width. */
    if (cp < 32 || (cp >= 0x7F && cp < 0xA0)) return 0;
    if (isCombiningMark(cp)) return 0;

    /* Grapheme-extending characters: zero width.
     * These modify the preceding character rather than taking space. */
    if (isVariationSelector(cp)) return 0;
    if (isSkinToneModifier(cp)) return 0;
    if (isZWJ(cp)) return 0;

    /* Wide character ranges - these display as 2 columns:
     * - CJK Unified Ideographs and Extensions
     * - Fullwidth forms
     * - Various emoji ranges */
    if (cp >= 0x1100 &&
        (cp <= 0x115F ||                      /* Hangul Jamo */
         cp == 0x2329 || cp == 0x232A ||      /* Angle brackets */
         (cp >= 0x231A && cp <= 0x231B) ||    /* Watch, Hourglass */
         (cp >= 0x23E9 && cp <= 0x23F3) ||    /* Various symbols */
         (cp >= 0x23F8 && cp <= 0x23FA) ||    /* Various symbols */
         (cp >= 0x25AA && cp <= 0x25AB) ||    /* Small squares */
         (cp >= 0x25B6 && cp <= 0x25C0) ||    /* Play/reverse buttons */
         (cp >= 0x25FB && cp <= 0x25FE) ||    /* Squares */
         (cp >= 0x2600 && cp <= 0x26FF) ||    /* Misc Symbols (sun, cloud, etc) */
         (cp >= 0x2700 && cp <= 0x27BF) ||    /* Dingbats (❤, ✂, etc) */
         (cp >= 0x2934 && cp <= 0x2935) ||    /* Arrows */
         (cp >= 0x2B05 && cp <= 0x2B07) ||    /* Arrows */
         (cp >= 0x2B1B && cp <= 0x2B1C) ||    /* Squares */
         cp == 0x2B50 || cp == 0x2B55 ||      /* Star, circle */
         (cp >= 0x2E80 && cp <= 0xA4CF &&
          cp != 0x303F) ||                    /* CJK ... Yi */
         (cp >= 0xAC00 && cp <= 0xD7A3) ||    /* Hangul Syllables */
         (cp >= 0xF900 && cp <= 0xFAFF) ||    /* CJK Compatibility Ideographs */
         (cp >= 0xFE10 && cp <= 0xFE1F) ||    /* Vertical forms */
         (cp >= 0xFE30 && cp <= 0xFE6F) ||    /* CJK Compatibility Forms */
         (cp >= 0xFF00 && cp <= 0xFF60) ||    /* Fullwidth Forms */
         (cp >= 0xFFE0 && cp <= 0xFFE6) ||    /* Fullwidth Signs */
         (cp >= 0x1F1E6 && cp <= 0x1F1FF) ||  /* Regional Indicators (flags) */
         (cp >= 0x1F300 && cp <= 0x1F64F) ||  /* Misc Symbols and Emoticons */
         (cp >= 0x1F680 && cp <= 0x1F6FF) ||  /* Transport and Map Symbols */
         (cp >= 0x1F900 && cp <= 0x1F9FF) ||  /* Supplemental Symbols */
         (cp >= 0x1FA00 && cp <= 0x1FAFF) ||  /* Chess, Extended-A */
         (cp >= 0x20000 && cp <= 0x2FFFF)))   /* CJK Extension B and beyond */
        return 2;

    return 1; /* Default: single width */
}

/* If s[] points at an ANSI CSI escape sequence (e.g. a color change like
 * ESC [ 1 ; 32 m), return its length in bytes. Otherwise return 0.
 *
 * The caller must have already verified that s[0] == ESC (0x1b). The
 * sequence layout follows ECMA-48: ESC '[' , parameter bytes (0x30-0x3f),
 * intermediate bytes (0x20-0x2f), and a final byte (0x40-0x7e). */
static size_t ansiEscapeLen(const char *s, size_t len) {
    size_t i;
    if (len < 2 || s[1] != '[') return 0;
    i = 2;
    while (i < len && (unsigned char)s[i] >= 0x30 && (unsigned char)s[i] <= 0x3f) i++;
    while (i < len && (unsigned char)s[i] >= 0x20 && (unsigned char)s[i] <= 0x2f) i++;
    if (i >= len || (unsigned char)s[i] < 0x40 || (unsigned char)s[i] > 0x7e) return 0;
    return i + 1;
}

/* Calculate the display width of a UTF-8 string of 'len' bytes.
 * This is used for cursor positioning in the terminal.
 * Handles grapheme clusters: characters joined by ZWJ contribute 0 width
 * after the first character in the sequence.
 * ANSI CSI escape sequences (e.g. color codes in the prompt) are treated
 * as zero-width. */
static size_t utf8StrWidth(const char *s, size_t len) {
    size_t width = 0;
    size_t i = 0;
    int after_zwj = 0;  /* Track if previous char was ZWJ */

    while (i < len) {
        size_t clen;
        uint32_t cp = utf8DecodeChar(s + i, &clen);

        /* Skip ANSI CSI escape sequences entirely: they produce no
         * glyph, so they must not contribute to the display width.
         * Checked before the ZWJ state so a stray ZWJ immediately
         * followed by ESC cannot swallow the ESC byte. */
        if (cp == 0x1b) {
            size_t skip = ansiEscapeLen(s + i, len - i);
            if (skip > 0) {
                i += skip;
                continue;
            }
        }

        if (after_zwj) {
            /* Character after ZWJ: don't add width, it's joined.
             * But do check for extending chars after it. */
            after_zwj = 0;
        } else {
            width += utf8CharWidth(cp);
        }

        /* Check if this is a ZWJ - next char will be joined. */
        if (isZWJ(cp)) {
            after_zwj = 1;
        }

        i += clen;
    }
    return width;
}

/* Return the display width of a single UTF-8 character at position 's'. */
static int utf8SingleCharWidth(const char *s, size_t len) {
    if (len == 0) return 0;
    size_t clen;
    uint32_t cp = utf8DecodeChar(s, &clen);
    return utf8CharWidth(cp);
}

enum KEY_ACTION{
	KEY_NULL = 0,	    /* NULL */
	CTRL_A = 1,         /* Ctrl+a */
	CTRL_B = 2,         /* Ctrl-b */
	CTRL_C = 3,         /* Ctrl-c */
	CTRL_D = 4,         /* Ctrl-d */
	CTRL_E = 5,         /* Ctrl-e */
	CTRL_F = 6,         /* Ctrl-f */
	CTRL_H = 8,         /* Ctrl-h */
	TAB = 9,            /* Tab */
	CTRL_K = 11,        /* Ctrl+k */
	CTRL_L = 12,        /* Ctrl+l */
	ENTER = 13,         /* Enter */
	CTRL_N = 14,        /* Ctrl-n */
	CTRL_P = 16,        /* Ctrl-p */
	CTRL_T = 20,        /* Ctrl-t */
	CTRL_U = 21,        /* Ctrl+u */
	CTRL_W = 23,        /* Ctrl+w */
	ESC = 27,           /* Escape */
	BACKSPACE =  127    /* Backspace */
};

static void linenoiseAtExit(void);
int linenoiseHistoryAdd(const char *line);
#define REFRESH_CLEAN (1<<0)    // Clean the old prompt from the screen
#define REFRESH_WRITE (1<<1)    // Rewrite the prompt on the screen.
#define REFRESH_ALL (REFRESH_CLEAN|REFRESH_WRITE) // Do both.
static void refreshLine(struct linenoiseState *l);

/* Debugging macro. */
#if 0
FILE *lndebug_fp = NULL;
#define lndebug(...) \
    do { \
        if (lndebug_fp == NULL) { \
            lndebug_fp = fopen("/tmp/lndebug.txt","a"); \
            fprintf(lndebug_fp, \
            "[%d %d %d] p: %d, rows: %d, rpos: %d, max: %d, oldmax: %d\n", \
            (int)l->len,(int)l->pos,(int)l->oldpos,plen,rows,rpos, \
            (int)l->oldrows,old_rows); \
        } \
        fprintf(lndebug_fp, ", " __VA_ARGS__); \
        fflush(lndebug_fp); \
    } while (0)
#else
#define lndebug(fmt, ...)
#endif

/* ======================= Low level terminal handling ====================== */

/* Enable "mask mode". When it is enabled, instead of the input that
 * the user is typing, the terminal will just display a corresponding
 * number of asterisks, like "****". This is useful for passwords and other
 * secrets that should not be displayed. */
void linenoiseMaskModeEnable(void) {
    maskmode = 1;
}

/* Disable mask mode. */
void linenoiseMaskModeDisable(void) {
    maskmode = 0;
}

/* Set if to use or not the multi line mode. */
void linenoiseSetMultiLine(int ml) {
    mlmode = ml;
}

/* Return true if the terminal name is in the list of terminals we know are
 * not able to understand basic escape sequences. */
static int isUnsupportedTerm(void) {
    char *term = getenv("TERM");
    int j;

    if (term == NULL) return 0;
    for (j = 0; unsupported_term[j]; j++)
        if (!strcasecmp(term,unsupported_term[j])) return 1;
    return 0;
}

/* Raw mode: 1960 magic shit. */
static int enableRawMode(int fd) {
    struct termios raw;

    /* Test mode: when LINENOISE_ASSUME_TTY is set, skip terminal setup.
     * This allows testing via pipes without a real terminal. */
    if (getenv("LINENOISE_ASSUME_TTY")) {
        rawmode = 1;
        return 0;
    }

    if (!isatty(STDIN_FILENO)) goto fatal;
    if (!atexit_registered) {
        atexit(linenoiseAtExit);
        atexit_registered = 1;
    }
    if (tcgetattr(fd,&orig_termios) == -1) goto fatal;

    raw = orig_termios;  /* modify the original mode */
    /* input modes: no break, no CR to NL, no parity check, no strip char,
     * no start/stop output control. */
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    /* output modes - disable post processing */
    raw.c_oflag &= ~(OPOST);
    /* control modes - set 8 bit chars */
    raw.c_cflag |= (CS8);
    /* local modes - choing off, canonical off, no extended functions,
     * no signal chars (^Z,^C) */
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    /* control chars - set return condition: min number of bytes and timer.
     * We want read to return every single byte, without timeout. */
    raw.c_cc[VMIN] = 1; raw.c_cc[VTIME] = 0; /* 1 byte, no timer */

    /* put terminal in raw mode after flushing */
    if (tcsetattr(fd,TCSAFLUSH,&raw) < 0) goto fatal;
    rawmode = 1;
    /* Ask the terminal to wrap paste input between ESC[200~ and ESC[201~. */
    if (write(rawmode_output, "\x1b[?2004h", 8) == -1) {}
    return 0;

fatal:
    errno = ENOTTY;
    return -1;
}

static void disableRawMode(int fd) {
    /* Test mode: nothing to restore. */
    if (getenv("LINENOISE_ASSUME_TTY")) {
        rawmode = 0;
        return;
    }
    /* Don't even check the return value as it's too late. */
    if (rawmode && tcsetattr(fd,TCSAFLUSH,&orig_termios) != -1) {
        /* Leave bracketed paste mode when leaving raw mode. */
        if (write(rawmode_output, "\x1b[?2004l", 8) == -1) {}
        rawmode = 0;
    }
}

/* Use the ESC [6n escape sequence to query the horizontal cursor position
 * and return it. On error -1 is returned, on success the position of the
 * cursor. */
static int getCursorPosition(int ifd, int ofd) {
    char buf[32];
    int cols, rows;
    unsigned int i = 0;

    /* Report cursor location */
    if (write(ofd, "\x1b[6n", 4) != 4) return -1;

    /* Read the response: ESC [ rows ; cols R */
    while (i < sizeof(buf)-1) {
        if (read(ifd,buf+i,1) != 1) break;
        if (buf[i] == 'R') break;
        i++;
    }
    buf[i] = '\0';

    /* Parse it. */
    if (buf[0] != ESC || buf[1] != '[') return -1;
    if (sscanf(buf+2,"%d;%d",&rows,&cols) != 2) return -1;
    return cols;
}

/* Try to get the number of columns in the current terminal, or assume 80
 * if it fails. */
static int getColumns(int ifd, int ofd) {
    struct winsize ws;

    /* Test mode: use LINENOISE_COLS env var for fixed width. */
    char *cols_env = getenv("LINENOISE_COLS");
    if (cols_env) return atoi(cols_env);

    if (ioctl(1, TIOCGWINSZ, &ws) == -1 || ws.ws_col == 0) {
        /* ioctl() failed. Try to query the terminal itself. */
        int start, cols;

        /* Get the initial position so we can restore it later. */
        start = getCursorPosition(ifd,ofd);
        if (start == -1) goto failed;

        /* Go to right margin and get position. */
        if (write(ofd,"\x1b[999C",6) != 6) goto failed;
        cols = getCursorPosition(ifd,ofd);
        if (cols == -1) goto failed;

        /* Restore position. */
        if (cols > start) {
            char seq[32];
            snprintf(seq,32,"\x1b[%dD",cols-start);
            if (write(ofd,seq,strlen(seq)) == -1) {
                /* Can't recover... */
            }
        }
        return cols;
    } else {
        return ws.ws_col;
    }

failed:
    return 80;
}

/* Clear the screen. Used to handle ctrl+l */
void linenoiseClearScreen(void) {
    if (write(STDOUT_FILENO,"\x1b[H\x1b[2J",7) <= 0) {
        /* nothing to do, just to avoid warning. */
    }
}

/* Beep, used for completion when there is nothing to complete or when all
 * the choices were already shown. */
static void linenoiseBeep(void) {
    fprintf(stderr, "\x7");
    fflush(stderr);
}

/* ============================== Completion ================================ */

/* Free a list of completion option populated by linenoiseAddCompletion(). */
static void freeCompletions(linenoiseCompletions *lc) {
    size_t i;
    for (i = 0; i < lc->len; i++)
        free(lc->cvec[i]);
    if (lc->cvec != NULL)
        free(lc->cvec);
}

/* Called by completeLine() and linenoiseShow() to render the current
 * edited line with the proposed completion. If the current completion table
 * is already available, it is passed as second argument, otherwise the
 * function will use the callback to obtain it.
 *
 * Flags are the same as refreshLine*(), that is REFRESH_* macros. */
static void refreshLineWithCompletion(struct linenoiseState *ls, linenoiseCompletions *lc, int flags) {
    /* Obtain the table of completions if the caller didn't provide one. */
    linenoiseCompletions ctable = { 0, NULL };
    if (lc == NULL) {
        completionCallback(ls->buf,&ctable);
        lc = &ctable;
    }

    /* Show the edited line with completion if possible, or just refresh. */
    if (ls->completion_idx < lc->len) {
        struct linenoiseState saved = *ls;
        ls->len = ls->pos = strlen(lc->cvec[ls->completion_idx]);
        ls->buf = lc->cvec[ls->completion_idx];
        ls->fold_count = 0;
        refreshLineWithFlags(ls,flags);
        ls->len = saved.len;
        ls->pos = saved.pos;
        ls->buf = saved.buf;
        ls->fold_count = saved.fold_count;
    } else {
        refreshLineWithFlags(ls,flags);
    }

    /* Free the completions table if needed. */
    if (lc != &ctable) freeCompletions(&ctable);
}

/* This is an helper function for linenoiseEdit*() and is called when the
 * user types the <tab> key in order to complete the string currently in the
 * input.
 *
 * The state of the editing is encapsulated into the pointed linenoiseState
 * structure as described in the structure definition.
 *
 * If the function returns non-zero, the caller should handle the
 * returned value as a byte read from the standard input, and process
 * it as usually: this basically means that the function may return a byte
 * read from the terminal but not processed. Otherwise, if zero is returned,
 * the input was consumed by the completeLine() function to navigate the
 * possible completions, and the caller should read for the next characters
 * from stdin. */
static int completeLine(struct linenoiseState *ls, int keypressed) {
    linenoiseCompletions lc = { 0, NULL };
    int nwritten;
    char c = keypressed;

    completionCallback(ls->buf,&lc);
    if (lc.len == 0) {
        linenoiseBeep();
        ls->in_completion = 0;
        c = 0;
    } else {
        switch(c) {
            case 9: /* tab */
                if (ls->in_completion == 0) {
                    ls->in_completion = 1;
                    ls->completion_idx = 0;
                } else {
                    ls->completion_idx = (ls->completion_idx+1) % (lc.len+1);
                    if (ls->completion_idx == lc.len) linenoiseBeep();
                }
                c = 0;
                break;
            case 27: /* escape */
                /* Re-show original buffer */
                if (ls->completion_idx < lc.len) refreshLine(ls);
                ls->in_completion = 0;
                c = 0;
                break;
            default:
                /* Update buffer and return */
                if (ls->completion_idx < lc.len) {
                    nwritten = snprintf(ls->buf,ls->buflen,"%s",
                        lc.cvec[ls->completion_idx]);
                    ls->len = ls->pos = nwritten;
                    linenoiseFoldClear(ls);
                }
                ls->in_completion = 0;
                break;
        }

        /* Show completion or original buffer */
        if (ls->in_completion && ls->completion_idx < lc.len) {
            refreshLineWithCompletion(ls,&lc,REFRESH_ALL);
        } else {
            refreshLine(ls);
        }
    }

    freeCompletions(&lc);
    return c; /* Return last read character */
}

/* Register a callback function to be called for tab-completion. */
void linenoiseSetCompletionCallback(linenoiseCompletionCallback *fn) {
    completionCallback = fn;
}

/* Register a hits function to be called to show hits to the user at the
 * right of the prompt. */
void linenoiseSetHintsCallback(linenoiseHintsCallback *fn) {
    hintsCallback = fn;
}

/* Register a function to free the hints returned by the hints callback
 * registered with linenoiseSetHintsCallback(). */
void linenoiseSetFreeHintsCallback(linenoiseFreeHintsCallback *fn) {
    freeHintsCallback = fn;
}

/* This function is used by the callback function registered by the user
 * in order to add completion options given the input string when the
 * user typed <tab>. See the example.c source code for a very easy to
 * understand example. */
void linenoiseAddCompletion(linenoiseCompletions *lc, const char *str) {
    size_t len = strlen(str);
    char *copy, **cvec;

    copy = malloc(len+1);
    if (copy == NULL) return;
    memcpy(copy,str,len+1);
    cvec = realloc(lc->cvec,sizeof(char*)*(lc->len+1));
    if (cvec == NULL) {
        free(copy);
        return;
    }
    lc->cvec = cvec;
    lc->cvec[lc->len++] = copy;
}

/* =========================== Line editing ================================= */

/* We define a very simple "append buffer" structure, that is an heap
 * allocated string where we can append to. This is useful in order to
 * write all the escape sequences in a buffer and flush them to the standard
 * output in a single call, to avoid flickering effects. */
struct abuf {
    char *b;
    int len;
};

static void abInit(struct abuf *ab) {
    ab->b = NULL;
    ab->len = 0;
}

static void abAppend(struct abuf *ab, const char *s, int len) {
    char *new = realloc(ab->b,ab->len+len);

    if (new == NULL) return;
    memcpy(new+ab->len,s,len);
    ab->b = new;
    ab->len += len;
}

static void abFree(struct abuf *ab) {
    free(ab->b);
}

/* A fold is a display-only replacement for a range in l->buf. The edited
 * buffer always keeps the real bytes; refresh code asks linenoiseRenderBuffer()
 * for a temporary printable version plus the cursor position inside it. */
struct linenoiseFold {
    size_t start;
    size_t end;
    char display[64];
    size_t displaylen;
};

struct linenoiseFolds {
    int count;
    struct linenoiseFold fold[LINENOISE_MAX_FOLDS];
};

/* Return the number of logical lines in the range. */
static size_t foldCountLines(const char *buf, size_t len) {
    size_t lines = 1, j;
    for (j = 0; j < len; j++) {
        if (buf[j] == '\n') lines++;
    }
    return lines;
}

/* Return true if the text should be folded: if it contains newlines or is at
 * least PASTE_FOLD_THRESHOLD bytes long. */
static int shouldFoldText(const char *buf, size_t len) {
    return memchr(buf, '\n', len) != NULL || len >= PASTE_FOLD_THRESHOLD;
}

/* Fill f->display with the text shown instead of the folded range. */
static void foldSetRenderedText(struct linenoiseFold *f, const char *buf) {
    size_t hidden = f->end - f->start;
    size_t lines = foldCountLines(buf + f->start, hidden);
    int n;

    if (lines > 1)
        n = snprintf(f->display,sizeof(f->display),"[... %zu pasted lines ...]",lines);
    else
        n = snprintf(f->display,sizeof(f->display),"[... %zu pasted chars ...]",hidden);
    if (n < 0) n = 0;
    f->displaylen = (size_t)n;
}

/* Populate f with one fold reconstructed from a history entry. History stores
 * the real text, but not the original paste boundaries, so we reconstruct
 * an approximation of text we want to hide on the fly: if it is long or
 * contains newlines. */
static int linenoiseBuildHistoryFold(struct linenoiseState *l, struct linenoiseFold *f) {
    f->start = f->end = f->displaylen = 0;
    if (l->len == 0 || maskmode) return 0;
    if (!shouldFoldText(l->buf,l->len)) return 0;

    f->start = 0;
    f->end = l->len;
    if (l->len > PASTE_FOLD_CONTEXT*2) {
        size_t pos = 0, chars = 0;
        int nl = 0;

        /* We leave (if possible) a few chars on
         * the start before the fold, to give context. */
        while (pos < l->len && chars < PASTE_FOLD_CONTEXT) {
            size_t step = utf8NextCharLen(l->buf,pos,l->len);
            if (step == 0 || pos + step > l->len) break;
            if (l->buf[pos] == '\n') nl = 1;
            pos += step;
            chars++;
        }
        f->start = nl ? 0 : pos;

        /* And also on the end side. */
        pos = l->len;
        chars = 0;
        nl = 0;
        while (pos > 0 && chars < PASTE_FOLD_CONTEXT) {
            size_t step = utf8PrevCharLen(l->buf,pos);
            if (step == 0 || step > pos) break;
            pos -= step;
            if (l->buf[pos] == '\n') nl = 1;
            chars++;
        }
        f->end = nl ? l->len : pos;
        if (f->start >= f->end) {
            f->start = 0;
            f->end = l->len;
        }
    }
    foldSetRenderedText(f,l->buf);
    return 1;
}

/* Populate fs with the folds to render for the current buffer. As a side
 * effect, the rendered text of each fold is updated. Return 1 if folding
 * should be used, or 0 if the buffer should be rendered as-is. */
static int linenoiseGetRenderFolds(struct linenoiseState *l, struct linenoiseFolds *fs) {
    int j;

    fs->count = 0;
    if (l->len == 0 || maskmode) return 0;

    for (j = 0; j < l->fold_count; j++) {
        struct linenoiseFold *f;
        size_t start = l->fold_start[j];
        size_t end = l->fold_end[j];

        if (start >= end || end > l->len) continue;
        f = fs->fold + fs->count++;
        f->start = start;
        f->end = end;
        foldSetRenderedText(f,l->buf);
    }
    return fs->count != 0;
}

/* Return the freshly allocated string content that is actually displayed in
 * the user prompt. It can be the actual edited line, or a special version
 * where pasted or multiline history ranges are replaced by their folded
 * "[...]" style versions. outpos is l->pos translated into this rendered
 * buffer. */
static int linenoiseRenderBuffer(struct linenoiseState *l, char **out, size_t *outlen, size_t *outpos) {
    struct linenoiseFolds fs;
    size_t len, pos, src, dst;
    char *r;
    int j, pos_set = 0;

    if (!linenoiseGetRenderFolds(l,&fs)) {
        /* Keep the refresh code simple: it always owns a temporary render
         * buffer, even when the render is identical to the real edit buffer. */
        r = malloc(l->len+1);
        if (r == NULL) return -1;
        memcpy(r,l->buf,l->len);
        r[l->len] = '\0';
        *out = r;
        *outlen = l->len;
        *outpos = l->pos;
        return 0;
    }

    /* Gaps are copied as-is, folded ranges are replaced by their markers.
     * The bytes inside each [start,end) range stay in l->buf but are not
     * emitted to the terminal. */
    len = l->len;
    for (j = 0; j < fs.count; j++) {
        struct linenoiseFold *f = fs.fold+j;
        len -= f->end - f->start;
        len += f->displaylen;
    }
    r = malloc(len+1);
    if (r == NULL) return -1;

    src = dst = 0;
    pos = 0;
    for (j = 0; j < fs.count; j++) {
        struct linenoiseFold *f = fs.fold+j;
        size_t gap = f->start - src;

        if (!pos_set && l->pos <= f->start) {
            pos = dst + (l->pos - src);
            pos_set = 1;
        }
        memcpy(r+dst,l->buf+src,gap);
        dst += gap;

        if (!pos_set && l->pos < f->end) {
            pos = dst + f->displaylen;
            pos_set = 1;
        }
        memcpy(r+dst,f->display,f->displaylen);
        dst += f->displaylen;
        if (!pos_set && l->pos == f->end) {
            pos = dst;
            pos_set = 1;
        }
        src = f->end;
    }
    if (!pos_set) pos = dst + (l->pos - src);
    memcpy(r+dst,l->buf+src,l->len-src);
    r[len] = '\0';

    *out = r;
    *outlen = len;
    *outpos = pos;
    return 0;
}

/* Return the number of bytes to move right from pos. If pos is at the start of
 * a folded range, the whole hidden range is skipped by one cursor movement. */
static size_t linenoiseEditNextLen(struct linenoiseState *l, size_t pos) {
    struct linenoiseFolds fs;
    int j;

    if (linenoiseGetRenderFolds(l,&fs)) {
        for (j = 0; j < fs.count; j++) {
            if (pos == fs.fold[j].start)
                return fs.fold[j].end - fs.fold[j].start;
        }
    }
    return utf8NextCharLen(l->buf,pos,l->len);
}

/* Return the number of bytes to move left from pos. If pos is at the end of a
 * folded range, the whole hidden range is skipped by one cursor movement. */
static size_t linenoiseEditPrevLen(struct linenoiseState *l, size_t pos) {
    struct linenoiseFolds fs;
    int j;

    if (linenoiseGetRenderFolds(l,&fs)) {
        for (j = 0; j < fs.count; j++) {
            if (pos == fs.fold[j].end)
                return fs.fold[j].end - fs.fold[j].start;
        }
    }
    return utf8PrevCharLen(l->buf,pos);
}

/* Add a fold range, keeping the array sorted by start offset. */
static void linenoiseFoldAdd(struct linenoiseState *l, size_t start, size_t end) {
    int j;

    if (start >= end || l->fold_count == LINENOISE_MAX_FOLDS) return;
    j = l->fold_count;
    while (j > 0 && start < l->fold_start[j-1]) {
        l->fold_start[j] = l->fold_start[j-1];
        l->fold_end[j] = l->fold_end[j-1];
        j--;
    }
    l->fold_start[j] = start;
    l->fold_end[j] = end;
    l->fold_count++;
}

/* Clear all remembered fold ranges. */
static void linenoiseFoldClear(struct linenoiseState *l) {
    l->fold_count = 0;
}

/* Remove one remembered fold range. */
static void linenoiseFoldRemove(struct linenoiseState *l, int j) {
    memmove(l->fold_start+j,l->fold_start+j+1,
            sizeof(size_t)*(l->fold_count-j-1));
    memmove(l->fold_end+j,l->fold_end+j+1,
            sizeof(size_t)*(l->fold_count-j-1));
    l->fold_count--;
}

/* Return true if [pos,pos+len) overlaps any folded range. */
static int linenoiseRangeOverlapsFold(struct linenoiseState *l, size_t pos, size_t len) {
    size_t end = pos + len;
    int j;

    for (j = 0; j < l->fold_count; j++) {
        if (end > l->fold_start[j] && pos < l->fold_end[j])
            return 1;
    }
    return 0;
}

/* Adjust fold ranges after an insertion. If insertion somehow lands inside a
 * fold, remove that fold because it no longer maps to an unchanged range. */
static void linenoiseAdjustFoldsAfterInsert(struct linenoiseState *l, size_t pos, size_t len) {
    int j = 0;

    while (j < l->fold_count) {
        if (pos <= l->fold_start[j]) {
            l->fold_start[j] += len;
            l->fold_end[j] += len;
            j++;
        } else if (pos < l->fold_end[j]) {
            linenoiseFoldRemove(l,j);
        } else {
            j++;
        }
    }
}

/* Adjust fold ranges after a deletion. If deletion overlaps a fold, remove
 * that fold because it no longer maps to an unchanged range. */
static void linenoiseAdjustFoldsAfterDelete(struct linenoiseState *l, size_t pos, size_t len) {
    size_t end = pos + len;
    int j = 0;

    while (j < l->fold_count) {
        if (end <= l->fold_start[j]) {
            l->fold_start[j] -= len;
            l->fold_end[j] -= len;
            j++;
        } else if (pos >= l->fold_end[j]) {
            j++;
        } else {
            linenoiseFoldRemove(l,j);
        }
    }
}

/* Helper of refreshSingleLine() and refreshMultiLine() to show hints
 * to the right of the prompt. Now uses display widths for proper UTF-8. */
void refreshShowHints(struct abuf *ab, struct linenoiseState *l, int pwidth, size_t bufwidth) {
    if (hintsCallback) {
        char seq[64];
        int color = -1, bold = 0;
        char *hint;

        if (pwidth + bufwidth >= l->cols) return;
        hint = hintsCallback(l->buf,&color,&bold);
        if (hint) {
            size_t hintlen = strlen(hint);
            size_t hintwidth = utf8StrWidth(hint, hintlen);
            size_t hintmaxwidth = l->cols - (pwidth + bufwidth);
            /* Truncate hint to fit, respecting UTF-8 boundaries. */
            if (hintwidth > hintmaxwidth) {
                size_t i = 0, w = 0;
                while (i < hintlen) {
                    size_t clen = utf8NextCharLen(hint, i, hintlen);
                    int cwidth = utf8SingleCharWidth(hint + i, clen);
                    if (w + cwidth > hintmaxwidth) break;
                    w += cwidth;
                    i += clen;
                }
                hintlen = i;
            }
            if (bold == 1 && color == -1) color = 37;
            if (color != -1 || bold != 0)
                snprintf(seq,64,"\033[%d;%d;49m",bold,color);
            else
                seq[0] = '\0';
            abAppend(ab,seq,strlen(seq));
            abAppend(ab,hint,hintlen);
            if (color != -1 || bold != 0)
                abAppend(ab,"\033[0m",4);
            /* Call the function to free the hint returned. */
            if (freeHintsCallback) freeHintsCallback(hint);
        }
    }
}

/* Single line low level line refresh.
 *
 * Rewrite the currently edited line accordingly to the buffer content,
 * cursor position, and number of columns of the terminal.
 *
 * Flags is REFRESH_* macros. The function can just remove the old
 * prompt, just write it, or both.
 *
 * This function is UTF-8 aware and uses display widths (not byte counts)
 * for cursor positioning and horizontal scrolling. */
static void refreshSingleLine(struct linenoiseState *l, int flags) {
    char seq[64];
    size_t pwidth = utf8StrWidth(l->prompt, l->plen); /* Prompt display width */
    int fd = l->ofd;
    char *render = NULL;
    char *buf;
    size_t len;             /* Byte length of buffer to display */
    size_t pos;             /* Byte position of cursor in display buffer */
    size_t poscol;          /* Display column of cursor */
    size_t lencol;          /* Display width of buffer */
    size_t fullwidth;        /* Display width before horizontal trimming. */
    struct abuf ab;

    if (linenoiseRenderBuffer(l,&render,&len,&pos) == -1) return;
    buf = render;

    /* Calculate the display width up to cursor and total display width. */
    poscol = utf8StrWidth(buf, pos);
    lencol = utf8StrWidth(buf, len);
    fullwidth = lencol;

    /* Scroll the buffer horizontally if cursor is past the right edge.
     * We need to trim full UTF-8 characters from the left until the
     * cursor position fits within the terminal width. */
    while (pwidth + poscol >= l->cols) {
        size_t clen = utf8NextCharLen(buf, 0, len);
        int cwidth = utf8SingleCharWidth(buf, clen);
        buf += clen;
        len -= clen;
        pos -= clen;
        poscol -= cwidth;
        lencol -= cwidth;
    }

    /* Trim from the right if the line still doesn't fit. */
    while (pwidth + lencol > l->cols) {
        size_t clen = utf8PrevCharLen(buf, len);
        int cwidth = utf8SingleCharWidth(buf + len - clen, clen);
        len -= clen;
        lencol -= cwidth;
    }

    abInit(&ab);
    /* Cursor to left edge */
    snprintf(seq,sizeof(seq),"\r");
    abAppend(&ab,seq,strlen(seq));

    if (flags & REFRESH_WRITE) {
        /* Write the prompt and the current buffer content */
        abAppend(&ab,l->prompt,l->plen);
        if (maskmode == 1) {
            /* In mask mode, we output one '*' per UTF-8 character, not byte */
            size_t i = 0;
            while (i < len) {
                abAppend(&ab,"*",1);
                i += utf8NextCharLen(buf, i, len);
            }
        } else {
            abAppend(&ab,buf,len);
        }
        /* Show hints if any. */
        refreshShowHints(&ab,l,pwidth,fullwidth);
    }

    /* Erase to right */
    snprintf(seq,sizeof(seq),"\x1b[0K");
    abAppend(&ab,seq,strlen(seq));

    if (flags & REFRESH_WRITE) {
        /* Move cursor to original position (using display column, not byte). */
        snprintf(seq,sizeof(seq),"\r\x1b[%dC", (int)(poscol+pwidth));
        abAppend(&ab,seq,strlen(seq));
    }

    if (write(fd,ab.b,ab.len) == -1) {} /* Can't recover from write error. */
    abFree(&ab);
    free(render);
}

/* Multi line low level line refresh.
 *
 * Rewrite the currently edited line accordingly to the buffer content,
 * cursor position, and number of columns of the terminal.
 *
 * Flags is REFRESH_* macros. The function can just remove the old
 * prompt, just write it, or both.
 *
 * This function is UTF-8 aware and uses display widths for positioning. */
static void refreshMultiLine(struct linenoiseState *l, int flags) {
    char seq[64];
    size_t pwidth = utf8StrWidth(l->prompt, l->plen);  /* Prompt display width */
    char *render = NULL;
    size_t render_len, render_pos;
    size_t bufwidth;
    size_t poswidth;
    int rows; /* rows used by current rendered buffer. */
    int rpos = l->oldrpos;   /* cursor relative row from previous refresh. */
    int rpos2; /* rpos after refresh. */
    int col; /* column position, zero-based. */
    int old_rows = l->oldrows;
    int fd = l->ofd, j;
    struct abuf ab;

    if (linenoiseRenderBuffer(l,&render,&render_len,&render_pos) == -1) return;
    bufwidth = utf8StrWidth(render, render_len);
    poswidth = utf8StrWidth(render, render_pos);
    rows = (pwidth+bufwidth+l->cols-1)/l->cols;
    l->oldrows = rows;

    /* First step: clear all the lines used before. To do so start by
     * going to the last row. */
    abInit(&ab);

    if (flags & REFRESH_CLEAN) {
        if (old_rows-rpos > 0) {
            lndebug("go down %d", old_rows-rpos);
            snprintf(seq,64,"\x1b[%dB", old_rows-rpos);
            abAppend(&ab,seq,strlen(seq));
        }

        /* Now for every row clear it, go up. */
        for (j = 0; j < old_rows-1; j++) {
            lndebug("clear+up");
            snprintf(seq,64,"\r\x1b[0K\x1b[1A");
            abAppend(&ab,seq,strlen(seq));
        }
    }

    if (flags & REFRESH_ALL) {
        /* Clean the top line. */
        lndebug("clear");
        snprintf(seq,64,"\r\x1b[0K");
        abAppend(&ab,seq,strlen(seq));
    }

    if (flags & REFRESH_WRITE) {
        /* Write the prompt and the current buffer content */
        abAppend(&ab,l->prompt,l->plen);
        if (maskmode == 1) {
            /* In mask mode, output one '*' per UTF-8 character, not byte */
            size_t i = 0;
            while (i < render_len) {
                abAppend(&ab,"*",1);
                i += utf8NextCharLen(render, i, render_len);
            }
        } else {
            abAppend(&ab,render,render_len);
        }

        /* Show hints if any. */
        refreshShowHints(&ab,l,pwidth,bufwidth);

        /* If we are at the very end of the screen with our prompt, we need to
         * emit a newline and move the prompt to the first column. */
        if (l->pos &&
            render_pos == render_len &&
            (poswidth+pwidth) % l->cols == 0)
        {
            lndebug("<newline>");
            abAppend(&ab,"\n",1);
            snprintf(seq,64,"\r");
            abAppend(&ab,seq,strlen(seq));
            rows++;
            if (rows > (int)l->oldrows) l->oldrows = rows;
        }

        /* Move cursor to right position. */
        rpos2 = (pwidth+poswidth+l->cols)/l->cols; /* Current cursor relative row */
        lndebug("rpos2 %d", rpos2);

        /* Go up till we reach the expected position. */
        if (rows-rpos2 > 0) {
            lndebug("go-up %d", rows-rpos2);
            snprintf(seq,64,"\x1b[%dA", rows-rpos2);
            abAppend(&ab,seq,strlen(seq));
        }

        /* Set column. */
        col = (pwidth+poswidth) % l->cols;
        lndebug("set col %d", 1+col);
        if (col)
            snprintf(seq,64,"\r\x1b[%dC", col);
        else
            snprintf(seq,64,"\r");
        abAppend(&ab,seq,strlen(seq));
    }

    lndebug("\n");
    l->oldpos = l->pos;
    if (flags & REFRESH_WRITE) l->oldrpos = rpos2;

    if (write(fd,ab.b,ab.len) == -1) {} /* Can't recover from write error. */
    abFree(&ab);
    free(render);
}

/* Calls the two low level functions refreshSingleLine() or
 * refreshMultiLine() according to the selected mode. */
static void refreshLineWithFlags(struct linenoiseState *l, int flags) {
    if (mlmode)
        refreshMultiLine(l,flags);
    else
        refreshSingleLine(l,flags);
}

/* Utility function to avoid specifying REFRESH_ALL all the times. */
static void refreshLine(struct linenoiseState *l) {
    refreshLineWithFlags(l,REFRESH_ALL);
}

/* Hide the current line, when using the multiplexing API. */
void linenoiseHide(struct linenoiseState *l) {
    if (mlmode)
        refreshMultiLine(l,REFRESH_CLEAN);
    else
        refreshSingleLine(l,REFRESH_CLEAN);
}

/* Show the current line, when using the multiplexing API. */
void linenoiseShow(struct linenoiseState *l) {
    if (l->in_completion) {
        refreshLineWithCompletion(l,NULL,REFRESH_WRITE);
    } else {
        refreshLineWithFlags(l,REFRESH_WRITE);
    }
}

/* Grow the editing buffer if this state owns a growable buffer. Only the
 * blocking linenoise() API sets buflen_max: the multiplexing API still uses
 * the caller-provided fixed buffer. */
static int linenoiseEditGrow(struct linenoiseState *l, size_t needed) {
    size_t newlen;
    char *newbuf;

    if (needed <= l->buflen) return 0;

    /* buflen_max is zero when the caller provided a fixed buffer, as in the
     * multiplexing API: in that case there is nothing we can grow. */
    if (l->buflen_max == 0 || needed > l->buflen_max) return -1;

    /* Grow exponentially, but stop at the configured maximum before the
     * doubling would overflow or go past it. */
    newlen = l->buflen ? l->buflen : 16;
    while (newlen < needed) {
        if (newlen > l->buflen_max/2) {
            newlen = l->buflen_max;
            break;
        }
        newlen *= 2;
    }
    if (newlen < needed || newlen == SIZE_MAX) return -1;

    /* Allocate one extra byte for the nul terminator. */
    newbuf = realloc(l->buf,newlen+1);
    if (newbuf == NULL) return -1;
    l->buf = newbuf;
    l->buflen = newlen;
    return 0;
}

/* Insert bytes into l->buf without repainting the prompt. The paste path uses
 * this to first store the real pasted bytes, then mark their range as folded,
 * and only then refresh so raw pasted newlines are never printed directly. */
static int linenoiseEditInsertNoRefresh(struct linenoiseState *l, const char *c, size_t clen) {
    size_t insert_pos = l->pos;

    if (clen > SIZE_MAX-l->len || linenoiseEditGrow(l,l->len+clen) == -1)
        return -1;

    if (l->len == l->pos) {
        memcpy(l->buf+l->pos,c,clen);
    } else {
        memmove(l->buf+l->pos+clen,l->buf+l->pos,l->len-l->pos);
        memcpy(l->buf+l->pos,c,clen);
    }
    l->pos += clen;
    l->len += clen;
    l->buf[l->len] = '\0';
    linenoiseAdjustFoldsAfterInsert(l,insert_pos,clen);
    return 0;
}

/* Insert the character(s) 'c' of length 'clen' at cursor current position.
 * This handles both single-byte ASCII and multi-byte UTF-8 sequences.
 *
 * On error writing to the terminal -1 is returned, otherwise 0. */
int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen) {
    if (l->len == l->pos) {
        int needs_refresh = memchr(c, '\n', clen) != NULL ||
                             memchr(c, '\r', clen) != NULL;

        if (linenoiseEditInsertNoRefresh(l,c,clen) == -1) return 0;
        if (!needs_refresh && !mlmode && !hintsCallback &&
            (maskmode || l->fold_count == 0))
        {
            size_t bufwidth = utf8StrWidth(l->buf,l->len);
            if (utf8StrWidth(l->prompt,l->plen)+bufwidth < l->cols) {
                /* Avoid a full update of the line in the trivial case:
                 * single-width char, no hints, fits in one line. */
                if (maskmode == 1) {
                    if (write(l->ofd,"*",1) == -1) return -1;
                } else {
                    if (write(l->ofd,c,clen) == -1) return -1;
                }
                return 0;
            }
        }
        refreshLine(l);
    } else {
        if (linenoiseEditInsertNoRefresh(l,c,clen) == -1) return 0;
        refreshLine(l);
    }
    return 0;
}

/* Move cursor on the left. Moves by one UTF-8 character, not byte. */
void linenoiseEditMoveLeft(struct linenoiseState *l) {
    if (l->pos > 0) {
        l->pos -= linenoiseEditPrevLen(l, l->pos);
        refreshLine(l);
    }
}

/* Move cursor on the right. Moves by one UTF-8 character, not byte. */
void linenoiseEditMoveRight(struct linenoiseState *l) {
    if (l->pos != l->len) {
        l->pos += linenoiseEditNextLen(l, l->pos);
        refreshLine(l);
    }
}

/* Move cursor to the start of the line. */
void linenoiseEditMoveHome(struct linenoiseState *l) {
    if (l->pos != 0) {
        l->pos = 0;
        refreshLine(l);
    }
}

/* Move cursor to the end of the line. */
void linenoiseEditMoveEnd(struct linenoiseState *l) {
    if (l->pos != l->len) {
        l->pos = l->len;
        refreshLine(l);
    }
}

/* Substitute the currently edited line with the next or previous history
 * entry as specified by 'dir'. */
#define LINENOISE_HISTORY_NEXT 0
#define LINENOISE_HISTORY_PREV 1
void linenoiseEditHistoryNext(struct linenoiseState *l, int dir) {
    if (history_len > 1) {
        const char *src;
        size_t len;
        struct linenoiseFold f;

        /* Update the current history entry before to
         * overwrite it with the next one. */
        free(history[history_len - 1 - l->history_index]);
        history[history_len - 1 - l->history_index] = strdup(l->buf);
        /* Show the new entry */
        l->history_index += (dir == LINENOISE_HISTORY_PREV) ? 1 : -1;
        if (l->history_index < 0) {
            l->history_index = 0;
            return;
        } else if (l->history_index >= history_len) {
            l->history_index = history_len-1;
            return;
        }

        /* Copy the selected history entry into the edit buffer. With the
         * fixed-buffer API, truncate the entry if it does not fit. */
        src = history[history_len - 1 - l->history_index];
        len = strlen(src);
        if (linenoiseEditGrow(l,len) == -1 && len > l->buflen)
            len = l->buflen;
        memcpy(l->buf,src,len);
        l->buf[len] = '\0';
        l->len = l->pos = len;
        linenoiseFoldClear(l);

        /* History stores the real text, but not the original paste ranges.
         * If the recalled entry needs folding, create one display fold now
         * so text typed after recall remains outside the folded range. */
        if (linenoiseBuildHistoryFold(l,&f))
            linenoiseFoldAdd(l,f.start,f.end);
        refreshLine(l);
    }
}

/* Delete the character at the right of the cursor without altering the cursor
 * position. Basically this is what happens with the "Delete" keyboard key.
 * Now handles multi-byte UTF-8 characters. */
void linenoiseEditDelete(struct linenoiseState *l) {
    if (l->len > 0 && l->pos < l->len) {
        size_t clen = linenoiseEditNextLen(l, l->pos);
        linenoiseAdjustFoldsAfterDelete(l,l->pos,clen);
        memmove(l->buf+l->pos, l->buf+l->pos+clen, l->len-l->pos-clen);
        l->len -= clen;
        l->buf[l->len] = '\0';
        refreshLine(l);
    }
}

/* Backspace implementation. Deletes the UTF-8 character before the cursor. */
void linenoiseEditBackspace(struct linenoiseState *l) {
    if (l->pos > 0 && l->len > 0) {
        size_t clen = linenoiseEditPrevLen(l, l->pos);
        linenoiseAdjustFoldsAfterDelete(l,l->pos-clen,clen);
        memmove(l->buf+l->pos-clen, l->buf+l->pos, l->len-l->pos);
        l->pos -= clen;
        l->len -= clen;
        l->buf[l->len] = '\0';
        refreshLine(l);
    }
}

/* Delete the previous word, maintaining the cursor at the start of the
 * current word. Handles UTF-8 by moving character-by-character. */
void linenoiseEditDeletePrevWord(struct linenoiseState *l) {
    size_t old_pos = l->pos;
    size_t diff;

    /* Skip spaces before the word (move backwards by UTF-8 chars). */
    while (l->pos > 0 && l->buf[l->pos-1] == ' ')
        l->pos -= linenoiseEditPrevLen(l, l->pos);
    /* Skip non-space characters (move backwards by UTF-8 chars). */
    while (l->pos > 0 && l->buf[l->pos-1] != ' ')
        l->pos -= linenoiseEditPrevLen(l, l->pos);
    diff = old_pos - l->pos;
    linenoiseAdjustFoldsAfterDelete(l,l->pos,diff);
    memmove(l->buf+l->pos, l->buf+old_pos, l->len-old_pos+1);
    l->len -= diff;
    refreshLine(l);
}

/* This function is part of the multiplexed API of Linenoise, that is used
 * in order to implement the blocking variant of the API but can also be
 * called by the user directly in an event driven program. It will:
 *
 * 1. Initialize the linenoise state passed by the user.
 * 2. Put the terminal in RAW mode.
 * 3. Show the prompt.
 * 4. Return control to the user, that will have to call linenoiseEditFeed()
 *    each time there is some data arriving in the standard input.
 *
 * The user can also call linenoiseEditHide() and linenoiseEditShow() if it
 * is required to show some input arriving asyncronously, without mixing
 * it with the currently edited line.
 *
 * When linenoiseEditFeed() returns non-NULL, the user finished with the
 * line editing session (pressed enter CTRL-D/C): in this case the caller
 * needs to call linenoiseEditStop() to put back the terminal in normal
 * mode. This will not destroy the buffer, as long as the linenoiseState
 * is still valid in the context of the caller.
 *
 * The function returns 0 on success, or -1 if writing to standard output
 * fails. If stdin_fd or stdout_fd are set to -1, the default is to use
 * STDIN_FILENO and STDOUT_FILENO.
 */
int linenoiseEditStart(struct linenoiseState *l, int stdin_fd, int stdout_fd, char *buf, size_t buflen, const char *prompt) {
    /* Populate the linenoise state that we pass to functions implementing
     * specific editing functionalities. */
    l->in_completion = 0;
    l->ifd = stdin_fd != -1 ? stdin_fd : STDIN_FILENO;
    l->ofd = stdout_fd != -1 ? stdout_fd : STDOUT_FILENO;
    l->buf = buf;
    l->buflen = buflen;
    l->buflen_max = 0;
    l->prompt = prompt;
    l->plen = strlen(prompt);
    l->oldpos = l->pos = 0;
    l->len = 0;
    linenoiseFoldClear(l);

    /* Enter raw mode. */
    rawmode_output = l->ofd;
    if (enableRawMode(l->ifd) == -1) return -1;

    l->cols = getColumns(stdin_fd, stdout_fd);
    l->oldrows = 0;
    l->oldrpos = 1;  /* Cursor starts on row 1. */
    l->history_index = 0;

    /* Buffer starts empty. */
    l->buf[0] = '\0';
    l->buflen--; /* Make sure there is always space for the nulterm */

    /* If stdin is not a tty, stop here with the initialization. We
     * will actually just read a line from standard input in blocking
     * mode later, in linenoiseEditFeed(). */
    if (!isatty(l->ifd) && !getenv("LINENOISE_ASSUME_TTY")) return 0;

    /* The latest history entry is always our current buffer, that
     * initially is just an empty string. */
    linenoiseHistoryAdd("");

    if (write(l->ofd,prompt,l->plen) == -1) return -1;
    return 0;
}

/* Make sure the temporary paste buffer can hold len+need bytes. Return -1 on
 * allocation failure or if the requested size is over PASTE_MAX_BYTES. */
static int pasteBufferReserve(char **buf, size_t *cap, size_t len, size_t need) {
    size_t want;
    char *nb;

    /* Nothing to do if the current paste buffer already has room for the
     * bytes collected so far plus the new bytes we want to append. */
    if (*cap >= len + need) return 0;

    /* Start small, then double like the line buffer. The cap avoids turning a
     * huge paste into an unbounded allocation attempt. */
    want = *cap ? *cap : 64;
    while (want < len + need) {
        size_t doubled = want*2;
        if (doubled <= want || doubled > PASTE_MAX_BYTES) {
            want = PASTE_MAX_BYTES;
            break;
        }
        want = doubled;
    }
    if (want < len + need) return -1;

    /* realloc(NULL, want) handles the first allocation too. */
    nb = realloc(*buf, want);
    if (nb == NULL) return -1;
    *buf = nb;
    *cap = want;
    return 0;
}

/* Append bytes to the temporary paste buffer, growing both it and l->buf as
 * needed. Return -1 if the paste is too large or allocation fails. */
static int pasteBufferAppend(struct linenoiseState *l, char **buf, size_t *cap,
                             size_t *len, const char *s, size_t slen, size_t maxlen) {
    size_t needed;

    if (*len > maxlen || slen > maxlen-*len) return -1;
    if (*len > SIZE_MAX-slen) return -1;
    needed = *len+slen;
    if (l->len > SIZE_MAX-needed) return -1;
    if (linenoiseEditGrow(l,l->len+needed) == -1) return -1;
    if (pasteBufferReserve(buf,cap,*len,slen) == -1) return -1;
    memcpy(*buf+*len,s,slen);
    *len = needed;
    return 0;
}

/* Read a bracketed paste until ESC[201~ and insert the real bytes. If folding
 * is needed, remember the inserted range so only rendering is shortened. */
static void linenoiseEditPaste(struct linenoiseState *l) {
    static const char END[] = "\x1b[201~";
    const size_t ENDLEN = sizeof(END)-1;
    char *buf = NULL;
    size_t cap = 0, len = 0, match = 0;
    size_t maxlen = l->buflen_max ? l->buflen_max : l->buflen;
    int overflowed = 0;

    maxlen = maxlen > l->len ? maxlen - l->len : 0;
    if (maxlen > PASTE_MAX_BYTES) maxlen = PASTE_MAX_BYTES;
    /* Once all fold slots are used, consume later pastes without storing them. */
    if (l->fold_count == LINENOISE_MAX_FOLDS) maxlen = 0;

    while (1) {
        char c;
        if (read(l->ifd, &c, 1) != 1) break;

        /* Track a possible ESC[201~ terminator without copying it into the
         * paste. If it turns out to be ordinary input, flush the partial
         * match below. */
        if (c == END[match]) {
            match++;
            if (match == ENDLEN) break;
            continue;
        }

        if (match > 0) {
            if (!overflowed &&
                pasteBufferAppend(l,&buf,&cap,&len,END,match,maxlen) == -1)
                overflowed = 1;
            match = 0;
            if (c == END[0]) {
                match = 1;
                continue;
            }
        }

        if (!overflowed &&
            pasteBufferAppend(l,&buf,&cap,&len,&c,1,maxlen) == -1)
            overflowed = 1;
    }

    if (overflowed) {
        free(buf);
        linenoiseBeep();
        return;
    }
    if (buf == NULL) return;

    {
        /* Normalize pasted CR and CRLF to LF, so the edit buffer uses one
         * internal newline representation. */
        size_t r = 0, w = 0;
        while (r < len) {
            if (buf[r] == '\r') {
                buf[w++] = '\n';
                r += (r+1 < len && buf[r+1] == '\n') ? 2 : 1;
            } else {
                buf[w++] = buf[r++];
            }
        }
        len = w;
    }

    if (!maskmode && shouldFoldText(buf,len)) {
        size_t start = l->pos;
        if (linenoiseEditInsertNoRefresh(l,buf,len) == -1) {
            free(buf);
            linenoiseBeep();
            return;
        }
        linenoiseFoldAdd(l,start,start+len);
        refreshLine(l);
    } else {
        linenoiseEditInsert(l,buf,len);
    }
    free(buf);
}

char *linenoiseEditMore = "If you see this, you are misusing the API: when linenoiseEditFeed() is called, if it returns linenoiseEditMore the user is yet editing the line. See the README file for more information.";

/* This function is part of the multiplexed API of linenoise, see the top
 * comment on linenoiseEditStart() for more information. Call this function
 * each time there is some data to read from the standard input file
 * descriptor. In the case of blocking operations, this function can just be
 * called in a loop, and block.
 *
 * The function returns linenoiseEditMore to signal that line editing is still
 * in progress, that is, the user didn't yet pressed enter / CTRL-D. Otherwise
 * the function returns the pointer to the heap-allocated buffer with the
 * edited line, that the user should free with linenoiseFree().
 *
 * On special conditions, NULL is returned and errno is populated:
 *
 * EAGAIN if the user pressed Ctrl-C
 * ENOENT if the user pressed Ctrl-D
 *
 * Some other errno: I/O error.
 */
char *linenoiseEditFeed(struct linenoiseState *l) {
    /* Not a TTY, pass control to line reading without character
     * count limits. */
    if (!isatty(l->ifd) && !getenv("LINENOISE_ASSUME_TTY")) return linenoiseNoTTY();

    char c;
    int nread;
    char seq[3];

    nread = read(l->ifd,&c,1);
    if (nread < 0) {
        return (errno == EAGAIN || errno == EWOULDBLOCK) ? linenoiseEditMore : NULL;
    } else if (nread == 0) {
        return NULL;
    }

    /* Only autocomplete when the callback is set. completeLine()
     * returns the character to be handled next, or zero when the
     * key was consumed to navigate completions. */
    if ((l->in_completion || c == 9 /* TAB */) && completionCallback != NULL) {
        int retval = completeLine(l,c);
        /* Read next character when 0 */
        if (retval == 0) return linenoiseEditMore;
        c = retval;
    }

    switch(c) {
    case ENTER:    /* enter */
        history_len--;
        free(history[history_len]);
        if (mlmode) linenoiseEditMoveEnd(l);
        if (hintsCallback) {
            /* Force a refresh without hints to leave the previous
             * line as the user typed it after a newline. */
            linenoiseHintsCallback *hc = hintsCallback;
            hintsCallback = NULL;
            refreshLine(l);
            hintsCallback = hc;
        }
        return strdup(l->buf);
    case CTRL_C:     /* ctrl-c */
        errno = EAGAIN;
        return NULL;
    case BACKSPACE:   /* backspace */
    case 8:     /* ctrl-h */
        linenoiseEditBackspace(l);
        break;
    case CTRL_D:     /* ctrl-d, remove char at right of cursor, or if the
                        line is empty, act as end-of-file. */
        if (l->len > 0) {
            linenoiseEditDelete(l);
        } else {
            history_len--;
            free(history[history_len]);
            errno = ENOENT;
            return NULL;
        }
        break;
    case CTRL_T:    /* ctrl-t, swaps current character with previous. */
        /* Handle UTF-8: swap the two UTF-8 characters around cursor. */
        if (l->pos > 0 && l->pos < l->len) {
            char tmp[32];
            size_t prevlen = linenoiseEditPrevLen(l, l->pos);
            size_t currlen = linenoiseEditNextLen(l, l->pos);
            size_t prevstart = l->pos - prevlen;
            if (prevlen > sizeof(tmp) || currlen > sizeof(tmp)) break;
            if (linenoiseRangeOverlapsFold(l,prevstart,prevlen+currlen)) {
                linenoiseBeep();
                break;
            }
            /* Copy current char to tmp, move previous char right, paste tmp. */
            memcpy(tmp, l->buf + l->pos, currlen);
            memmove(l->buf + prevstart + currlen, l->buf + prevstart, prevlen);
            memcpy(l->buf + prevstart, tmp, currlen);
            if (l->pos + currlen <= l->len) l->pos += currlen;
            refreshLine(l);
        }
        break;
    case CTRL_B:     /* ctrl-b */
        linenoiseEditMoveLeft(l);
        break;
    case CTRL_F:     /* ctrl-f */
        linenoiseEditMoveRight(l);
        break;
    case CTRL_P:    /* ctrl-p */
        linenoiseEditHistoryNext(l, LINENOISE_HISTORY_PREV);
        break;
    case CTRL_N:    /* ctrl-n */
        linenoiseEditHistoryNext(l, LINENOISE_HISTORY_NEXT);
        break;
    case ESC:    /* escape sequence */
        /* Read the next two bytes representing the escape sequence.
         * Use two calls to handle slow terminals returning the two
         * chars at different times. */
        if (read(l->ifd,seq,1) == -1) break;
        if (read(l->ifd,seq+1,1) == -1) break;

        /* ESC [ sequences. */
        if (seq[0] == '[') {
            if (seq[1] >= '0' && seq[1] <= '9') {
                char param[8];
                size_t plen = 1;
                char final = 0;

                param[0] = seq[1];
                while (plen < sizeof(param)) {
                    char p;
                    if (read(l->ifd,&p,1) != 1) break;
                    if (p >= '0' && p <= '9') {
                        param[plen++] = p;
                    } else {
                        final = p;
                        break;
                    }
                }
                if (final == '~') {
                    if (plen == 1 && param[0] == '3') {
                        linenoiseEditDelete(l);
                    } else if (plen == 3 && memcmp(param,"200",3) == 0) {
                        linenoiseEditPaste(l);
                    }
                }
            } else {
                switch(seq[1]) {
                case 'A': /* Up */
                    linenoiseEditHistoryNext(l, LINENOISE_HISTORY_PREV);
                    break;
                case 'B': /* Down */
                    linenoiseEditHistoryNext(l, LINENOISE_HISTORY_NEXT);
                    break;
                case 'C': /* Right */
                    linenoiseEditMoveRight(l);
                    break;
                case 'D': /* Left */
                    linenoiseEditMoveLeft(l);
                    break;
                case 'H': /* Home */
                    linenoiseEditMoveHome(l);
                    break;
                case 'F': /* End*/
                    linenoiseEditMoveEnd(l);
                    break;
                }
            }
        }

        /* ESC O sequences. */
        else if (seq[0] == 'O') {
            switch(seq[1]) {
            case 'H': /* Home */
                linenoiseEditMoveHome(l);
                break;
            case 'F': /* End*/
                linenoiseEditMoveEnd(l);
                break;
            }
        }
        break;
    default:
        /* Handle UTF-8 multi-byte sequences. When we receive the first byte
         * of a multi-byte UTF-8 character, read the remaining bytes to
         * complete the sequence before inserting. */
        {
            char utf8[4];
            int utf8len = utf8ByteLen(c);
            utf8[0] = c;
            if (utf8len > 1) {
                /* Read remaining bytes of the UTF-8 sequence. */
                int i;
                for (i = 1; i < utf8len; i++) {
                    if (read(l->ifd, utf8+i, 1) != 1) break;
                }
            }
            if (linenoiseEditInsert(l, utf8, utf8len)) return NULL;
        }
        break;
    case CTRL_U: /* Ctrl+u, delete the whole line. */
        l->buf[0] = '\0';
        l->pos = l->len = 0;
        linenoiseFoldClear(l);
        refreshLine(l);
        break;
    case CTRL_K: /* Ctrl+k, delete from current to end of line. */
        linenoiseAdjustFoldsAfterDelete(l,l->pos,l->len-l->pos);
        l->buf[l->pos] = '\0';
        l->len = l->pos;
        refreshLine(l);
        break;
    case CTRL_A: /* Ctrl+a, go to the start of the line */
        linenoiseEditMoveHome(l);
        break;
    case CTRL_E: /* ctrl+e, go to the end of the line */
        linenoiseEditMoveEnd(l);
        break;
    case CTRL_L: /* ctrl+l, clear screen */
        linenoiseClearScreen();
        refreshLine(l);
        break;
    case CTRL_W: /* ctrl+w, delete previous word */
        linenoiseEditDeletePrevWord(l);
        break;
    }
    return linenoiseEditMore;
}

/* This is part of the multiplexed linenoise API. See linenoiseEditStart()
 * for more information. This function is called when linenoiseEditFeed()
 * returns something different than NULL. At this point the user input
 * is in the buffer, and we can restore the terminal in normal mode. */
void linenoiseEditStop(struct linenoiseState *l) {
    if (!isatty(l->ifd) && !getenv("LINENOISE_ASSUME_TTY")) return;
    disableRawMode(l->ifd);
    printf("\n");
}

/* This just implements a blocking loop for the multiplexed API.
 * In many applications that are not event-drivern, we can just call
 * the blocking linenoise API, wait for the user to complete the editing
 * and return the buffer. This wrapper owns l.buf, so it can let the edit
 * state grow it dynamically for large pasted input. */
static char *linenoiseBlockingEdit(int stdin_fd, int stdout_fd, const char *prompt)
{
    struct linenoiseState l;
    char *buf = malloc(LINENOISE_INITIAL_BUFLEN);
    char *res;

    if (buf == NULL) {
        errno = ENOMEM;
        return NULL;
    }

    if (linenoiseEditStart(&l,stdin_fd,stdout_fd,buf,
                           LINENOISE_INITIAL_BUFLEN,prompt) == -1)
    {
        free(buf);
        return NULL;
    }
    l.buflen_max = LINENOISE_MAX_LINE;
    while((res = linenoiseEditFeed(&l)) == linenoiseEditMore);
    linenoiseEditStop(&l);
    free(l.buf);
    return res;
}

/* This special mode is used by linenoise in order to print scan codes
 * on screen for debugging / development purposes. It is implemented
 * by the linenoise_example program using the --keycodes option. */
void linenoisePrintKeyCodes(void) {
    char quit[4];

    printf("Linenoise key codes debugging mode.\n"
            "Press keys to see scan codes. Type 'quit' at any time to exit.\n");
    rawmode_output = STDOUT_FILENO;
    if (enableRawMode(STDIN_FILENO) == -1) return;
    memset(quit,' ',4);
    while(1) {
        char c;
        int nread;

        nread = read(STDIN_FILENO,&c,1);
        if (nread <= 0) continue;
        memmove(quit,quit+1,sizeof(quit)-1); /* shift string to left. */
        quit[sizeof(quit)-1] = c; /* Insert current char on the right. */
        if (memcmp(quit,"quit",sizeof(quit)) == 0) break;

        printf("'%c' %02x (%d) (type quit to exit)\n",
            isprint(c) ? c : '?', (int)c, (int)c);
        printf("\r"); /* Go left edge manually, we are in raw mode. */
        fflush(stdout);
    }
    disableRawMode(STDIN_FILENO);
}

/* Read a newline-terminated record from fp with no fixed-size stack buffer.
 * Used for non-tty input, unsupported terminals, and history loading. */
static char *linenoiseReadLine(FILE *fp, int *err) {
    char *line = NULL;
    size_t len = 0, cap = 0;

    if (err) *err = 0;

    while(1) {
        if (len+1 >= cap) {
            size_t newcap = cap ? cap*2 : 16;
            char *oldval = line;
            if (newcap <= cap) {
                free(line);
                if (err) *err = 1;
                errno = ENOMEM;
                return NULL;
            }
            line = realloc(line,newcap);
            if (line == NULL) {
                if (oldval) free(oldval);
                if (err) *err = 1;
                return NULL;
            }
            cap = newcap;
        }
        int c = fgetc(fp);
        if (c == EOF || c == '\n') {
            if (c == EOF && len == 0) {
                free(line);
                return NULL;
            } else {
                line[len] = '\0';
                return line;
            }
        } else {
            line[len] = c;
            len++;
        }
    }
}

/* This function is called when linenoise() is called with the standard
 * input file descriptor not attached to a TTY. So for example when the
 * program using linenoise is called in pipe or with a file redirected
 * to its standard input. In this case, we want to be able to return the
 * line regardless of its length. */
static char *linenoiseNoTTY(void) {
    return linenoiseReadLine(stdin,NULL);
}

/* The high level function that is the main API of the linenoise library.
 * This function checks if the terminal has basic capabilities, just checking
 * for a blacklist of stupid terminals, and later either calls the line
 * editing function or uses a simple line reader so that you will be able
 * to type something even in the most desperate of the conditions. */
char *linenoise(const char *prompt) {
    if (!isatty(STDIN_FILENO) && !getenv("LINENOISE_ASSUME_TTY")) {
        /* Not a tty: read from file / pipe. In this mode we don't want any
         * limit to the line size, so we call a function to handle that. */
        return linenoiseNoTTY();
    }

    if (isUnsupportedTerm()) {
        char *retval;
        size_t len;

        printf("%s",prompt);
        fflush(stdout);
        retval = linenoiseNoTTY();
        if (retval == NULL) return NULL;
        len = strlen(retval);
        while(len && retval[len-1] == '\r') {
            len--;
            retval[len] = '\0';
        }
        return retval;
    } else {
        return linenoiseBlockingEdit(STDIN_FILENO,STDOUT_FILENO,prompt);
    }
}

/* This is just a wrapper the user may want to call in order to make sure
 * the linenoise returned buffer is freed with the same allocator it was
 * created with. Useful when the main program is using an alternative
 * allocator. */
void linenoiseFree(void *ptr) {
    if (ptr == linenoiseEditMore) return; // Protect from API misuse.
    free(ptr);
}

/* ================================ History ================================= */

/* Free the history, but does not reset it. Only used when we have to
 * exit() to avoid memory leaks are reported by valgrind & co. */
static void freeHistory(void) {
    if (history) {
        int j;

        for (j = 0; j < history_len; j++)
            free(history[j]);
        free(history);
    }
}

/* At exit we'll try to fix the terminal to the initial conditions. */
static void linenoiseAtExit(void) {
    disableRawMode(STDIN_FILENO);
    freeHistory();
}

/* This is the API call to add a new entry in the linenoise history.
 * It uses a fixed array of char pointers that are shifted (memmoved)
 * when the history max length is reached in order to remove the older
 * entry and make room for the new one, so it is not exactly suitable for huge
 * histories, but will work well for a few hundred of entries.
 *
 * Using a circular buffer is smarter, but a bit more complex to handle. */
int linenoiseHistoryAdd(const char *line) {
    char *linecopy;

    if (history_max_len == 0) return 0;

    /* Initialization on first call. */
    if (history == NULL) {
        history = malloc(sizeof(char*)*history_max_len);
        if (history == NULL) return 0;
        memset(history,0,(sizeof(char*)*history_max_len));
    }

    /* Don't add duplicated lines. */
    if (history_len && !strcmp(history[history_len-1], line)) return 0;

    /* Add an heap allocated copy of the line in the history.
     * If we reached the max length, remove the older line. */
    linecopy = strdup(line);
    if (!linecopy) return 0;
    if (history_len == history_max_len) {
        free(history[0]);
        memmove(history,history+1,sizeof(char*)*(history_max_len-1));
        history_len--;
    }
    history[history_len] = linecopy;
    history_len++;
    return 1;
}

/* Set the maximum length for the history. This function can be called even
 * if there is already some history, the function will make sure to retain
 * just the latest 'len' elements if the new history length value is smaller
 * than the amount of items already inside the history. */
int linenoiseHistorySetMaxLen(int len) {
    char **new;

    if (len < 1) return 0;
    if (history) {
        int tocopy = history_len;

        new = malloc(sizeof(char*)*len);
        if (new == NULL) return 0;

        /* If we can't copy everything, free the elements we'll not use. */
        if (len < tocopy) {
            int j;

            for (j = 0; j < tocopy-len; j++) free(history[j]);
            tocopy = len;
        }
        memset(new,0,sizeof(char*)*len);
        memcpy(new,history+(history_len-tocopy), sizeof(char*)*tocopy);
        free(history);
        history = new;
    }
    history_max_len = len;
    if (history_len > history_max_len)
        history_len = history_max_len;
    return 1;
}

/* Save the history in the specified file. On success 0 is returned
 * otherwise -1 is returned. */
int linenoiseHistorySave(const char *filename) {
    mode_t old_umask = umask(S_IXUSR|S_IRWXG|S_IRWXO);
    FILE *fp;
    int j;

    fp = fopen(filename,"w");
    umask(old_umask);
    if (fp == NULL) return -1;
    fchmod(fileno(fp),S_IRUSR|S_IWUSR);
    for (j = 0; j < history_len; j++) {
        char *p = history[j];
        /* Keep the history file newline-separated: embedded newlines in an
         * entry are stored as CR and converted back by linenoiseHistoryLoad(). */
        while (*p) {
            fputc(*p == '\n' ? '\r' : *p, fp);
            p++;
        }
        fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

/* Load the history from the specified file. If the file does not exist
 * zero is returned and no operation is performed.
 *
 * If the file exists and the operation succeeded 0 is returned, otherwise
 * on error -1 is returned. */
int linenoiseHistoryLoad(const char *filename) {
    FILE *fp = fopen(filename,"r");
    char *buf;
    int err = 0;

    if (fp == NULL) return -1;

    while ((buf = linenoiseReadLine(fp,&err)) != NULL) {
        size_t j;

        /* Rebuild embedded newlines that were saved as CR. */
        for (j = 0; buf[j]; j++) {
            if (buf[j] == '\r') buf[j] = '\n';
        }
        linenoiseHistoryAdd(buf);
        free(buf);
    }
    if (err || ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}
