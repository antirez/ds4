#!/usr/bin/env python3
"""Convert a DS4_MOE_RECORD_SELECTED_HOTLIST recording into a hotlist .inc table.

Usage:
    python3 gguf-tools/hotlist_to_inc.py RECORD_FILE OUT.inc [TABLE_NAME]

The output pairs are kept in hits-descending order (re-sorted here, so
hand-merged files work too), because the hotlist loader derives its
rank-based priorities from that order. Rows with out-of-range layers or
experts, or non-positive counts, are rejected.

How to record a hotlist, and how generated tables are wired into a
model, are documented in docs/SSD_STREAMING.md ("Seeding the cache from
a routing hotlist"), including the exact corpus behind the shipped
ds4_streaming_hotlist_v41.inc.
"""

import sys

# The recorder's counting-grid bounds (ds4_metal.m); the engine's loader
# additionally re-checks the model's own layer/expert ranges when seeding.
MAX_LAYER = 80
MAX_EXPERT = 512


def fail(msg):
    sys.stderr.write("hotlist_to_inc: %s\n" % msg)
    sys.exit(1)


def parse(path):
    """Read the recorder's text format: '# ...' headers then
    'layer expert hits weight' rows."""
    rows = []
    selections = 0
    skipped = 0
    try:
        with open(path) as fp:
            for line in fp:
                fields = line.split()
                if not fields:
                    continue
                if fields[0].startswith("#"):
                    continue
                if len(fields) != 4:
                    fail("expected 'layer expert hits weight' in %s, got: %s"
                         % (path, line.strip()))
                layer, expert, hits = int(fields[0]), int(fields[1]), int(fields[2])
                if not (0 <= layer < MAX_LAYER):
                    fail("layer %d out of range 0..%d in %s" % (layer, MAX_LAYER - 1, path))
                if not (0 <= expert < MAX_EXPERT):
                    fail("expert %d out of range 0..%d in %s" % (expert, MAX_EXPERT - 1, path))
                if hits < 0:
                    fail("negative hits for layer %d expert %d in %s" % (layer, expert, path))
                if hits == 0:
                    # Skip like the engine loader does; hand-merged files
                    # may carry zero-count rows.
                    skipped += 1
                    continue
                rows.append((layer, expert, hits))
                selections += hits
    except ValueError:
        fail("non-numeric field in %s" % path)
    except OSError as e:
        fail("cannot read %s: %s" % (path, e))
    return rows, selections, skipped


def main():
    if len(sys.argv) not in (3, 4):
        sys.stderr.write(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) == 4 else "ds4_default_streaming_hotlist_v41"

    rows, selections, skipped = parse(src)
    if not rows:
        fail("no entries in %s" % src)
    rows.sort(key=lambda r: (-r[2], r[0], r[1]))  # recorder order: hits, layer, expert

    with open(dst, "w") as f:
        f.write("/* Generated from routing recorded with\n"
                " * DS4_MOE_RECORD_SELECTED_HOTLIST (%d selections); every\n"
                " * expert recorded selected at least once, %d (layer, expert)\n"
                " * pairs, hits-descending so the loader's rank-based priority\n"
                " * matches observed frequency. Regenerate: see\n"
                " * gguf-tools/hotlist_to_inc.py and docs/SSD_STREAMING.md. */\n"
                % (selections, len(rows)))
        f.write("static const uint16_t %s[][2] = {\n" % name)
        for i in range(0, len(rows), 6):
            f.write("    " + " ".join("{%d,%d}," % (r[0], r[1]) for r in rows[i:i+6]) + "\n")
        f.write("};\n\n"
                "static const uint32_t %s_count =\n"
                "    (uint32_t)(sizeof(%s) /\n"
                "               sizeof(%s[0]));\n"
                % (name, name, name))
    layers = len(set(r[0] for r in rows))
    print("%s: %d pairs, %d layers, %d selections" % (dst, len(rows), layers, selections))
    if skipped:
        print("%s: skipped %d zero-count rows" % (dst, skipped))


if __name__ == "__main__":
    main()
