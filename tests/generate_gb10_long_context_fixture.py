#!/usr/bin/env python3
"""Generate the deterministic >100K-token GB10 benchmark fixture.

The canonical story remains unchanged. This fixture repeats only its narrative
body, keeps one final fact-recall query, and uses identical assignment facts in
every copy so the expected answer is unambiguous.
"""

from pathlib import Path

from generate_long_context_story_prompt import (
    ASSISTANT,
    BOS,
    FACTS,
    USER,
    make_story,
)


FULL_BODY_COPIES = 3
PARTIAL_BODY_PARAGRAPHS = 125
OUTPUT_NAME = "long_context_story_prompt_gb10.txt"
FINAL_MARKER = "\nFinal task:\n"


def make_fixture() -> str:
    story = make_story()
    body, marker, final_tail = story.partition(FINAL_MARKER)
    if not marker:
        raise RuntimeError("canonical story is missing the final-task marker")

    paragraphs = body.split("\n\n")
    if PARTIAL_BODY_PARAGRAPHS > len(paragraphs):
        raise RuntimeError("partial-copy paragraph count exceeds canonical body")

    copies = []
    for index in range(FULL_BODY_COPIES):
        copies.append(
            "\n\n===== IDENTICAL CANONICAL LEDGER COPY "
            f"{index + 1} OF {FULL_BODY_COPIES} =====\n\n"
        )
        copies.append(body)

    copies.append(
        "\n\n===== DETERMINISTIC PREFIX OF CANONICAL LEDGER COPY 4 =====\n\n"
    )
    copies.append("\n\n".join(paragraphs[:PARTIAL_BODY_PARAGRAPHS]))

    system = (
        "You are a careful assistant. Read the repeated ledger, remember the "
        "assignments, and answer the single final task exactly. Repeated copies "
        "contain identical assignments and never create a new value."
    )
    fixture = (
        BOS
        + system
        + USER
        + "".join(copies)
        + FINAL_MARKER
        + final_tail
        + ASSISTANT
        + "</think>"
    )
    if fixture.count(FINAL_MARKER) != 1:
        raise RuntimeError("extended fixture must contain exactly one final task")
    for name, word, _value in FACTS:
        assignment = f"{name} was assigned the number {word}."
        if fixture.count(assignment) < FULL_BODY_COPIES:
            raise RuntimeError(f"extended fixture is missing repeated fact: {name}")
    return fixture


def main() -> None:
    output = Path(__file__).resolve().parent / OUTPUT_NAME
    output.write_text(make_fixture(), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
