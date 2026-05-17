#!/usr/bin/env python3
"""Generate the long-context fact-recall prompt used by ds4_test.

The fixture is intentionally prose instead of a synthetic table.  The model has
to retrieve person -> number assignments scattered through a long story, convert
the spelled-out numbers to digits, and emit a parseable list.
"""

from __future__ import annotations

import random
from pathlib import Path

BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"

FACTS = [
    ("Bob", "thirty-four", 34),
    ("Alice", "fifty-two", 52),
    ("Clara", "seventy-one", 71),
    ("Diego", "ninety-three", 93),
    ("Elena", "sixteen", 16),
    ("Felix", "eighty-eight", 88),
    ("Greta", "forty-seven", 47),
    ("Hugo", "twenty-nine", 29),
    ("Iris", "sixty-four", 64),
    ("Jonas", "twelve", 12),
    ("Kira", "eighty-one", 81),
    ("Leo", "thirty-nine", 39),
    ("Marta", "seventy-six", 76),
    ("Nadia", "twenty-three", 23),
    ("Owen", "fifty-eight", 58),
    ("Priya", "ninety-seven", 97),
]

OPENING = """\
You are reading a long story from the harbor town of Bellwether. The story is
ordinary on purpose: people speak, walk, remember, repair things, argue about
weather, and sometimes receive a private assignment number written out in
words. Your job at the end is to recover the assignment numbers.

Important rule while reading: only assignments stated as "was assigned the
number ..." count. Other ages, prices, dates, distances, room numbers, rumors,
or guesses do not count. The assignment numbers in the story are written in
words, not numerals.

"""

SCENE_TEMPLATES = [
    """\
At first light the harbor smelled of rope, rain, and cedar smoke. {lead} crossed
the quay with a folded map tucked under one arm, stopping whenever gulls made a
mess of the chalk marks near the fish stalls. {friend} had promised to fix the
south gate before supper, but the hinges complained so loudly that everyone
pretended not to hear them. In the bakery window, loaves cooled beneath linen
while a child counted shells in a wooden bowl. No one was in a hurry, because
Bellwether moved by tide and habit, not by the bells on the council tower.

The archivist Mara wrote notes in brown ink, never black, because black ink made
old ledgers look like court summonses. She watched {lead} and {friend} pass the
fountain, then added a line about the morning fog. Her notes often wandered into
small details: the color of a scarf, the chipped rim of a cup, the way a door
kept opening after it had been firmly shut.
""",
    """\
By noon the market had filled with baskets of pears, lamp oil, brass hooks, and
paper flowers. {lead} bargained for twine while {friend} listened to a sailor
describe a storm that seemed to grow taller every time he retold it. The town
clock had stopped again, but nobody agreed on when, so every shopkeeper chose a
different hour and defended it with confidence.

Mara sat outside the apothecary and copied the day's ordinary business into the
festival ledger. She liked ordinary business best. Extraordinary business came
with signatures, seals, and people who leaned over her shoulder. Ordinary
business arrived quietly, sat down, and became history before anyone noticed.
""",
    """\
In the afternoon, a rehearsal for the midsummer play blocked the west road.
{lead} carried a crate of lantern glass through the crowd while {friend} read
lines from a damp script. Someone had painted the moon too blue on the backdrop,
and three people argued about whether a theatrical moon was allowed to be wrong.
The argument lasted longer than the scene.

The ledger lay open on a bench. Mara kept it weighted with two smooth stones
from the beach. She recorded who borrowed the theater ladder, who returned the
wrong kettle, and who claimed the missing red umbrella. The handwriting was calm
even when the town was not.
""",
    """\
Evening brought a quiet wind and the sound of shutters being latched one after
another. {lead} helped carry chairs into the assembly hall, where the floor had
been scrubbed until it smelled faintly of salt. {friend} found a lost button
near the door and pinned it to the notice board with a note that said, simply,
"lonely."

Mara walked the perimeter of the hall with the ledger pressed to her chest. She
had learned that important facts hid best inside unimportant days. A missing
button, a changed route, a corrected name, a number assigned without ceremony:
these were the things that later made sense of everything else.
""",
    """\
Rain arrived after midnight and softened every sound in Bellwether. {lead}
stood beneath the awning of the rope-maker's shop, waiting for {friend}, who had
gone back for a forgotten satchel. The street lamps shone in puddles like coins
that nobody could spend. From the hill, the lighthouse blinked with patient
regularity.

Mara remained awake in the archive room. She sharpened a pencil, rejected it,
and returned to brown ink. The festival ledger had grown heavy with the week:
weather notes, repairs, errands, apologies, and a few facts she underlined only
once so they would not look too important.
""",
]

BRIDGE_SENTENCES = [
    "The town talked around the matter without naming it directly.",
    "A kettle whistled somewhere nearby and broke the silence at exactly the right moment.",
    "Mara did not decorate the sentence; she wanted it to be easy to find later.",
    "The phrase was spoken once, then folded into the rest of the day's business.",
    "No one treated the entry like a puzzle, which is why it survived unchanged.",
    "The ledger page smelled of dust, salt, and the faint sweetness of drying glue.",
]


def assignment_sentence(name: str, word: str) -> str:
    return (
        f"During that same scene, {name} was assigned the number {word}. "
        f"Mara wrote the assignment in words, closed the ledger for a moment, "
        f"and then returned to the smaller gossip of the harbor."
    )


def make_story() -> str:
    rng = random.Random(20260513)
    names = [name for name, _, _ in FACTS]
    fact_by_scene = {7 + i * 11: fact for i, fact in enumerate(FACTS)}
    scenes: list[str] = []

    for scene_index in range(190):
        lead = rng.choice(names)
        friend = rng.choice([n for n in names if n != lead])
        template = SCENE_TEMPLATES[scene_index % len(SCENE_TEMPLATES)]
        scene = template.format(lead=lead, friend=friend)

        if scene_index in fact_by_scene:
            name, word, _ = fact_by_scene[scene_index]
            scene += "\n" + rng.choice(BRIDGE_SENTENCES) + " " + assignment_sentence(name, word) + "\n"
        elif scene_index % 9 == 3:
            scene += (
                "\nMara heard someone mention an old rumor about a numbered key, "
                "but she crossed it out because it was not an assignment and did "
                "not belong in the final list.\n"
            )
        elif scene_index % 11 == 6:
            scene += (
                "\nA shop sign advertised a discount in careful words, but prices "
                "and discounts were not assignment numbers, so Mara ignored them.\n"
            )

        scenes.append(scene)

    final_names = ", ".join(name for name, _, _ in FACTS)
    expected_hint = "Bob=34"
    question = f"""\

Final task:

Compile the assignment ledger from the story. Convert the spelled-out numbers to
ordinary decimal numerals. Write only lines in the form Name=number. The first
example line is {expected_hint}; include that line and all remaining people.

People to list: {final_names}.

No bullets, no prose, no explanation.
"""
    return OPENING + "\n".join(scenes) + question


def main() -> None:
    root = Path(__file__).resolve().parent
    story = make_story()
    rendered = (
        BOS
        + "You are a careful assistant. Read the story, remember the assignments, "
        + "and answer the final task exactly."
        + USER
        + story
        + ASSISTANT
        + "</think>"
    )
    (root / "long_context_story_prompt.txt").write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
