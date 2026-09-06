#include "ds4_eval_cases.h"

/* Curated hard cases. Source revisions and redistribution notes are recorded in
 * EVAL_DATA.md. Keep this table data-only; grading belongs in ds4_eval.c. */

const eval_source eval_hard_sources[] = {
    {"MMLU-Pro", "https://github.com/TIGER-AI-Lab/MMLU-Pro", "Apache-2.0"},
    {"OlympiadBench", "https://huggingface.co/datasets/Hothan/OlympiadBench", "Apache-2.0"},
    {"LiveBench", "https://huggingface.co/datasets/livebench/reasoning", "Apache-2.0"},
    {"NIST Juliet", "https://samate.nist.gov/SARD/test-suites/112", "CC0-1.0"},
};

const size_t eval_hard_source_count =
    sizeof(eval_hard_sources) / sizeof(eval_hard_sources[0]);

const eval_case eval_hard_cases[] = {
    {
        .source = "MMLU-Pro",
        .id = "2826",
        .domain = "biology",
        .title = "Biology challenge",
        .question = "A gene C mutates to c with a frequency of 2 \303\227 10^-6 per generation. There are no othe"
        "r forces acting on these alleles and mating is random. How many generations are needed to in"
        "crease the frequency of gene c from 2 percent to 3 percent?",
        .choice[0] = "2,500 generations",
        .choice[1] = "6,200 generations",
        .choice[2] = "15,000 generations",
        .choice[3] = "8,764 generations",
        .choice[4] = "10,000 generations",
        .choice[5] = "20,000 generations",
        .choice[6] = "3,000 generations",
        .choice[7] = "1,000 generations",
        .choice[8] = "7,500 generations",
        .choice[9] = "5,128 generations",
        .answer = "J",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "86",
        .domain = "business",
        .title = "Business challenge",
        .question = "The stock of the CCC Corporation is currently valued at $12 and is assumed to possess all th"
        "e properties of geometric Brownian motion. It has an expected annual return of 15%, an annua"
        "l volatility of 20%, and the annual risk-free is 10%. Using a binomial lattice, determine th"
        "e price of a call option on CCC stock maturing in 10 monthes time with a strike price of $14"
        " (Let the distance between nodes on your tree be 1 month in length).",
        .choice[0] = "50.0",
        .choice[1] = "60.0",
        .choice[2] = "55.0",
        .choice[3] = "44.0",
        .choice[4] = "48.0",
        .choice[5] = "53.0",
        .choice[6] = "42.0",
        .choice[7] = "46.0",
        .choice[8] = "51.0",
        .choice[9] = "45.0",
        .answer = "F",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "3546",
        .domain = "chemistry",
        .title = "Chemistry challenge",
        .question = "Suppose that the normalized wavefunction for an electron in a carbon nanotube of length $L=1"
        "0.0 \\mathrm{~nm}$ is: $\\psi=(2 / L)^{1 / 2} \\sin (\\pi x / L)$. Calculate the probability"
        " that the electron is between $x=4.95 \\mathrm{~nm}$ and $5.05 \\mathrm{~nm}$.",
        .choice[0] = "0.020",
        .choice[1] = "0.050",
        .choice[2] = "0.030",
        .choice[3] = "0.045",
        .choice[4] = "0.010",
        .choice[5] = "0.015",
        .choice[6] = "0.035",
        .choice[7] = "0.060",
        .choice[8] = "0.025",
        .choice[9] = "0.040",
        .answer = "A",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "10360",
        .domain = "computer science",
        .title = "Computer Science challenge",
        .question = "Two expressions E and F are said to be unifiable if there are substitutions for the variable"
        "s of E and F that make the expressions lexically identical. In the following three expressio"
        "ns, only w, x, y, and z are variables.\nI. f(w,w)\nII. f(x,1)\nIII. f(y,g(z))\nWhich pairs o"
        "f these expressions is (are) pairs of unifiable expressions?",
        .choice[0] = "(I, III) and (II, III) only",
        .choice[1] = "(II, III) only",
        .choice[2] = "(I, II) and (I, III) only",
        .choice[3] = "(I, II) only",
        .choice[4] = "Only single expressions are unifiable, not pairs",
        .choice[5] = "None of the pairs are unifiable",
        .choice[6] = "(I, II), (I, III), and (II, III)",
        .choice[7] = "All three expressions are unifiable",
        .choice[8] = "(I, II) and (II, III) only",
        .choice[9] = "(I, III) only",
        .answer = "C",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "6844",
        .domain = "economics",
        .title = "Economics challenge",
        .question = "The purpose of \"augmenting\" the Dickey-Fuller test regression is to",
        .choice[0] = "Ensure that there is no heteroscedasticity in the test regression residuals.",
        .choice[1] = "Improve the accuracy of the Dickey-Fuller test",
        .choice[2] = "Ensure that the test regression residuals are negatively skewed",
        .choice[3] = "Ensure that the test regression residuals are normally distributed",
        .choice[4] = "Ensure that all of the non-stationarity is taken into account.",
        .choice[5] = "Ensure that the test regression residuals have no outliers",
        .choice[6] = "Ensure that the test regression residuals are positively skewed",
        .choice[7] = "Ensure that there is no autocorrelation in the test regression residuals",
        .choice[8] = "Ensure that the test regression residuals are balanced",
        .choice[9] = "Increase the complexity of the Dickey-Fuller test",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "11292",
        .domain = "engineering",
        .title = "Engineering challenge",
        .question = "Calculate the entropy change for (a) H_2O (s, - 20\302\260C, 1atm) \\rightarrow H_2O (s, 0"
        "\302\260C, 1atm) (b) H_2O (s, - 20\302\260C, 1atm) \\rightarrow H_2O (1, 20\302\260C, 1atm) "
        "Assume C_p (H_2O, 1) = 18 cal deg^-1 mole^-1, C_p (H_2O, s) = 9 cal deg^-1 mole^-1 and\\Delt"
        "aH_fusion= 1440 cal deg^-1 mole^-1, andalso that the heat capacities are independent of temp"
        "erature.",
        .choice[0] = "0.68eu/mole for (a) and 7.22eu for (b)",
        .choice[1] = "1.27eu/mole for (a) and 7.22eu for (b)",
        .choice[2] = "0.68eu/mole for (a) and 5.27eu for (b)",
        .choice[3] = "1.27eu/mole for (a) and 5.27eu for (b)",
        .answer = "A",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "6034",
        .domain = "health",
        .title = "Health challenge",
        .question = "A 23-year-old woman with bone marrow failure is treated with a large dose of rabbit antithym"
        "ocyte globulin. Ten days later, she develops fever, lymphadenopathy, arthralgias, and erythe"
        "ma on her hands and feet. Which of the following is the most likely cause of these symptoms?",
        .choice[0] = "Activation of complement system",
        .choice[1] = "Overproduction of monocytes",
        .choice[2] = "Cytokine secretion by natural killer cells",
        .choice[3] = "Hyperactive B-lymphocyte response",
        .choice[4] = "Allergic reaction to rabbit proteins",
        .choice[5] = "Immune complex deposition in tissues",
        .choice[6] = "Polyclonal T-lymphocyte activation",
        .choice[7] = "Eosinophil degranulation",
        .choice[8] = "Increased production of interferon-gamma",
        .choice[9] = "Acute graft-versus-host disease",
        .answer = "F",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "4699",
        .domain = "history",
        .title = "History challenge",
        .question = "This question refers to the following information.\n\"At least one of the [world's] societie"
        "s would have to somehow enormously increase its productivity [in order to achieve global heg"
        "emony]. That quantum jump would have to be made before the various scientific, technological"
        ", agricultural, and industrial revolutions on which our post-quantum-leap world rests. It co"
        "uld only be accomplished by exploiting the ecosystems, mineral resources, and human assets o"
        "f whole continents outside the lands of the society making the jump. Western Europe did just"
        " that by means of its brutality and guns and, more important, by geographical and ecological"
        " luck.\"\nCopyright \302\251 2015 Cambridge University Press.\nAlfred Crosby, historian, Eco"
        "logical Imperialism, 2004\nBased on your knowledge of world history, which of the following "
        "contributed LEAST to Western European global hegemony between 1450 and 1750 C.E.?",
        .choice[0] = "The implementation of joint-stock companies",
        .choice[1] = "The exploitation of labor in colonized regions",
        .choice[2] = "The exploration of new sea routes to Asia and the Americas",
        .choice[3] = "The establishment of trading posts along Asian and African coasts",
        .choice[4] = "The exchange of food sources between the Americas and Europe",
        .choice[5] = "The spread of Christianity to non-European regions",
        .choice[6] = "The development and application of steam-powered technologies",
        .choice[7] = "The discovery and colonization of the Americas",
        .choice[8] = "The development of advanced naval technology",
        .choice[9] = "Refinement of gunpowder technologies",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "899",
        .domain = "law",
        .title = "Law challenge",
        .question = "A man and a woman were passengers on a plane. They were seated next to one another in the fi"
        "rst- class section of the plane. Midway through the flight, the woman excused herself and we"
        "nt to the restroom. While she was gone, the man rummaged through the woman's purse and stole"
        " $100 in cash. Unknown to the man, a flight attendant saw him steal the money. As is customa"
        "ry in such situations, the flight attendant radioed ahead, and police officers arrested the "
        "man when the plane landed. The man was subsequently charged with violating an applicable fed"
        "eral larceny statute. During the trial, the prosecuting attorney called the man's wife as a "
        "willing witness against her husband. She proposed to testify that the man confided to her th"
        "at he did steal the woman's money during the flight. The man's attorney objected on the grou"
        "nds of hearsay and privilege. The wife's proposed testimony is",
        .choice[0] = "admissible, because it is privileged, though it is not hearsay.",
        .choice[1] = "admissible, because it is neither hearsay nor privileged.",
        .choice[2] = "admissible, because it is hearsay within a recognized exception and not privileged.",
        .choice[3] = "inadmissible, both because it discloses a privileged communication and because it is hearsay"
        " not within any recognized exception.",
        .choice[4] = "admissible, both because it discloses a privileged communication and because it is hearsay n"
        "ot within any recognized exception.",
        .choice[5] = "inadmissible, because it is hearsay not within any recognized exception, and it discloses a "
        "privileged communication.",
        .choice[6] = "inadmissible, because it discloses a privileged communication, though it is not hearsay.",
        .choice[7] = "admissible, both because it does not disclose a privileged communication and because it is h"
        "earsay within a recognized exception.",
        .choice[8] = "inadmissible, because it is hearsay within a recognized exception, though it is not privileg"
        "ed.",
        .choice[9] = "inadmissible, because it is hearsay not within any recognized exception, though it is not pr"
        "ivileged.",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "7738",
        .domain = "math",
        .title = "Math challenge",
        .question = "Suppose that $f(x)$ is a polynomial that has degree $6$ and $g(x)$ is a polynomial that has "
        "degree $3$. If $h(x)$ is also a polynomial such that $f(g(x)) + g(h(x)) + h(f(x))$ is a poly"
        "nomial of degree $36$, then what is the degree of the polynomial $h$?",
        .choice[0] = "3",
        .choice[1] = "21",
        .choice[2] = "6",
        .choice[3] = "9",
        .choice[4] = "30",
        .choice[5] = "72",
        .choice[6] = "12",
        .choice[7] = "15",
        .choice[8] = "24",
        .choice[9] = "18",
        .answer = "C",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "5078",
        .domain = "other",
        .title = "Other challenge",
        .question = "Are strategies of Arms Control and Deterrence useful in cyber-security?",
        .choice[0] = "No, arms control does not serve any purpose outside the realm of traditional military weapon"
        "s.",
        .choice[1] = "Arms control strategies can be implemented, but deterrence strategies are impossible due to "
        "the anonymous nature of cyber attacks.",
        .choice[2] = "Yes, both strategies are effective in cyber-security as long as there is strong internationa"
        "l cooperation and regulation.",
        .choice[3] = "Effective deterrence requires international cooperation which is achievable in theory, but t"
        "he verification required for traditional arms control strategies is impossible in cyber(ed) "
        "warfare.",
        .choice[4] = "Only arms control is unproblematic, deterrence has failed because international cooperation "
        "cannot be reached and common rules or practices cannot be established.",
        .choice[5] = "Both strategies can only work if there is a clear agreement on what constitutes a cyber weap"
        "on and what constitutes a cyber attack.",
        .choice[6] = "Yes, successful policies of arms control and deterrence have translated well from other secu"
        "rity realms, such as nuclear weapons control strategies.",
        .choice[7] = "Deterrence is ineffective because it encourages an arms race in cyberspace, while arms contr"
        "ol is difficult due to the ease of creating and distributing cyber weapons.",
        .choice[8] = "Both arms control and deterrence strategies are ineffective due to the rapid and unpredictab"
        "le development of cyber warfare technologies.",
        .choice[9] = "No, both strategies are obsolete in the face of cyber warfare's unique challenges such as at"
        "tribution and boundary issues.",
        .answer = "D",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "10775",
        .domain = "philosophy",
        .title = "Philosophy challenge",
        .question = " Construct a complete truth table for the following pairs of propositions. Then, using the t"
        "ruth tables, determine whether the statements are logically equivalent or contradictory. If "
        "neither, determine whether they are consistent or inconsistent. Justify your answers.\n(~M "
        "\342\212\203 ~N) \342\210\250 (O \342\211\241 N) and (~M \302\267 N) \302\267 [(~O \342\210"
        "\250 ~N) \302\267 (O \342\210\250 N)]",
        .choice[0] = "Both logically equivalent and consistent",
        .choice[1] = "Both consistent and contradictory",
        .choice[2] = "Both logically equivalent and contradictory",
        .choice[3] = "Contradictory",
        .choice[4] = "Neither logically equivalent nor contradictory, but inconsistent",
        .choice[5] = "Logically equivalent",
        .choice[6] = "Both inconsistent and contradictory",
        .choice[7] = "Inconsistent",
        .choice[8] = "Neither logically equivalent nor contradictory, but consistent",
        .choice[9] = "Neither logically equivalent nor inconsistent, but contradictory",
        .answer = "D",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "9055",
        .domain = "physics",
        .title = "Physics challenge",
        .question = "Two spaceships approach Earth with equal speeds, as measured by an observer on Earth, but fr"
        "om opposite directions. A meterstick on one spaceship is measured to be 60 cm long by an occ"
        "upant of the other spaceship. What is the speed of each spaceship, as measured by the observ"
        "er on Earth?",
        .choice[0] = "0.4c",
        .choice[1] = "0.1c",
        .choice[2] = "1.0c",
        .choice[3] = "0.9c",
        .choice[4] = "0.2c",
        .choice[5] = "0.5c",
        .choice[6] = "0.6c",
        .choice[7] = "0.3c",
        .choice[8] = "0.8c",
        .choice[9] = "0.7c",
        .answer = "F",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "1995",
        .domain = "psychology",
        .title = "Psychology challenge",
        .question = "An examinee who makes a number of confabulatory responses on the Rorschach has overgeneraliz"
        "ed from a part of the inkblot to the whole. Such responses suggest which of the following?",
        .choice[0] = "an inability to perceive abstract concepts",
        .choice[1] = "high level of stress or anxiety",
        .choice[2] = "a low level of self-esteem",
        .choice[3] = "high intelligence and creativity",
        .choice[4] = "a lack of attention to detail",
        .choice[5] = "a need for immediate gratification",
        .choice[6] = "brain damage, intellectual disability, or emotional disturbance",
        .choice[7] = "a high degree of defensiveness or denial",
        .choice[8] = "a highly imaginative personality",
        .choice[9] = "a high degree of introversion or social withdrawal",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "7754",
        .domain = "math",
        .title = "Math challenge",
        .question = "How many triangles are there whose sides are all integers and whose maximum side length equa"
        "ls 11?",
        .choice[0] = "45",
        .choice[1] = "36",
        .choice[2] = "48",
        .choice[3] = "33",
        .choice[4] = "50",
        .choice[5] = "55",
        .choice[6] = "24",
        .choice[7] = "42",
        .choice[8] = "30",
        .choice[9] = "60",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "3024",
        .domain = "biology",
        .title = "Biology challenge",
        .question = "Two 30\" plants are crossed, resulting in progeny of the following ratio: one 22\", eight 24"
        "\\textquotedblright, twenty-eight 26\", fifty-six 28\", seventy 30\", fifty-six 32\", twenty"
        "-eight 34\", eight 36\", and one 38\". Starting with A and going through the alphabet to rep"
        "resent different alleles, what are the most probable genotypes of the parents?",
        .choice[0] = "AaBbCcDd\303\227AABBCCDD",
        .choice[1] = "AaBbCcEeFf\303\227AaBbCcEeFf",
        .choice[2] = "AaBbCc\303\227AaBbCc",
        .choice[3] = "AaBb\303\227AaBb",
        .choice[4] = "AaBbCcDd\303\227AaBbCcDd",
        .choice[5] = "AaBbCcDdEe\303\227AaBbCcDdEe",
        .choice[6] = "Aa\303\227Aa",
        .choice[7] = "AaBbDd\303\227AaBbDd",
        .choice[8] = "AaBbCc\303\227AaBb",
        .choice[9] = "AABB\303\227AABB",
        .answer = "E",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "671",
        .domain = "business",
        .title = "Business challenge",
        .question = "Frank Bryant is a salesman for the Fast Cool Refrigerator Co. He is compensated for all trav"
        "el expenses, and is allowed a drawing account of $600 per month. He receives a monthly commi"
        "ssion of 2% of his first $30,000 of sales, 2.5% of the next $10,000 of sales, and 3% of all "
        "sales in excess of $40,000. Find the amount due him for a month in which his travel expenses"
        " paid out of pocket are $332.75, his drawings are $600, and his sales total $41,180.",
        .choice[0] = "$1,332.75",
        .choice[1] = "$942.55",
        .choice[2] = "$783.40",
        .choice[3] = "$618.15",
        .choice[4] = "$725.15",
        .choice[5] = "$1,035.40",
        .choice[6] = "$885.40",
        .choice[7] = "$1,118.15",
        .choice[8] = "$1,218.15",
        .choice[9] = "$332.75",
        .answer = "D",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "3696",
        .domain = "chemistry",
        .title = "Chemistry challenge",
        .question = "2 moles of a monoatomic ideal gas at 45\302\260C are compressed adiabatically and reversibly"
        " from 75 liters to 20 liters. Calculate q, W, \\DeltaH and \\DeltaE for the process, excludi"
        "ng vi-brationalcontributions.",
        .choice[0] = "q=0, \316\224E=15.00kJ, \316\224H=22.50kJ",
        .choice[1] = "q=11.21kJ, \316\224E=0, \316\224H=18.68kJ",
        .choice[2] = "q=0, \316\224E=18.68kJ, \316\224H=11.21kJ",
        .choice[3] = "q=18.68kJ, \316\224E=18.68kJ, \316\224H=18.68kJ",
        .choice[4] = "q=0, \316\224E=11.21kJ, \316\224H=18.68kJ",
        .choice[5] = "q=11.21kJ, \316\224E=18.68kJ, \316\224H=0",
        .choice[6] = "q=22.50kJ, \316\224E=15.00kJ, \316\224H=0",
        .choice[7] = "q=0, \316\224E=22.50kJ, \316\224H=15.00kJ",
        .choice[8] = "q=18.68kJ, \316\224E=11.21kJ, \316\224H=0",
        .choice[9] = "q=18.68kJ, \316\224E=0, \316\224H=11.21kJ",
        .answer = "E",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "10567",
        .domain = "computer science",
        .title = "Computer Science challenge",
        .question = "Let F: K \\times R \\to MF:K\303\227R\342\206\222M be a secure PRF. For m \\in Mm\342\210"
        "\210M define E(k,m) = \\big[ r \\gets R,\\ \\text{output } \\big(r,\\ F(k,r) \\oplus m\\big)"
        "\\ \\big]E(k,m)=[r\342\206\220R, output (r, F(k,r)\342\212\225m) ] Is EE symantically secure"
        " under CPA?",
        .choice[0] = "It depends on the size of R",
        .choice[1] = "Yes, but only if R is large enough so r never repeats (w.h.p)",
        .choice[2] = "Yes, but only if M is small enough so m never repeats (w.h.p)",
        .choice[3] = "No, unless F is a weak PRF",
        .choice[4] = "No, unless F is a strong PRF",
        .choice[5] = "Yes, but only if K is large enough so k never repeats (w.h.p)",
        .choice[6] = "It depends on the size of K",
        .choice[7] = "No, there is always a CPA attack on this system",
        .choice[8] = "It depends on what F is used",
        .choice[9] = "Yes, whenever F is a secure PRF",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "6955",
        .domain = "economics",
        .title = "Economics challenge",
        .question = "Which of the following is included in U.S. GDP? I.   The market value of movies made in Afri"
        "ca by U.S. citizens II.   The market value of olive oil made in Italy and sold in the United"
        " States. III.   The market value of blue jeans made in the United States and sold in Japan I"
        "V.   The market value of wine made in the United States by Canadian citizens",
        .choice[0] = "II and IV only",
        .choice[1] = "I, II and IV only",
        .choice[2] = "I only",
        .choice[3] = "IV only",
        .choice[4] = "I, II and III only",
        .choice[5] = "I and IV only",
        .choice[6] = "III and IV only",
        .choice[7] = "I and III only",
        .choice[8] = "II and III only",
        .choice[9] = "II III and IV only",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "11346",
        .domain = "engineering",
        .title = "Engineering challenge",
        .question = "Calculate the loss of head in a pipe (1/4) in. in diameter and 16 ft. long, when water flows"
        " at half the critical velocity. The critical velocity occurs when Reynolds number is 2500. T"
        "ake viscosity \\mu = 1.01 centipoise = 0.679 \303\227 10^-3 lbm/ft-sec.",
        .choice[0] = "3.12 in",
        .choice[1] = "7.5 in",
        .choice[2] = "0.8 in",
        .choice[3] = "6.3 in",
        .choice[4] = "2.5 in",
        .choice[5] = "5.0 in",
        .choice[6] = "1.75 in",
        .choice[7] = "4.2 in",
        .choice[8] = "4.8 in",
        .choice[9] = "2.0 in",
        .answer = "A",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "6500",
        .domain = "health",
        .title = "Health challenge",
        .question = "A 33-year-old man undergoes a radical thyroidectomy for thyroid cancer. During the operation"
        ", moderate hemorrhaging requires ligation of several vessels in the left side of the neck. P"
        "ostoperatively, serum studies show a calcium concentration of 7.5 mg/dL, albumin concentrati"
        "on of 4 g/dL, and parathyroid hormone concentration of 200 pg/mL. Damage to which of the fol"
        "lowing vessels caused the findings in this patient?",
        .choice[0] = "Branch of the costocervical trunk",
        .choice[1] = "Branch of the external carotid artery",
        .choice[2] = "Branch of the thyrocervical trunk",
        .choice[3] = "Tributary of the internal jugular vein",
        .answer = "C",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "4674",
        .domain = "history",
        .title = "History challenge",
        .question = "Recent discoveries of Homo naledi fossils in the Dinaledi Chamber of the Rising Star cave sy"
        "stem in South Africa suggest a mixture of post-cranial morphologies or mosaic of hominid tra"
        "its. One of the reasons this new species is controversial is that:",
        .choice[0] = "the Homo naledi fossils were found in the same layer as modern human fossils, suggesting the"
        "y lived at the same time.",
        .choice[1] = "the Homo naledi fossils have traits that do not fit into the current understanding of homini"
        "d evolution.",
        .choice[2] = "paleoanthropologists did not make the discovery and it was found in an inaccessible cave in "
        "South Africa.",
        .choice[3] = "the fossils were found in a region previously thought to be inhabited by a different species"
        " of Homo.",
        .choice[4] = "the Homo naledi fossils show traits of both Homo and Australopithecus, suggesting a potentia"
        "l hybrid species.",
        .choice[5] = "the methods used to extract the fossils from the cave were controversial and potentially dam"
        "aging to the specimens.",
        .choice[6] = "there is, as yet, no firm date that can be applied to the specimens.",
        .choice[7] = "the Homo naledi fossils have been shown to pre-date the earliest known hominid fossils by mi"
        "llions of years.",
        .choice[8] = "there are already two well-known species of Homo and it is unlikely there were more than two"
        " species at one time.",
        .choice[9] = "the researchers who discovered the fossils have been accused of falsifying their findings.",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "1467",
        .domain = "law",
        .title = "Law challenge",
        .question = "A motorcycle collector and his friend had known each other for years. The collector sent a l"
        "etter to the friend that said, \"I am willing to sell you my oldest motorcycle for $4,000.\""
        " The friend sent the collector an email saying, \"I accept your offer.\" The collector owned"
        " three motorcycles: a 1985 model, a 1987 model, and a 1992 model. The friend did not know ab"
        "out the 1985 model, which was the one the collector intended to sell. The collector was unaw"
        "are that the friend intended to buy the 1987 model. When the collector tendered the 1985 mod"
        "el, the friend refused to accept it. Is the friend likely to prevail in a breach of contract"
        " suit against the collector?",
        .choice[0] = "No, because it was the friend's responsibility to know the details about the collector's mot"
        "orcycles before agreeing to purchase.",
        .choice[1] = "No, because by not inquiring further about the meaning of the phrase \"oldest motorcycle,\" "
        "the friend assumed the risk of being mistaken.",
        .choice[2] = "No, the friend won't prevail as he misunderstood the term \"oldest motorcycle\".",
        .choice[3] = "Yes, so long as a reasonable person in the friend's position would have considered the lette"
        "r as referring to the 1987 model.",
        .choice[4] = "No, because neither the friend nor the collector had reason to know of the other's interpret"
        "ation of the words \"oldest motorcycle.\"",
        .choice[5] = "Yes, the friend will prevail as the collector falsely advertised the motorcycle model.",
        .choice[6] = "No, because the friend should have asked for clarification about which model was being sold "
        "before agreeing to the purchase.",
        .choice[7] = "Yes, because the friend could have reasonably assumed that the \"oldest motorcycle\" referre"
        "d to the 1987 model, given his knowledge.",
        .choice[8] = "Yes, because the friend had no reason to know of the collector's meaning and the collector c"
        "ould have avoided the confusion by more careful drafting.",
        .choice[9] = "Yes, because the collector should have clarified which model he was referring to in the lett"
        "er.",
        .answer = "E",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "8771",
        .domain = "math",
        .title = "Math challenge",
        .question = "7.4-11. Some dentists were interested in studying the fusion of embryonic rat palates by a s"
        "tandard transplantation technique. When no treatment is used, the probability of fusion equa"
        "ls approximately 0.89 . The dentists would like to estimate $p$, the probability of fusion, "
        "when vitamin A is lacking.\n(a) How large a sample $n$ of rat embryos is needed for $y / n "
        "\\pm 0.10$ to be a $95 \\%$ confidence interval for $p$ ?",
        .choice[0] = "70",
        .choice[1] = "50",
        .choice[2] = "45",
        .choice[3] = "40",
        .choice[4] = "30",
        .choice[5] = "55",
        .choice[6] = "$38$",
        .choice[7] = "65",
        .choice[8] = "60",
        .choice[9] = "25",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "5848",
        .domain = "other",
        .title = "Other challenge",
        .question = "Lake Trust a simple trust reported the following items of income and expense during the year"
        ": Dividend income $2500 Taxable interest income 2000 Capital gains (allocable to corpus) 500"
        "0 Accounting fees (allocable to income) (500) Trustee fees (allocable to income) (750) What "
        "is Lake's distributable net income?",
        .choice[0] = "$5,000",
        .choice[1] = "$6,000",
        .choice[2] = "$9,500",
        .choice[3] = "$2,000",
        .choice[4] = "$4,500",
        .choice[5] = "$10,000",
        .choice[6] = "$8,250",
        .choice[7] = "$7,500",
        .choice[8] = "$1,750",
        .choice[9] = "3250",
        .answer = "J",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "11120",
        .domain = "philosophy",
        .title = "Philosophy challenge",
        .question = " Use indirect truth tables to determine whether each set of propositions is consistent. If t"
        "he set is consistent, choose an option with a consistent valuation. (There may be other cons"
        "istent valuations.)\nD \342\211\241 E\nF \342\210\250 D\n~E \302\267 ~F",
        .choice[0] = "Consistent. Consistent valuation when D and E are true and F is true",
        .choice[1] = "Inconsistent",
        .choice[2] = "Consistent. Consistent valuation when D is true, E and F are false",
        .choice[3] = "Consistent. Consistent valuation when E is true, D and F are false",
        .choice[4] = "Consistent. Consistent valuation when D and E are true and F is false",
        .choice[5] = "Consistent. Consistent valuation when D, E are false and F is true",
        .choice[6] = "Consistent. Consistent valuation when D and F are true and E is false",
        .choice[7] = "Consistent. Consistent valuation when D, E, and F are true",
        .choice[8] = "Consistent. Consistent valuation when F is true, D and E are false",
        .choice[9] = "Consistent. Consistent valuation when D, E and F are false",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "9487",
        .domain = "physics",
        .title = "Physics challenge",
        .question = "A gas sample is known to be a mixture of ethane and butane. A bulb having a $230.0 \\mathrm{"
        "~cm}^3$ capacity is filled with the gas to a pressure of $97.5 \\times 10^3 \\mathrm{~Pa}$ a"
        "t $23.1^{\\circ} \\mathrm{C}$. If the mass of the gas in the bulb is $0.3554 \\mathrm{~g}$, "
        "what is the mole percent of butane in the mixture?\n",
        .choice[0] = "70 %",
        .choice[1] = "27 %",
        .choice[2] = "50 %",
        .choice[3] = "10 %",
        .choice[4] = "60 %",
        .choice[5] = "15 %",
        .choice[6] = "25 %",
        .choice[7] = "32 %",
        .choice[8] = "45 %",
        .choice[9] = "38 %",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "2398",
        .domain = "psychology",
        .title = "Psychology challenge",
        .question = "Susan is a first-grade student who is referred to the school psychologist because she will n"
        "ot remain seated at her desk. Her teacher reports that Susan is \"always getting up and walk"
        "ing around the room.\" What observation data is LEAST relevant to the referral concern?",
        .choice[0] = "Susan's academic performance",
        .choice[1] = "The teacher's teaching style",
        .choice[2] = "Number of students in the class",
        .choice[3] = "The school's disciplinary policies",
        .choice[4] = "Classroom environment",
        .choice[5] = "Peer comparison",
        .choice[6] = "Frequency",
        .choice[7] = "Intensity",
        .choice[8] = "Susan's diet",
        .choice[9] = "Duration",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "8863",
        .domain = "math",
        .title = "Math challenge",
        .question = "An investor deposits $1000 in an account paying interest at a rate of 8% compounded monthly,"
        " and also makes additional deposits of \\$25 per month. Find the balance in the account afte"
        "r 3 years.",
        .choice[0] = "$3200.67",
        .choice[1] = "$2050.89",
        .choice[2] = "$2100.45",
        .choice[3] = "$1950.32",
        .choice[4] = "$3000.59",
        .choice[5] = "$2500.00",
        .choice[6] = "2283.63 $",
        .choice[7] = "$2750.48",
        .choice[8] = "$2600.25",
        .choice[9] = "$1800.75",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "OlympiadBench",
        .id = "1606",
        .domain = "Combinatorics",
        .title = "Olympiad problem 1606",
        .question = "Xenia and Sergey play the following game. Xenia thinks of a positive integer $N$ not exceedi"
        "ng 5000. Then she fixes 20 distinct positive integers $a_{1}, a_{2}, \\ldots, a_{20}$ such t"
        "hat, for each $k=1,2, \\ldots, 20$, the numbers $N$ and $a_{k}$ are congruent modulo $k$. By"
        " a move, Sergey tells Xenia a set $S$ of positive integers not exceeding 20 , and she tells "
        "him back the set $\\left\\{a_{k}: k \\in S\\right\\}$ without spelling out which number corr"
        "esponds to which index. How many moves does Sergey need to determine for sure the number Xen"
        "ia thought of?",
        .answer = "2",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1613",
        .domain = "Algebra",
        .title = "Olympiad problem 1613",
        .question = "Determine all positive integers $n$ satisfying the following condition: for every monic poly"
        "nomial $P$ of degree at most $n$ with integer coefficients, there exists a positive integer "
        "$k \\leq n$, and $k+1$ distinct integers $x_{1}, x_{2}, \\ldots, x_{k+1}$ such that\n\n\n\n$"
        "$\n\nP\\left(x_{1}\\right)+P\\left(x_{2}\\right)+\\cdots+P\\left(x_{k}\\right)=P\\left(x_{k+"
        "1}\\right) .\n\n$$\n\n\nNote. A polynomial is monic if the coefficient of the highest power "
        "is one.",
        .answer = "2",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1614",
        .domain = "Combinatorics",
        .title = "Olympiad problem 1614",
        .question = "Let $n$ be an integer greater than 1 and let $X$ be an $n$-element set. A non-empty collecti"
        "on of subsets $A_{1}, \\ldots, A_{k}$ of $X$ is tight if the union $A_{1} \\cup \\cdots \\cu"
        "p A_{k}$ is a proper subset of $X$ and no element of $X$ lies in exactly one of the $A_{i}$ "
        "s. Find the largest cardinality of a collection of proper non-empty subsets of $X$, no non-e"
        "mpty subcollection of which is tight.\n\n\n\nNote. A subset $A$ of $X$ is proper if $A \\neq"
        " X$. The sets in a collection are assumed to be distinct. The whole collection is assumed to"
        " be a subcollection.",
        .answer = "2n-2",
        .alias[0] = "2(n-1)",
        .answer_kind = EVAL_ANSWER_EXACT_TEXT,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1618",
        .domain = "Number Theory",
        .title = "Olympiad problem 1618",
        .question = "Determine all prime numbers $p$ and all positive integers $x$ and $y$ satisfying $x^{3}+y^{3"
        "}=$ $p(x y+p)$. Report each solution once as $(x,y,p)$ with $x \\leq y$.",
        .answer = "(1,8,19),(2,7,13),(4,5,7)",
        .alias[0] = "(4,5,7),(2,7,13),(1,8,19)",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1697",
        .domain = "Combinatorics",
        .title = "Olympiad problem 1697",
        .question = "On a square table of 2011 by 2011 cells we place a finite number of napkins that each cover "
        "a square of 52 by 52 cells. In each cell we write the number of napkins covering it, and we "
        "record the maximal number $k$ of cells that all contain the same nonzero number. Considering"
        " all possible napkin configurations, what is the largest value of $k$ ?",
        .answer = "3986729",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1709",
        .domain = "Number Theory",
        .title = "Olympiad problem 1709",
        .question = "For each positive integer $k$, let $t(k)$ be the largest odd divisor of $k$. Determine all p"
        "ositive integers $a$ for which there exists a positive integer $n$ such that all the differe"
        "nces\n\n$$\nt(n+a)-t(n), \\quad t(n+a+1)-t(n+1), \\quad \\ldots, \\quad t(n+2 a-1)-t(n+a-1)"
        "\n$$\n\nare divisible by 4 .",
        .answer = "1,3,5",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1716",
        .domain = "Algebra",
        .title = "Olympiad problem 1716",
        .question = "Let $x_{1}, \\ldots, x_{100}$ be nonnegative real numbers such that $x_{i}+x_{i+1}+x_{i+2} "
        "\\leq 1$ for all $i=1, \\ldots, 100$ (we put $x_{101}=x_{1}, x_{102}=x_{2}$ ). Find the maxi"
        "mal possible value of the sum\n\n$$\nS=\\sum_{i=1}^{100} x_{i} x_{i+2}\n$$",
        .answer = "25/2",
        .alias[0] = "12.5",
        .answer_kind = EVAL_ANSWER_RATIONAL,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1736",
        .domain = "Number Theory",
        .title = "Olympiad problem 1736",
        .question = "Find the least positive integer $n$ for which there exists a set $\\left\\{s_{1}, s_{2}, \\l"
        "dots, s_{n}\\right\\}$ consisting of $n$ distinct positive integers such that\n\n$$\n\\left("
        "1-\\frac{1}{s_{1}}\\right)\\left(1-\\frac{1}{s_{2}}\\right) \\ldots\\left(1-\\frac{1}{s_{n}}"
        "\\right)=\\frac{51}{2010}\n$$",
        .answer = "39",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1750",
        .domain = "Combinatorics",
        .title = "Olympiad problem 1750",
        .question = "A diagonal of a regular 2006-gon is called odd if its endpoints divide the boundary into two"
        " parts, each composed of an odd number of sides. Sides are also regarded as odd diagonals.\n"
        "\nSuppose the 2006-gon has been dissected into triangles by 2003 nonintersecting diagonals. "
        "Find the maximum possible number of isosceles triangles with two odd sides.",
        .answer = "1003",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1766",
        .domain = "Number Theory",
        .title = "Olympiad problem 1766",
        .question = "Determine all pairs $(x, y)$ of integers satisfying the equation\n\n$$\n1+2^{x}+2^{2 x+1}=y^"
        "{2}\n$$",
        .answer = "(0,2),(0,-2),(4,23),(4,-23)",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "LiveBench",
        .id = "c833574b0ff6e8609bee19b36fabc48501cdcd9a64b99cd0422f54ada1acc53d",
        .domain = "zebra_puzzle",
        .title = "Zebra logic puzzle",
        .question = "There are 4 people standing in a line. From left to right, they are numbered 1 to 4.\nEach p"
        "erson has a set of attributes: Beverage, Job, Movie-Genre, Transport.\nThe attributes have t"
        "he following possible values:\nBeverage: lemonade, fanta, iced-tea, hot-chocolate\nJob: soci"
        "al-worker, software-developer, coach, accountant\nMovie-Genre: martial-arts, horror, documen"
        "tary, satire\nTransport: skateboard, train, ship, helicopter\nEach person has a unique value"
        " for each attribute.\nYou know the following about the people:\nThe person who watches docum"
        "entary is somewhere between the person who watches martial-arts and the person who is a acco"
        "untant\nThe person who travels by skateboard is not anywhere to the left of the person who t"
        "ravels by train\nThe person who watches martial-arts and the person who drinks iced-tea have"
        " the same parity positions\nThe person who watches satire is not anywhere to the right of th"
        "e person who watches horror\nThe person who is a accountant and the person who drinks hot-ch"
        "ocolate have the same parity positions\nThe person who travels by train and the person who d"
        "rinks hot-chocolate have different parity positions\nEither the person who is a coach is the"
        " same as the person who travels by helicopter or the person who is a coach is the same as th"
        "e person who watches documentary, but not both\nThe person who watches satire and the person"
        " who is a accountant have different parity positions\nThe person who watches satire is not a"
        "nywhere to the right of the person who watches documentary\nThe person who travels by skateb"
        "oard is the same as the person who watches horror or the person who drinks fanta is the same"
        " as the person who travels by skateboard or both\nThe person who drinks fanta is somewhere b"
        "etween the person who watches satire and the person who watches martial-arts\nThe person who"
        " is a social-worker and the person who travels by helicopter have the same parity positions"
        "\nThe person who is a accountant is not anywhere to the left of the person who travels by sh"
        "ip\nThe person who drinks iced-tea is not anywhere to the right of the person who drinks fan"
        "ta\nIn the above, 'parity position' means the evenness or oddness of the person's position i"
        "n the line.\nGiven this information, answer the following questions:\nWhat is the movie genr"
        "e of the person who is a software-developer?\nWhat movie genre does the person who drinks ho"
        "t-chocolate watch?\nWhat movie genre does the person in position 2 watch?\nAt what position "
        "is the person who watches horror?",
        .answer = "satire, martial-arts, horror, 2",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 8192,
    },
    {
        .source = "LiveBench",
        .id = "b7b7e331df4614775210ef5e524ed7f20c5fec6655fb0dd179c94e4d71170862",
        .domain = "zebra_puzzle",
        .title = "Zebra logic puzzle",
        .question = "There are 3 people standing in a line. From left to right, they are numbered 1 to 3.\nEach p"
        "erson has a set of attributes: Beverage, Food, Movie-Genre, Nationality.\nThe attributes hav"
        "e the following possible values:\nBeverage: mirinda, cola, water\nFood: cauliflower, radish,"
        " zucchini\nMovie-Genre: family, spy, crime\nNationality: brazilian, german, canadian\nEach p"
        "erson has a unique value for each attribute.\nYou know the following about the people:\nThe "
        "person who is brazilian is somewhere between the person who is german and the person who dri"
        "nks cola\nThe person who eats cauliflower is not anywhere to the right of the person who dri"
        "nks cola\nThe person who drinks mirinda is somewhere between the person who drinks cola and "
        "the person who is german\nThe person who watches family and the person who eats radish have "
        "the same parity positions\nThe person who watches spy is not anywhere to the right of the pe"
        "rson who watches family\nEither the person who is canadian is the same as the person who wat"
        "ches spy or the person who eats cauliflower is the same as the person who is canadian, but n"
        "ot both\nThe person who eats cauliflower is not anywhere to the right of the person who eats"
        " radish\nIn the above, 'parity position' means the evenness or oddness of the person's posit"
        "ion in the line.\nGiven this information, answer the following questions:\nWhat movie genre "
        "does the person in position 1 watch?\nAt what position is the person who watches crime?\nAt "
        "what position is the person who eats zucchini?\nWhat movie genre does the person in position"
        " 3 watch?",
        .answer = "crime, 1, 2, family",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 8192,
    },
    {
        .source = "LiveBench",
        .id = "0daa7ca38beec4441b9d5c04d0b98912322926f0a3ac28a5097889d4ed83506f",
        .domain = "web_of_lies_v2",
        .title = "Web of lies",
        .question = "In this question, assume each person either always tells the truth or always lies. Tala is a"
        "t the movie theater. The person at the restaurant says the person at the aquarium lies. Ayaa"
        "n is at the aquarium. Ryan is at the botanical garden. The person at the park says the perso"
        "n at the art gallery lies. The person at the museum tells the truth. Zara is at the museum. "
        "Jake is at the art gallery. The person at the art gallery says the person at the theater lie"
        "s. Beatriz is at the park. The person at the movie theater says the person at the train stat"
        "ion lies. Nadia is at the campground. The person at the campground says the person at the ar"
        "t gallery tells the truth. The person at the theater lies. The person at the amusement park "
        "says the person at the aquarium tells the truth. Grace is at the restaurant. The person at t"
        "he aquarium thinks their friend is lying. Nia is at the theater. Kehinde is at the train sta"
        "tion. The person at the theater thinks their friend is lying. The person at the botanical ga"
        "rden says the person at the train station tells the truth. The person at the aquarium says t"
        "he person at the campground tells the truth. The person at the aquarium saw a firetruck. The"
        " person at the train station says the person at the amusement park lies. Mateo is at the amu"
        "sement park. Does the person at the train station tell the truth? Does the person at the amu"
        "sement park tell the truth? Does the person at the aquarium tell the truth?",
        .answer = "no, yes, yes",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 8192,
    },
    {
        .source = "LiveBench",
        .id = "9ee37b9a04ab050936c86b2c5bb7abbaa0bc0e737d59a7bff9ba11e9b4069c1d",
        .domain = "web_of_lies_v2",
        .title = "Web of lies",
        .question = "In this question, assume each person either always tells the truth or always lies. Liam is a"
        "t the movie theater. The person at the beach says the person at the restaurant tells the tru"
        "th. Kehinde is at the aquarium. The person at the amusement park saw a firetruck. The person"
        " at the aquarium tells the truth. Luna is at the library. Jaxon is at the amusement park. Th"
        "e person at the amusement park says the person at the beach tells the truth. Anika is at the"
        " restaurant. The person at the barbershop tells the truth. The person at the observatory say"
        "s the person at the planetarium lies. Hiroshi is at the beach. Isabella is at the planetariu"
        "m. Nia is at the barbershop. The person at the movie theater says the person at the observat"
        "ory lies. The person at the restaurant tells the truth. Max is at the observatory. The perso"
        "n at the library says the person at the amusement park tells the truth. The person at the pl"
        "anetarium says the person at the library tells the truth. Does the person at the movie theat"
        "er tell the truth? Does the person at the observatory tell the truth? Does the person at the"
        " planetarium tell the truth?",
        .answer = "yes, no, yes",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 8192,
    },
    {
        .source = "LiveBench",
        .id = "d7071c9ff5d9779e7ab955366d0ae8db40f785aadfe7ff0b5a7ede98c05c44ea",
        .domain = "web_of_lies_v2",
        .title = "Web of lies",
        .question = "In this question, assume each person either always tells the truth or always lies. Liam is a"
        "t the restaurant. The person at the observatory says the person at the botanical garden lies"
        ". Elowen is at the vineyard. The person at the library says the person at the botanical gard"
        "en tells the truth. The person at the train station tells the truth. The person at the botan"
        "ical garden saw a firetruck. The person at the botanical garden says the person at the train"
        " station lies. Jake is at the aquarium. Soren is at the farm. Theo is at the gym. The person"
        " at the train station saw a firetruck. Devika is at the train station. Kehinde is at the lib"
        "rary. The person at the restaurant lies. The person at the farm says the person at the train"
        " station lies. The person at the gym says the person at the train station lies. Hiroshi is a"
        "t the botanical garden. The person at the vineyard lies. The person at the aquarium says the"
        " person at the library tells the truth. Quan is at the observatory. Does the person at the l"
        "ibrary tell the truth? Does the person at the botanical garden tell the truth? Does the pers"
        "on at the train station tell the truth?",
        .answer = "no, no, yes",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 8192,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe190-int-multiply-01",
        .domain = "CWE-190",
        .title = "Integer multiplication overflow",
        .question = "The following C function is intended to allocate space for count integers. Identify the line"
        " containing the primary weakness.\n\n 1  int *make_array(size_t count) {\n 2      int *resul"
        "t = NULL;\n 3      if (count == 0)\n 4          return NULL;\n 5      size_t bytes;\n 6     "
        " if (count > SIZE_MAX / sizeof(int))\n 7          return NULL;\n 8      bytes = count * size"
        "of(int);\n 9      result = malloc(bytes);\n10      if (result == NULL)\n11          return N"
        "ULL;\n12      memset(result, 0, count * count * sizeof(int));\n13      return result;\n14  }",
        .answer = "12",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe121-array-index-01",
        .domain = "CWE-121",
        .title = "Stack array index validation",
        .question = "The following C function should update one element of a fixed-size stack array. Identify the"
        " smallest exact set of line numbers containing the primary weakness.\n\n 1  int update_value"
        "(int index, int value) {\n 2      int data[10] = {0};\n 3      if (index < 10)\n 4          "
        "data[index] = value;\n 5      return data[0];\n 6  }",
        .answer = "3-4",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe401-realloc-01",
        .domain = "CWE-401",
        .title = "Lost allocation after realloc",
        .question = "The following C function grows an allocated buffer. Identify the line containing the primary"
        " weakness.\n\n 1  char *grow(char *buf, size_t size) {\n 2      if (size > SIZE_MAX / 2)\n 3"
        "          return NULL;\n 4      size *= 2;\n 5      buf = realloc(buf, size);\n 6      if (b"
        "uf == NULL)\n 7          return NULL;\n 8      return buf;\n 9  }",
        .answer = "5",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe476-null-after-deref-01",
        .domain = "CWE-476",
        .title = "Null check after dereference",
        .question = "The following C function reads the first byte of a caller-supplied buffer. Identify the line"
        " containing the primary weakness.\n\n 1  int first_byte(const unsigned char *p) {\n 2      i"
        "nt value = p[0];\n 3      if (p == NULL)\n 4          return -1;\n 5      return value;\n 6 "
        " }",
        .answer = "2",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe369-int-divide-01",
        .domain = "CWE-369",
        .title = "Unchecked integer division",
        .question = "The following C function computes a ratio from caller-supplied integers. Identify the line c"
        "ontaining the primary weakness.\n\n 1  int ratio(int total, int parts) {\n 2      if (total "
        "< 0)\n 3          return -1;\n 4      return total / parts;\n 5  }",
        .answer = "4",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
};

const size_t eval_hard_case_count =
    sizeof(eval_hard_cases) / sizeof(eval_hard_cases[0]);
