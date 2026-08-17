#!/usr/bin/env python3
"""Build a behavior-grounded context-memory conflict dataset.

Research question
-----------------
When a language model follows a false contextual claim because an instruction
asks it to use the paragraph, does it still internally represent that the claim
conflicts with its parametric knowledge?

The generator deliberately does *not* decide whether a model knows a fact.  It
produces two stages of data:

1. ``screening.jsonl`` contains context-free prompts used to establish that the
   evaluated model reliably knows each fact and prefers the true answer over
   every false answer assigned to that fact.
2. ``experiment.jsonl`` crosses claim truth, claim relevance, and answer-source
   policy.  Behavioral evaluation later records whether the model chooses the
   contextual or parametric answer.

Important design choices
------------------------
* False answers are derangements of the true-answer list.  Therefore every
  answer string appears equally often as a true and false answer, preventing a
  probe from learning a separate "wrong answer" vocabulary.
* Distractor facts are also assigned by derangement, so every fact is used
  equally often as an irrelevant context.
* Counterbalancing happens inside each fact split and CV fold.  Held-out facts
  therefore cannot leak into training as distractors or false-answer sources.
* Neutral, context-grounded, and parametric-grounded prompts are exact content
  pairs.  Only the answer-source instruction changes.
* Three prompt bundles vary claim syntax, filler, question wording, policy
  wording, and the final one-word response constraint.  One bundle is marked
  as held out for paraphrase generalization.
* Character spans and semantic endpoints are written during prompt assembly.
  Downstream code must never reconstruct them with fragile string searches.
* Facts with disputed, changing, multiple, or multi-word canonical answers are
  mostly excluded.  Model-specific knowledge is still an empirical property
  and must be established from the screening records.

Outputs
-------
``facts.jsonl``
    One record per candidate fact, including deterministic fact split,
    cross-validation fold, aliases, counterbalanced false answers, and
    distractor assignments.
``screening.jsonl``
    Context-free prompts for model-specific parametric-knowledge screening.
``experiment.jsonl``
    Fully crossed experimental prompts and explicit span metadata.
``manifest.json``
    Schema, templates, labels, configuration, counts, and file hashes.
``DATASET_CARD.md``
    Short human-readable usage and leakage-prevention notes.

Example
-------
python build_conflict_awareness_dataset.py \
    --output-dir conflict_awareness_dataset_v1 \
    --seed 20260816 \
    --counterbalance-rounds 1

The script uses only the Python standard library and validates every generated
record before writing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.1.0"
DEFAULT_SEED = 20260816
FACT_AUDIT_DATE = "2026-08-16"


RELATION_CURATION: dict[str, dict[str, Any]] = {
    "country_capital": {
        "audit_date": FACT_AUDIT_DATE,
        "reference_urls": [
            "https://unstats.un.org/unsd/geoinfo/ungegn/docs/geoname.pdf",
        ],
        "notes": (
            "Manually audited after applying the one-orthographic-word rule. "
            "Cases with disputed, changing, multiple, de facto-only, or technically "
            "ambiguous capitals were excluded; model knowledge is screened separately."
        ),
        "illustrative_exclusions": [
            "Equatorial Guinea (capital changed in 2026 and new answer is multi-word)",
            "Indonesia (capital transition)",
            "Switzerland (Bern is formally the federal city, not a constitutional capital)",
            "South Africa and Eswatini (multiple capitals)",
            "Bolivia, Benin, and Malaysia (capital/seat distinctions that can invite ambiguity)",
            "The Netherlands was retained because Amsterdam is constitutionally the capital, "
            "despite the seat of government being in The Hague",
        ],
    },
    "element_symbol": {
        "audit_date": FACT_AUDIT_DATE,
        "reference_urls": [
            "https://iupac.org/what-we-do/periodic-table-of-elements/",
        ],
        "notes": "All 118 IUPAC element names and symbols, in atomic-number order.",
        "illustrative_exclusions": [],
    },
}


@dataclass(frozen=True)
class Fact:
    relation_id: str
    subject: str
    answer: str
    answer_aliases: tuple[str, ...] = ()

    @property
    def fact_id(self) -> str:
        return f"{self.relation_id}:{slugify(self.subject)}"


@dataclass(frozen=True)
class RelationSpec:
    relation_id: str
    display_name: str
    transfer_role: str
    facts: tuple[Fact, ...]
    claim_templates: tuple[str, ...]
    fillers: tuple[str, ...]
    question_templates: tuple[str, ...]
    answer_normalization: str


@dataclass(frozen=True)
class TemplateBundle:
    bundle_id: str
    analysis_role: str
    claim_template_index: int
    filler_index: int
    question_template_index: int
    response_constraint_index: int
    policy_template_index: int
    policy_placement: str


CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "condition_id": "true_relevant",
        "claim_is_world_true": True,
        "claim_is_query_relevant": True,
        "query_conflict_label": False,
    },
    {
        "condition_id": "false_relevant",
        "claim_is_world_true": False,
        "claim_is_query_relevant": True,
        "query_conflict_label": True,
    },
    {
        "condition_id": "true_irrelevant",
        "claim_is_world_true": True,
        "claim_is_query_relevant": False,
        "query_conflict_label": False,
    },
    {
        "condition_id": "false_irrelevant",
        "claim_is_world_true": False,
        "claim_is_query_relevant": False,
        "query_conflict_label": False,
    },
)


POLICY_TEMPLATES: dict[str, tuple[str, ...]] = {
    # The absence of a source instruction is intentional: this recreates the
    # user's observed neutral condition instead of inserting a synthetic cue.
    "neutral": ("", "", ""),
    "context": (
        "Based on this paragraph, answer the question below.",
        "Answer according to the paragraph.",
        "Use the paragraph to answer the question.",
    ),
    "parametric": (
        "Ignoring this paragraph, answer the question below from your own knowledge.",
        "Do not use the paragraph; rely on your own knowledge.",
        "Set the paragraph aside and answer from what you already know.",
    ),
}


POLICY_METADATA: dict[str, dict[str, Any]] = {
    "neutral": {
        "policy_target": "unspecified",
        "analysis_role": "primary",
        "description": "No instruction chooses between context and parametric knowledge.",
    },
    "context": {
        "policy_target": "contextual",
        "analysis_role": "primary",
        "description": "The paragraph is presented as the basis for answering, without mentioning conflict.",
    },
    "parametric": {
        "policy_target": "parametric",
        "analysis_role": "anchor_control",
        "description": "Explicit positive-control instruction to ignore the paragraph.",
    },
}


RESPONSE_CONSTRAINTS: tuple[str, ...] = (
    "Answer in one word.",
    "Respond with one word only.",
    "Output a single word and nothing else.",
)


TEMPLATE_BUNDLES: tuple[TemplateBundle, ...] = (
    TemplateBundle(
        bundle_id="development",
        analysis_role="development",
        claim_template_index=0,
        filler_index=0,
        question_template_index=0,
        response_constraint_index=0,
        policy_template_index=0,
        policy_placement="instruction_line",
    ),
    TemplateBundle(
        bundle_id="validation",
        analysis_role="validation",
        claim_template_index=1,
        filler_index=1,
        question_template_index=1,
        response_constraint_index=1,
        policy_template_index=1,
        policy_placement="instruction_line",
    ),
    TemplateBundle(
        bundle_id="heldout_paraphrase",
        analysis_role="heldout_paraphrase",
        claim_template_index=2,
        filler_index=2,
        question_template_index=2,
        response_constraint_index=2,
        policy_template_index=2,
        policy_placement="instruction_line",
    ),
)


CAPITAL_ROWS: tuple[tuple[str, str], ...] = (
    # Europe and nearby transcontinental states
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Portugal", "Lisbon"),
    ("United Kingdom", "London"),
    ("Ireland", "Dublin"),
    ("Belgium", "Brussels"),
    ("Netherlands", "Amsterdam"),
    ("Austria", "Vienna"),
    ("Denmark", "Copenhagen"),
    ("Norway", "Oslo"),
    ("Sweden", "Stockholm"),
    ("Finland", "Helsinki"),
    ("Iceland", "Reykjavik"),
    ("Poland", "Warsaw"),
    ("Czechia", "Prague"),
    ("Slovakia", "Bratislava"),
    ("Hungary", "Budapest"),
    ("Romania", "Bucharest"),
    ("Bulgaria", "Sofia"),
    ("Greece", "Athens"),
    ("Croatia", "Zagreb"),
    ("Serbia", "Belgrade"),
    ("Slovenia", "Ljubljana"),
    ("Albania", "Tirana"),
    ("Ukraine", "Kyiv"),
    ("Belarus", "Minsk"),
    ("Lithuania", "Vilnius"),
    ("Latvia", "Riga"),
    ("Estonia", "Tallinn"),
    ("Russia", "Moscow"),
    ("Turkey", "Ankara"),
    ("Cyprus", "Nicosia"),
    ("Malta", "Valletta"),
    ("Moldova", "Chisinau"),
    ("North Macedonia", "Skopje"),
    ("Montenegro", "Podgorica"),
    ("Bosnia and Herzegovina", "Sarajevo"),
    ("Georgia", "Tbilisi"),
    ("Armenia", "Yerevan"),
    ("Azerbaijan", "Baku"),
    ("Liechtenstein", "Vaduz"),
    # Americas
    ("United States", "Washington"),
    ("Canada", "Ottawa"),
    ("Cuba", "Havana"),
    ("Jamaica", "Kingston"),
    ("Haiti", "Port-au-Prince"),
    ("Nicaragua", "Managua"),
    ("Honduras", "Tegucigalpa"),
    ("Colombia", "Bogota"),
    ("Venezuela", "Caracas"),
    ("Ecuador", "Quito"),
    ("Peru", "Lima"),
    ("Brazil", "Brasilia"),
    ("Chile", "Santiago"),
    ("Uruguay", "Montevideo"),
    ("Paraguay", "Asuncion"),
    ("Guyana", "Georgetown"),
    ("Suriname", "Paramaribo"),
    ("Bahamas", "Nassau"),
    ("Barbados", "Bridgetown"),
    ("Dominica", "Roseau"),
    ("Saint Lucia", "Castries"),
    ("Saint Vincent and the Grenadines", "Kingstown"),
    ("Belize", "Belmopan"),
    ("Saint Kitts and Nevis", "Basseterre"),
    # Asia
    ("China", "Beijing"),
    ("Japan", "Tokyo"),
    ("South Korea", "Seoul"),
    ("North Korea", "Pyongyang"),
    ("Mongolia", "Ulaanbaatar"),
    ("Vietnam", "Hanoi"),
    ("Thailand", "Bangkok"),
    ("Laos", "Vientiane"),
    ("Myanmar", "Naypyidaw"),
    ("Philippines", "Manila"),
    ("Pakistan", "Islamabad"),
    ("Bangladesh", "Dhaka"),
    ("Nepal", "Kathmandu"),
    ("Afghanistan", "Kabul"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    ("Syria", "Damascus"),
    ("Lebanon", "Beirut"),
    ("Jordan", "Amman"),
    ("Saudi Arabia", "Riyadh"),
    ("Oman", "Muscat"),
    ("Qatar", "Doha"),
    ("Bahrain", "Manama"),
    ("Kazakhstan", "Astana"),
    ("Uzbekistan", "Tashkent"),
    ("Turkmenistan", "Ashgabat"),
    ("Kyrgyzstan", "Bishkek"),
    ("Tajikistan", "Dushanbe"),
    ("Bhutan", "Thimphu"),
    ("Maldives", "Male"),
    ("Timor-Leste", "Dili"),
    # Africa
    ("Egypt", "Cairo"),
    ("Libya", "Tripoli"),
    ("Tunisia", "Tunis"),
    ("Algeria", "Algiers"),
    ("Morocco", "Rabat"),
    ("Kenya", "Nairobi"),
    ("Uganda", "Kampala"),
    ("Tanzania", "Dodoma"),
    ("Nigeria", "Abuja"),
    ("Ghana", "Accra"),
    ("Senegal", "Dakar"),
    ("Botswana", "Gaborone"),
    ("Zimbabwe", "Harare"),
    ("Zambia", "Lusaka"),
    ("Malawi", "Lilongwe"),
    ("Mozambique", "Maputo"),
    ("Angola", "Luanda"),
    ("Namibia", "Windhoek"),
    ("Rwanda", "Kigali"),
    ("Burundi", "Gitega"),
    ("Somalia", "Mogadishu"),
    ("Eritrea", "Asmara"),
    ("Cameroon", "Yaounde"),
    ("Gabon", "Libreville"),
    ("Republic of the Congo", "Brazzaville"),
    ("Democratic Republic of the Congo", "Kinshasa"),
    ("Central African Republic", "Bangui"),
    ("Chad", "N'Djamena"),
    ("Niger", "Niamey"),
    ("Mali", "Bamako"),
    ("Burkina Faso", "Ouagadougou"),
    ("Guinea", "Conakry"),
    ("Sierra Leone", "Freetown"),
    ("Liberia", "Monrovia"),
    ("Ivory Coast", "Yamoussoukro"),
    ("Togo", "Lome"),
    ("Mauritania", "Nouakchott"),
    ("Gambia", "Banjul"),
    ("Guinea-Bissau", "Bissau"),
    ("Cabo Verde", "Praia"),
    ("Madagascar", "Antananarivo"),
    ("Seychelles", "Victoria"),
    ("Comoros", "Moroni"),
    ("Lesotho", "Maseru"),
    ("South Sudan", "Juba"),
    # Oceania
    ("Australia", "Canberra"),
    ("New Zealand", "Wellington"),
    ("Fiji", "Suva"),
    ("Samoa", "Apia"),
    ("Tonga", "Nuku'alofa"),
    ("Solomon Islands", "Honiara"),
    ("Marshall Islands", "Majuro"),
    ("Federated States of Micronesia", "Palikir"),
    ("Palau", "Ngerulmud"),
    ("Tuvalu", "Funafuti"),
)


CAPITAL_ALIASES: dict[str, tuple[str, ...]] = {
    "Washington": ("Washington, D.C.", "Washington DC"),
    "Reykjavik": ("Reykjavík",),
    "Kyiv": ("Kiev",),
    "Chisinau": ("Chișinău", "Chişinău"),
    "Bogota": ("Bogotá",),
    "Brasilia": ("Brasília",),
    "Asuncion": ("Asunción",),
    "Male": ("Malé",),
    "Yaounde": ("Yaoundé",),
    "Lome": ("Lomé",),
    "N'Djamena": ("N’Djamena",),
    "Nuku'alofa": ("Nukuʻalofa",),
}


ELEMENT_ROWS: tuple[tuple[str, str], ...] = (
    ("Hydrogen", "H"), ("Helium", "He"), ("Lithium", "Li"),
    ("Beryllium", "Be"), ("Boron", "B"), ("Carbon", "C"),
    ("Nitrogen", "N"), ("Oxygen", "O"), ("Fluorine", "F"),
    ("Neon", "Ne"), ("Sodium", "Na"), ("Magnesium", "Mg"),
    ("Aluminium", "Al"), ("Silicon", "Si"), ("Phosphorus", "P"),
    ("Sulfur", "S"), ("Chlorine", "Cl"), ("Argon", "Ar"),
    ("Potassium", "K"), ("Calcium", "Ca"), ("Scandium", "Sc"),
    ("Titanium", "Ti"), ("Vanadium", "V"), ("Chromium", "Cr"),
    ("Manganese", "Mn"), ("Iron", "Fe"), ("Cobalt", "Co"),
    ("Nickel", "Ni"), ("Copper", "Cu"), ("Zinc", "Zn"),
    ("Gallium", "Ga"), ("Germanium", "Ge"), ("Arsenic", "As"),
    ("Selenium", "Se"), ("Bromine", "Br"), ("Krypton", "Kr"),
    ("Rubidium", "Rb"), ("Strontium", "Sr"), ("Yttrium", "Y"),
    ("Zirconium", "Zr"), ("Niobium", "Nb"), ("Molybdenum", "Mo"),
    ("Technetium", "Tc"), ("Ruthenium", "Ru"), ("Rhodium", "Rh"),
    ("Palladium", "Pd"), ("Silver", "Ag"), ("Cadmium", "Cd"),
    ("Indium", "In"), ("Tin", "Sn"), ("Antimony", "Sb"),
    ("Tellurium", "Te"), ("Iodine", "I"), ("Xenon", "Xe"),
    ("Caesium", "Cs"), ("Barium", "Ba"), ("Lanthanum", "La"),
    ("Cerium", "Ce"), ("Praseodymium", "Pr"), ("Neodymium", "Nd"),
    ("Promethium", "Pm"), ("Samarium", "Sm"), ("Europium", "Eu"),
    ("Gadolinium", "Gd"), ("Terbium", "Tb"), ("Dysprosium", "Dy"),
    ("Holmium", "Ho"), ("Erbium", "Er"), ("Thulium", "Tm"),
    ("Ytterbium", "Yb"), ("Lutetium", "Lu"), ("Hafnium", "Hf"),
    ("Tantalum", "Ta"), ("Tungsten", "W"), ("Rhenium", "Re"),
    ("Osmium", "Os"), ("Iridium", "Ir"), ("Platinum", "Pt"),
    ("Gold", "Au"), ("Mercury", "Hg"), ("Thallium", "Tl"),
    ("Lead", "Pb"), ("Bismuth", "Bi"), ("Polonium", "Po"),
    ("Astatine", "At"), ("Radon", "Rn"), ("Francium", "Fr"),
    ("Radium", "Ra"), ("Actinium", "Ac"), ("Thorium", "Th"),
    ("Protactinium", "Pa"), ("Uranium", "U"), ("Neptunium", "Np"),
    ("Plutonium", "Pu"), ("Americium", "Am"), ("Curium", "Cm"),
    ("Berkelium", "Bk"), ("Californium", "Cf"), ("Einsteinium", "Es"),
    ("Fermium", "Fm"), ("Mendelevium", "Md"), ("Nobelium", "No"),
    ("Lawrencium", "Lr"), ("Rutherfordium", "Rf"), ("Dubnium", "Db"),
    ("Seaborgium", "Sg"), ("Bohrium", "Bh"), ("Hassium", "Hs"),
    ("Meitnerium", "Mt"), ("Darmstadtium", "Ds"), ("Roentgenium", "Rg"),
    ("Copernicium", "Cn"), ("Nihonium", "Nh"), ("Flerovium", "Fl"),
    ("Moscovium", "Mc"), ("Livermorium", "Lv"), ("Tennessine", "Ts"),
    ("Oganesson", "Og"),
)


CAPITAL_CLAIM_TEMPLATES: tuple[str, ...] = (
    "A guidebook describes {claim_answer}, the capital of {claim_subject}, as a hub for art and design.",
    "A travel article about {claim_subject} profiles {claim_answer}, its capital, and discusses the city's public spaces.",
    "In a documentary about {claim_subject}, {claim_answer}, the country's capital, is shown during the morning commute.",
)


CAPITAL_FILLERS: tuple[str, ...] = (
    "The guidebook also discusses museums and public spaces in the city. Public transportation is used by both residents and visitors.",
    "It goes on to mention a mix of older and newer neighborhoods. Cultural events attract visitors throughout the year.",
    "The article also describes local markets and several large parks. Universities contribute to the city's varied population.",
)


CAPITAL_QUESTION_TEMPLATES: tuple[str, ...] = (
    "What is the capital of {query_subject}?",
    "Which city is the capital of {query_subject}?",
    "Name the capital of {query_subject}.",
)


ELEMENT_CLAIM_TEMPLATES: tuple[str, ...] = (
    "A chemistry handbook lists {claim_answer}, the chemical symbol for {claim_subject}, in a table of common elements.",
    "A textbook section about {claim_subject} uses {claim_answer} as the element's chemical symbol throughout the chapter.",
    "In a classroom periodic-table display, {claim_answer} is shown as the symbol representing {claim_subject}.",
)


ELEMENT_FILLERS: tuple[str, ...] = (
    "The entry also summarizes several physical properties and common laboratory uses. Students encounter the topic in introductory chemistry courses.",
    "It goes on to discuss historical experiments and modern industrial applications. Reference books provide more detailed measurements.",
    "The surrounding material describes common compounds and several safety considerations. Researchers continue to study related reactions.",
)


ELEMENT_QUESTION_TEMPLATES: tuple[str, ...] = (
    "What is the chemical symbol for {query_subject}?",
    "Which chemical symbol represents {query_subject}?",
    "Give the chemical symbol of {query_subject}.",
)


def slugify(value: str) -> str:
    value = value.casefold().replace("’", "'")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    if not value:
        raise ValueError(f"Cannot create an identifier from {value!r}")
    return value


def stable_digest(*parts: Any, length: int = 16) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator_sha256() -> str | None:
    try:
        return file_sha256(Path(__file__).resolve())
    except OSError:
        return None


def capital_facts() -> tuple[Fact, ...]:
    return tuple(
        Fact(
            relation_id="country_capital",
            subject=country,
            answer=capital,
            answer_aliases=CAPITAL_ALIASES.get(capital, ()),
        )
        for country, capital in CAPITAL_ROWS
    )


def element_facts() -> tuple[Fact, ...]:
    return tuple(
        Fact(
            relation_id="element_symbol",
            subject=element,
            answer=symbol,
            answer_aliases=(),
        )
        for element, symbol in ELEMENT_ROWS
    )


def relation_specs() -> dict[str, RelationSpec]:
    return {
        "country_capital": RelationSpec(
            relation_id="country_capital",
            display_name="country capital",
            transfer_role="development_relation",
            facts=capital_facts(),
            claim_templates=CAPITAL_CLAIM_TEMPLATES,
            fillers=CAPITAL_FILLERS,
            question_templates=CAPITAL_QUESTION_TEMPLATES,
            answer_normalization="unicode_casefold_strip_outer_punctuation_and_whitespace",
        ),
        "element_symbol": RelationSpec(
            relation_id="element_symbol",
            display_name="chemical element symbol",
            transfer_role="heldout_transfer_relation",
            facts=element_facts(),
            claim_templates=ELEMENT_CLAIM_TEMPLATES,
            fillers=ELEMENT_FILLERS,
            question_templates=ELEMENT_QUESTION_TEMPLATES,
            answer_normalization="unicode_casefold_strip_outer_punctuation_and_whitespace",
        ),
    }


def validate_relation_spec(spec: RelationSpec) -> None:
    if len(spec.facts) < 20:
        raise ValueError(f"Relation {spec.relation_id} has too few facts for counterbalancing")
    if not (
        len(spec.claim_templates)
        == len(spec.fillers)
        == len(spec.question_templates)
        == len(TEMPLATE_BUNDLES)
        == len(RESPONSE_CONSTRAINTS)
    ):
        raise ValueError(f"Template counts are not aligned for {spec.relation_id}")

    fact_ids = [fact.fact_id for fact in spec.facts]
    subjects = [fact.subject.casefold() for fact in spec.facts]
    answers = [fact.answer.casefold() for fact in spec.facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError(f"Duplicate fact identifiers in {spec.relation_id}")
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"Duplicate subjects in {spec.relation_id}")
    if len(answers) != len(set(answers)):
        raise ValueError(
            f"Answers must be unique within {spec.relation_id} for exact lexical counterbalancing"
        )
    for fact in spec.facts:
        if re.search(r"\s", fact.answer):
            raise ValueError(
                f"Canonical answer is not one orthographic word: {fact.fact_id}={fact.answer!r}"
            )
        if fact.answer.casefold() == fact.subject.casefold():
            raise ValueError(f"Subject and answer are identical for {fact.fact_id}")

    required_claim_fields = {"claim_subject", "claim_answer"}
    for template in spec.claim_templates:
        fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
        if fields != required_claim_fields:
            raise ValueError(f"Unexpected claim fields {fields} in {template!r}")
    for template in spec.question_templates:
        fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
        if fields != {"query_subject"}:
            raise ValueError(f"Unexpected question fields {fields} in {template!r}")
    valid_policy_placements = {"question_prefix", "instruction_line"}
    for bundle in TEMPLATE_BUNDLES:
        if bundle.policy_placement not in valid_policy_placements:
            raise ValueError(
                f"Unknown policy placement {bundle.policy_placement!r} in {bundle.bundle_id}"
            )


def derangement(size: int, rng: random.Random, forbidden: Mapping[int, set[int]] | None = None) -> list[int]:
    """Return a random permutation with no fixed points and optional forbidden values."""
    forbidden = forbidden or {}
    base = list(range(size))
    for _ in range(100_000):
        candidate = base.copy()
        rng.shuffle(candidate)
        if all(candidate[i] != i and candidate[i] not in forbidden.get(i, set()) for i in base):
            return candidate
    raise RuntimeError("Could not construct a constrained derangement")


def counterbalance_maps(
    facts: Sequence[Fact],
    rounds: int,
    seed: int,
    split_by_fact_id: Mapping[str, str],
    cv_fold_by_fact_id: Mapping[str, int],
) -> list[dict[str, list[int]]]:
    """Create balanced false-answer and irrelevant-context assignments.

    For each query fact i and distractor d[i], constraints ensure:
      * false_source[i] is neither i nor d[i];
      * false_source[d[i]] is not i, so a false irrelevant context does not
        accidentally contain the correct answer to the query.
    """
    size = len(facts)
    strata: dict[tuple[str, int], list[int]] = {}
    for index, fact in enumerate(facts):
        key = (split_by_fact_id[fact.fact_id], cv_fold_by_fact_id[fact.fact_id])
        strata.setdefault(key, []).append(index)
    too_small = {key: len(indices) for key, indices in strata.items() if len(indices) < 4}
    if too_small:
        raise ValueError(
            "Every split × CV-fold counterbalancing stratum needs at least four facts; "
            f"found {too_small}"
        )

    output = []
    for round_id in range(rounds):
        distractor = [-1] * size
        false_source = [-1] * size
        for (fact_split, cv_fold), global_indices in sorted(strata.items()):
            rng = random.Random(
                stable_digest(
                    seed,
                    facts[0].relation_id,
                    round_id,
                    fact_split,
                    cv_fold,
                    length=16,
                )
            )
            local_size = len(global_indices)
            local_distractor = derangement(local_size, rng)
            for _ in range(100_000):
                local_false_source = derangement(
                    local_size,
                    rng,
                    forbidden={i: {local_distractor[i]} for i in range(local_size)},
                )
                if all(
                    local_false_source[local_distractor[i]] != i
                    for i in range(local_size)
                ):
                    break
            else:
                raise RuntimeError(
                    "Could not jointly counterbalance false answers and distractors "
                    f"in stratum {(fact_split, cv_fold)}"
                )

            for local_index, global_index in enumerate(global_indices):
                distractor[global_index] = global_indices[local_distractor[local_index]]
                false_source[global_index] = global_indices[local_false_source[local_index]]

        if -1 in distractor or -1 in false_source:
            raise RuntimeError("Counterbalancing left an unassigned fact")
        output.append({"false_source": false_source, "distractor": distractor})
    return output


def assign_fact_splits(facts: Sequence[Fact], seed: int, transfer_role: str) -> dict[str, str]:
    if transfer_role == "heldout_transfer_relation":
        return {fact.fact_id: "ood_test" for fact in facts}

    indices = list(range(len(facts)))
    rng = random.Random(stable_digest(seed, facts[0].relation_id, "fact-split", length=16))
    rng.shuffle(indices)
    n_train = int(0.60 * len(indices))
    n_validation = int(0.20 * len(indices))
    split_by_index = {}
    for rank, index in enumerate(indices):
        if rank < n_train:
            split_by_index[index] = "train"
        elif rank < n_train + n_validation:
            split_by_index[index] = "validation"
        else:
            split_by_index[index] = "test"
    return {facts[index].fact_id: split_by_index[index] for index in indices}


def assign_cv_folds(
    facts: Sequence[Fact],
    split_by_fact_id: Mapping[str, str],
    seed: int,
    n_folds: int = 5,
) -> dict[str, int]:
    """Assign balanced, deterministic folds separately inside each fact split."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    indices_by_split: dict[str, list[int]] = {}
    for index, fact in enumerate(facts):
        indices_by_split.setdefault(split_by_fact_id[fact.fact_id], []).append(index)

    output: dict[str, int] = {}
    for fact_split, indices in sorted(indices_by_split.items()):
        rng = random.Random(
            stable_digest(seed, facts[0].relation_id, fact_split, "cv-fold", length=16)
        )
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for rank, index in enumerate(shuffled):
            output[facts[index].fact_id] = rank % n_folds
    return output


def render_template(template: str, values: Mapping[str, str]) -> tuple[str, dict[str, dict[str, Any]]]:
    """Render a simple format template while recording exact slot spans."""
    pieces: list[str] = []
    spans: dict[str, dict[str, Any]] = {}
    cursor = 0
    for literal, field_name, format_spec, conversion in string.Formatter().parse(template):
        if conversion or format_spec:
            raise ValueError("Conversions and format specs are not supported in prompt templates")
        pieces.append(literal)
        cursor += len(literal)
        if field_name is None:
            continue
        if field_name not in values:
            raise KeyError(f"Missing template value {field_name!r}")
        if field_name in spans:
            raise ValueError(f"Repeated field {field_name!r} is intentionally unsupported")
        value = str(values[field_name])
        start = cursor
        pieces.append(value)
        cursor += len(value)
        spans[field_name] = {"start": start, "end": cursor, "text": value}
    return "".join(pieces), spans


class PromptBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.spans: dict[str, dict[str, Any] | None] = {}

    def add(self, text: str, span_name: str | None = None) -> tuple[int, int]:
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        end = self.length
        if span_name is not None:
            if span_name in self.spans:
                raise ValueError(f"Duplicate span name {span_name}")
            self.spans[span_name] = {"start": start, "end": end, "text": text}
        return start, end

    def add_nested_spans(self, offset: int, nested: Mapping[str, Mapping[str, Any]]) -> None:
        for name, span in nested.items():
            if name in self.spans:
                raise ValueError(f"Duplicate span name {name}")
            self.spans[name] = {
                "start": offset + int(span["start"]),
                "end": offset + int(span["end"]),
                "text": span["text"],
            }

    def build(self) -> str:
        return "".join(self.parts)


def build_context(
    spec: RelationSpec,
    bundle: TemplateBundle,
    claim_subject: str,
    claim_answer: str,
) -> tuple[str, dict[str, dict[str, Any]]]:
    claim, claim_slots = render_template(
        spec.claim_templates[bundle.claim_template_index],
        {"claim_subject": claim_subject, "claim_answer": claim_answer},
    )
    filler = spec.fillers[bundle.filler_index]
    context = f"{claim} {filler}"
    spans: dict[str, dict[str, Any]] = {
        "claim_sentence": {"start": 0, "end": len(claim), "text": claim},
        "filler": {
            "start": len(claim) + 1,
            "end": len(context),
            "text": filler,
        },
    }
    spans.update(claim_slots)
    return context, spans


def build_experiment_prompt(
    spec: RelationSpec,
    bundle: TemplateBundle,
    policy_id: str,
    claim_subject: str,
    claim_answer: str,
    query_subject: str,
) -> tuple[str, dict[str, dict[str, Any] | None], dict[str, int | None]]:
    context, context_spans = build_context(
        spec=spec,
        bundle=bundle,
        claim_subject=claim_subject,
        claim_answer=claim_answer,
    )
    question, question_spans = render_template(
        spec.question_templates[bundle.question_template_index],
        {"query_subject": query_subject},
    )
    policy = POLICY_TEMPLATES[policy_id][bundle.policy_template_index]
    constraint = RESPONSE_CONSTRAINTS[bundle.response_constraint_index]

    builder = PromptBuilder()
    context_start, _ = builder.add(context, "context")
    builder.add_nested_spans(context_start, context_spans)
    builder.add("\n\n")
    if policy:
        builder.add(policy, "policy")
        if bundle.policy_placement == "question_prefix":
            builder.add(" ")
        elif bundle.policy_placement == "instruction_line":
            builder.add("\n")
        else:
            raise ValueError(f"Unknown policy placement {bundle.policy_placement!r}")
    else:
        builder.spans["policy"] = None
    question_start, _ = builder.add(question, "question")
    builder.add_nested_spans(question_start, question_spans)
    builder.add("\n")
    builder.add(constraint, "response_constraint")
    prompt = builder.build()

    semantic_positions: dict[str, int | None] = {
        "claim_answer_end": int(builder.spans["claim_answer"]["end"]),  # type: ignore[index]
        "claim_end": int(builder.spans["claim_sentence"]["end"]),  # type: ignore[index]
        "context_end": int(builder.spans["context"]["end"]),  # type: ignore[index]
        "policy_end": (
            int(builder.spans["policy"]["end"]) if builder.spans["policy"] else None  # type: ignore[index]
        ),
        "query_subject_end": int(builder.spans["query_subject"]["end"]),  # type: ignore[index]
        "question_end": int(builder.spans["question"]["end"]),  # type: ignore[index]
        "prompt_end": len(prompt),
    }
    validate_prompt_and_spans(prompt, builder.spans, semantic_positions, constraint)
    return prompt, builder.spans, semantic_positions


def build_screening_prompt(
    spec: RelationSpec,
    bundle: TemplateBundle,
    query_subject: str,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, int]]:
    question, question_spans = render_template(
        spec.question_templates[bundle.question_template_index],
        {"query_subject": query_subject},
    )
    constraint = RESPONSE_CONSTRAINTS[bundle.response_constraint_index]
    builder = PromptBuilder()
    question_start, _ = builder.add(question, "question")
    builder.add_nested_spans(question_start, question_spans)
    builder.add("\n")
    builder.add(constraint, "response_constraint")
    prompt = builder.build()
    spans = {name: span for name, span in builder.spans.items() if span is not None}
    positions = {
        "query_subject_end": int(spans["query_subject"]["end"]),
        "question_end": int(spans["question"]["end"]),
        "prompt_end": len(prompt),
    }
    validate_prompt_and_spans(prompt, spans, positions, constraint)
    return prompt, spans, positions


def validate_prompt_and_spans(
    prompt: str,
    spans: Mapping[str, Mapping[str, Any] | None],
    semantic_positions: Mapping[str, int | None],
    expected_suffix: str,
) -> None:
    if not prompt.endswith(expected_suffix):
        raise ValueError("Prompt does not end in its one-word response constraint")
    if prompt != prompt.strip():
        raise ValueError("Prompt has leading or trailing whitespace")
    for name, span in spans.items():
        if span is None:
            continue
        start, end, text = int(span["start"]), int(span["end"]), str(span["text"])
        if not (0 <= start < end <= len(prompt)):
            raise ValueError(f"Invalid {name} span {start}:{end}")
        if prompt[start:end] != text:
            raise ValueError(f"Span text mismatch for {name}")
    for name, endpoint in semantic_positions.items():
        if endpoint is not None and not (0 < endpoint <= len(prompt)):
            raise ValueError(f"Invalid semantic endpoint {name}={endpoint}")

    ordered_names = ["claim_answer_end", "claim_end", "context_end", "query_subject_end", "question_end", "prompt_end"]
    ordered = [semantic_positions[name] for name in ordered_names if name in semantic_positions]
    if any(value is None for value in ordered):
        raise ValueError("Unexpected missing endpoint in required order")
    if list(ordered) != sorted(ordered):
        raise ValueError(f"Non-monotonic semantic positions: {semantic_positions}")


def fact_record(
    fact: Fact,
    fact_index: int,
    fact_split: str,
    cv_fold: int,
    maps: Sequence[Mapping[str, Sequence[int]]],
    facts: Sequence[Fact],
    spec: RelationSpec,
) -> dict[str, Any]:
    false_assignments = []
    distractor_assignments = []
    for round_id, mapping in enumerate(maps):
        false_fact = facts[mapping["false_source"][fact_index]]
        distractor_fact = facts[mapping["distractor"][fact_index]]
        false_assignments.append(
            {
                "counterbalance_round": round_id,
                "source_fact_id": false_fact.fact_id,
                "false_answer": false_fact.answer,
            }
        )
        distractor_assignments.append(
            {
                "counterbalance_round": round_id,
                "distractor_fact_id": distractor_fact.fact_id,
                "distractor_subject": distractor_fact.subject,
                "distractor_true_answer": distractor_fact.answer,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "fact_id": fact.fact_id,
        "relation_id": fact.relation_id,
        "relation_display_name": spec.display_name,
        "transfer_role": spec.transfer_role,
        "fact_index": fact_index,
        "relation_specific_metadata": (
            {"atomic_number": fact_index + 1}
            if fact.relation_id == "element_symbol"
            else {}
        ),
        "fact_split": fact_split,
        "cv_fold": cv_fold,
        "counterbalance_stratum": f"{fact_split}:fold-{cv_fold}",
        "subject": fact.subject,
        "world_true_answer": fact.answer,
        "acceptable_true_answers": [fact.answer, *fact.answer_aliases],
        "answer_normalization": spec.answer_normalization,
        "canonical_answer_is_one_orthographic_word": True,
        "false_answer_assignments": false_assignments,
        "distractor_assignments": distractor_assignments,
        "world_truth_status": "curated_static_fact",
        "world_truth_audit_date": RELATION_CURATION[fact.relation_id]["audit_date"],
        "model_parametric_knowledge_status": "pending_screening",
    }


def screening_record(
    fact: Fact,
    fact_split: str,
    cv_fold: int,
    spec: RelationSpec,
    bundle: TemplateBundle,
    false_answers: Sequence[str],
) -> dict[str, Any]:
    prompt, spans, semantic_positions = build_screening_prompt(spec, bundle, fact.subject)
    sample_id = "screen-" + stable_digest(fact.fact_id, bundle.bundle_id, prompt)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "parametric_knowledge_screen",
        "sample_id": sample_id,
        "fact_id": fact.fact_id,
        "relation_id": fact.relation_id,
        "transfer_role": spec.transfer_role,
        "fact_split": fact_split,
        "cv_fold": cv_fold,
        "template_bundle_id": bundle.bundle_id,
        "template_analysis_role": bundle.analysis_role,
        "messages": [{"role": "user", "content": prompt}],
        "raw_prompt": prompt,
        "spans": spans,
        "semantic_positions": semantic_positions,
        "world_true_answer": fact.answer,
        "acceptable_true_answers": [fact.answer, *fact.answer_aliases],
        "contrast_false_answers": list(false_answers),
        "answer_normalization": spec.answer_normalization,
        "screening_pass": None,
        "generated_answer": None,
        "true_answer_sequence_logprob": None,
        "false_answer_sequence_logprobs": None,
    }


def experiment_record(
    query_fact: Fact,
    query_index: int,
    fact_split: str,
    cv_fold: int,
    claim_fact: Fact,
    claim_answer: str,
    false_answer_source_fact_id: str | None,
    condition: Mapping[str, Any],
    policy_id: str,
    spec: RelationSpec,
    bundle: TemplateBundle,
    counterbalance_round: int,
) -> dict[str, Any]:
    prompt, spans, semantic_positions = build_experiment_prompt(
        spec=spec,
        bundle=bundle,
        policy_id=policy_id,
        claim_subject=claim_fact.subject,
        claim_answer=claim_answer,
        query_subject=query_fact.subject,
    )
    content_id = "content-" + stable_digest(
        query_fact.fact_id,
        counterbalance_round,
        condition["condition_id"],
        bundle.bundle_id,
        claim_fact.fact_id,
        claim_answer,
    )
    stimulus_family_id = "family-" + stable_digest(
        query_fact.fact_id,
        counterbalance_round,
    )
    factorial_group_id = "factorial-" + stable_digest(
        query_fact.fact_id,
        counterbalance_round,
        bundle.bundle_id,
    )
    sample_id = "exp-" + stable_digest(content_id, policy_id, prompt)
    relevant = bool(condition["claim_is_query_relevant"])
    true_claim = bool(condition["claim_is_world_true"])
    context_candidate = claim_answer if relevant else None

    expected_answer_under_policy: str | None
    if policy_id == "parametric":
        expected_answer_under_policy = query_fact.answer
    elif policy_id == "context" and relevant:
        expected_answer_under_policy = claim_answer
    else:
        expected_answer_under_policy = None

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "conflict_awareness_experiment",
        "sample_id": sample_id,
        "stimulus_family_id": stimulus_family_id,
        "matched_factorial_group_id": factorial_group_id,
        "content_pair_id": content_id,
        "minimal_policy_pair_id": content_id,
        "fact_id": query_fact.fact_id,
        "query_fact_index": query_index,
        "relation_id": query_fact.relation_id,
        "transfer_role": spec.transfer_role,
        "fact_split": fact_split,
        "cv_fold": cv_fold,
        "counterbalance_round": counterbalance_round,
        "condition_id": condition["condition_id"],
        "claim_is_world_true": true_claim,
        "claim_is_query_relevant": relevant,
        "claim_conflict_label": not true_claim,
        "query_conflict_label": bool(condition["query_conflict_label"]),
        "effective_claim_conflict": None,
        "effective_query_conflict": None,
        "policy_id": policy_id,
        "policy_target": POLICY_METADATA[policy_id]["policy_target"],
        "policy_analysis_role": POLICY_METADATA[policy_id]["analysis_role"],
        "template_bundle_id": bundle.bundle_id,
        "template_analysis_role": bundle.analysis_role,
        "template_indices": {
            "claim": bundle.claim_template_index,
            "filler": bundle.filler_index,
            "question": bundle.question_template_index,
            "response_constraint": bundle.response_constraint_index,
            "policy": bundle.policy_template_index,
        },
        "policy_placement": bundle.policy_placement,
        "query_subject": query_fact.subject,
        "world_true_answer": query_fact.answer,
        "acceptable_world_true_answers": [query_fact.answer, *query_fact.answer_aliases],
        "parametric_candidate_answer": query_fact.answer,
        "claim_fact_id": claim_fact.fact_id,
        "claim_subject": claim_fact.subject,
        "claim_answer": claim_answer,
        "claim_world_true_answer": claim_fact.answer,
        "false_answer_source_fact_id": false_answer_source_fact_id,
        "context_candidate_answer": context_candidate,
        "candidate_answers_are_distinct": (
            context_candidate is not None and context_candidate.casefold() != query_fact.answer.casefold()
        ),
        "expected_answer_under_policy": expected_answer_under_policy,
        "messages": [{"role": "user", "content": prompt}],
        "raw_prompt": prompt,
        "spans": spans,
        "semantic_positions": semantic_positions,
        "answer_normalization": spec.answer_normalization,
        "generated_answer": None,
        "parametric_answer_sequence_logprob": None,
        "context_answer_sequence_logprob": None,
        "context_minus_parametric_logprob_margin": None,
        "observed_knowledge_source": None,
    }


def jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )


def write_bytes(path: Path, data: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def validate_counterbalancing(
    spec: RelationSpec,
    maps: Sequence[Mapping[str, Sequence[int]]],
    split_by_fact_id: Mapping[str, str],
    cv_fold_by_fact_id: Mapping[str, int],
) -> None:
    expected = list(range(len(spec.facts)))
    for round_id, mapping in enumerate(maps):
        false_source = list(mapping["false_source"])
        distractor = list(mapping["distractor"])
        if sorted(false_source) != expected or sorted(distractor) != expected:
            raise ValueError(f"Round {round_id} is not a pair of permutations")
        for index in expected:
            if false_source[index] == index or distractor[index] == index:
                raise ValueError(f"Round {round_id} contains a fixed point")
            if false_source[index] == distractor[index]:
                raise ValueError(f"Round {round_id} reuses the same true answer in matched controls")
            if false_source[distractor[index]] == index:
                raise ValueError(
                    f"Round {round_id} false-irrelevant claim leaks the query's correct answer"
                )
            query_fact = spec.facts[index]
            query_stratum = (
                split_by_fact_id[query_fact.fact_id],
                cv_fold_by_fact_id[query_fact.fact_id],
            )
            for source_index in (false_source[index], distractor[index]):
                source_fact = spec.facts[source_index]
                source_stratum = (
                    split_by_fact_id[source_fact.fact_id],
                    cv_fold_by_fact_id[source_fact.fact_id],
                )
                if source_stratum != query_stratum:
                    raise ValueError(
                        f"Round {round_id} crosses split/fold strata: "
                        f"{query_fact.fact_id} -> {source_fact.fact_id}"
                    )


def validate_records(
    facts: Sequence[Mapping[str, Any]],
    screening: Sequence[Mapping[str, Any]],
    experiment: Sequence[Mapping[str, Any]],
    included_policies: Sequence[str],
    rounds: int,
) -> None:
    for name, records in (("facts", facts), ("screening", screening), ("experiment", experiment)):
        id_field = "fact_id" if name == "facts" else "sample_id"
        identifiers = [str(record[id_field]) for record in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Duplicate {id_field} values in {name}")

    expected_screening = len(facts) * len(TEMPLATE_BUNDLES)
    expected_experiment = (
        len(facts) * rounds * len(CONDITIONS) * len(TEMPLATE_BUNDLES) * len(included_policies)
    )
    if len(screening) != expected_screening:
        raise ValueError(f"Expected {expected_screening} screening records, found {len(screening)}")
    if len(experiment) != expected_experiment:
        raise ValueError(f"Expected {expected_experiment} experiment records, found {len(experiment)}")

    pair_groups: dict[str, list[Mapping[str, Any]]] = {}
    factorial_groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in experiment:
        pair_groups.setdefault(str(record["content_pair_id"]), []).append(record)
        factorial_groups.setdefault(str(record["matched_factorial_group_id"]), []).append(record)
        prompt = str(record["raw_prompt"])
        suffix = str(record["spans"]["response_constraint"]["text"])
        if not prompt.endswith(suffix):
            raise ValueError(f"Sample {record['sample_id']} does not end with its response constraint")
        if record["query_conflict_label"] != (
            (not record["claim_is_world_true"]) and record["claim_is_query_relevant"]
        ):
            raise ValueError(f"Incorrect conflict label in {record['sample_id']}")
        if record["condition_id"] == "false_relevant" and not record["candidate_answers_are_distinct"]:
            raise ValueError(f"False relevant answers are not distinct in {record['sample_id']}")

    for content_id, group in pair_groups.items():
        policies = {str(record["policy_id"]) for record in group}
        if policies != set(included_policies):
            raise ValueError(f"Incomplete policy pair {content_id}: {sorted(policies)}")
        invariant_fields = (
            "fact_id",
            "condition_id",
            "claim_fact_id",
            "claim_subject",
            "claim_answer",
            "world_true_answer",
            "template_bundle_id",
            "counterbalance_round",
        )
        reference = group[0]
        for record in group[1:]:
            if any(record[field] != reference[field] for field in invariant_fields):
                raise ValueError(f"Policy pair {content_id} changes content beyond policy")

    expected_conditions = {str(condition["condition_id"]) for condition in CONDITIONS}
    expected_cells = {
        (condition_id, policy_id)
        for condition_id in expected_conditions
        for policy_id in included_policies
    }
    for group_id, group in factorial_groups.items():
        cells = {(str(record["condition_id"]), str(record["policy_id"])) for record in group}
        if cells != expected_cells:
            raise ValueError(f"Incomplete factorial group {group_id}: {sorted(cells)}")
        for policy_id in included_policies:
            answers = {
                str(record["claim_answer"])
                for record in group
                if record["policy_id"] == policy_id
            }
            if len(answers) != len(CONDITIONS):
                raise ValueError(
                    f"Factorial group {group_id} does not use four distinct claim answers"
                )


def dataset_card_text(
    counts: Mapping[str, Any],
    included_policies: Sequence[str],
    rounds: int,
) -> str:
    return f"""# Conflict Awareness Under Instructed Obedience — Candidate Dataset

Schema version: `{SCHEMA_VERSION}`  
Counterbalance rounds: `{rounds}`  
Policies: `{', '.join(included_policies)}`

## Files

- `facts.jsonl`: candidate world facts and deterministic split metadata.
- `screening.jsonl`: context-free prompts used to establish model-specific parametric knowledge.
- `experiment.jsonl`: truth × relevance × answer-source-policy prompts.
- `manifest.json`: full schema, templates, counts, hashes, and generation settings.

## Required filtering before the main experiment

`query_conflict_label=true` means the paragraph contradicts the curated world fact and is relevant to the query. It does **not** yet prove an effective context-memory conflict for a specific model. A fact should enter the main experiment only after the evaluated model:

1. gives the true answer on the context-free screening prompts across prompt bundles;
2. assigns the true answer greater sequence log-probability than every false answer assigned to that fact; and
3. produces a parseable one-word answer.

For a `false_irrelevant` record, also require that the model passed screening for the `claim_fact_id` before treating `claim_conflict_label` as an effective false-claim signal.

Do not filter facts based on whether the context-grounded prompt later flips the answer. Flip rate and the continuous context-minus-parametric log-probability margin are experimental outcomes.

World-fact curation and model knowledge are intentionally separate.  Relation-level
reference links, the audit date, and notable exclusion rules are recorded in
`manifest.json`; each fact also carries the audit date.

## Leakage controls

- False answers are permutations of true answers, not a separate vocabulary.
- Every fact is used once as an irrelevant distractor per counterbalance round.
- Query facts, distractor facts, and false-answer sources are assigned within
  the same `fact_split` **and** `cv_fold`; no held-out fact can enter a training
  prompt through an irrelevant paragraph or false answer.
- `content_pair_id` links prompts whose content is identical apart from answer-source policy.
- `matched_factorial_group_id` links the four truth × relevance cells within a
  prompt bundle; `stimulus_family_id` additionally links those cells across bundles.
- The `heldout_paraphrase` bundle should not be used to fit probes or select layers.
- Character spans are authoritative; do not rediscover positions by searching prompt strings.

## Counts

```json
{json.dumps(counts, indent=2, sort_keys=True)}
```
"""


def build_dataset(
    selected_relation_ids: Sequence[str],
    output_dir: Path,
    seed: int,
    rounds: int,
    include_parametric_policy: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if rounds < 1:
        raise ValueError("--counterbalance-rounds must be at least 1")
    all_specs = relation_specs()
    unknown = sorted(set(selected_relation_ids).difference(all_specs))
    if unknown:
        raise ValueError(f"Unknown relation IDs: {unknown}")
    specs = [all_specs[relation_id] for relation_id in selected_relation_ids]
    for spec in specs:
        validate_relation_spec(spec)

    included_policies = ["neutral", "context"]
    if include_parametric_policy:
        included_policies.append("parametric")

    all_fact_records: list[dict[str, Any]] = []
    all_screening_records: list[dict[str, Any]] = []
    all_experiment_records: list[dict[str, Any]] = []
    relation_manifest: dict[str, Any] = {}

    for spec in specs:
        split_by_fact_id = assign_fact_splits(spec.facts, seed=seed, transfer_role=spec.transfer_role)
        cv_fold_by_fact_id = assign_cv_folds(
            spec.facts,
            split_by_fact_id=split_by_fact_id,
            seed=seed,
        )
        maps = counterbalance_maps(
            spec.facts,
            rounds=rounds,
            seed=seed,
            split_by_fact_id=split_by_fact_id,
            cv_fold_by_fact_id=cv_fold_by_fact_id,
        )
        validate_counterbalancing(
            spec,
            maps,
            split_by_fact_id=split_by_fact_id,
            cv_fold_by_fact_id=cv_fold_by_fact_id,
        )

        relation_fact_records: list[dict[str, Any]] = []
        relation_screening: list[dict[str, Any]] = []
        relation_experiment: list[dict[str, Any]] = []

        for fact_index, fact in enumerate(spec.facts):
            fact_split = split_by_fact_id[fact.fact_id]
            cv_fold = cv_fold_by_fact_id[fact.fact_id]
            false_answers = [
                spec.facts[mapping["false_source"][fact_index]].answer for mapping in maps
            ]
            relation_fact_records.append(
                fact_record(
                    fact=fact,
                    fact_index=fact_index,
                    fact_split=fact_split,
                    cv_fold=cv_fold,
                    maps=maps,
                    facts=spec.facts,
                    spec=spec,
                )
            )

            for bundle in TEMPLATE_BUNDLES:
                relation_screening.append(
                    screening_record(
                        fact=fact,
                        fact_split=fact_split,
                        cv_fold=cv_fold,
                        spec=spec,
                        bundle=bundle,
                        false_answers=false_answers,
                    )
                )

            for round_id, mapping in enumerate(maps):
                distractor_index = mapping["distractor"][fact_index]
                distractor_fact = spec.facts[distractor_index]
                for condition in CONDITIONS:
                    relevant = bool(condition["claim_is_query_relevant"])
                    true_claim = bool(condition["claim_is_world_true"])
                    claim_index = fact_index if relevant else distractor_index
                    claim_fact = spec.facts[claim_index]
                    if true_claim:
                        claim_answer = claim_fact.answer
                        false_source_fact_id = None
                    else:
                        false_source_index = mapping["false_source"][claim_index]
                        false_source_fact = spec.facts[false_source_index]
                        claim_answer = false_source_fact.answer
                        false_source_fact_id = false_source_fact.fact_id

                    for bundle in TEMPLATE_BUNDLES:
                        for policy_id in included_policies:
                            relation_experiment.append(
                                experiment_record(
                                    query_fact=fact,
                                    query_index=fact_index,
                                    fact_split=fact_split,
                                    cv_fold=cv_fold,
                                    claim_fact=claim_fact,
                                    claim_answer=claim_answer,
                                    false_answer_source_fact_id=false_source_fact_id,
                                    condition=condition,
                                    policy_id=policy_id,
                                    spec=spec,
                                    bundle=bundle,
                                    counterbalance_round=round_id,
                                )
                            )

        validate_records(
            facts=relation_fact_records,
            screening=relation_screening,
            experiment=relation_experiment,
            included_policies=included_policies,
            rounds=rounds,
        )
        all_fact_records.extend(relation_fact_records)
        all_screening_records.extend(relation_screening)
        all_experiment_records.extend(relation_experiment)
        relation_manifest[spec.relation_id] = {
            "display_name": spec.display_name,
            "transfer_role": spec.transfer_role,
            "fact_count": len(relation_fact_records),
            "screening_count": len(relation_screening),
            "experiment_count": len(relation_experiment),
            "fact_split_counts": count_by(relation_fact_records, "fact_split"),
            "condition_counts": count_by(relation_experiment, "condition_id"),
            "policy_counts": count_by(relation_experiment, "policy_id"),
            "template_bundle_counts": count_by(relation_experiment, "template_bundle_id"),
        }

    counts = {
        "facts": len(all_fact_records),
        "screening_records": len(all_screening_records),
        "experiment_records": len(all_experiment_records),
        "by_relation": relation_manifest,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fact_path = output_dir / "facts.jsonl"
    screening_path = output_dir / "screening.jsonl"
    experiment_path = output_dir / "experiment.jsonl"
    card_path = output_dir / "DATASET_CARD.md"
    manifest_path = output_dir / "manifest.json"

    write_bytes(fact_path, jsonl_bytes(all_fact_records), overwrite=overwrite)
    write_bytes(screening_path, jsonl_bytes(all_screening_records), overwrite=overwrite)
    write_bytes(experiment_path, jsonl_bytes(all_experiment_records), overwrite=overwrite)
    write_bytes(
        card_path,
        dataset_card_text(counts, included_policies, rounds).encode("utf-8"),
        overwrite=overwrite,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": generator_sha256(),
        "seed": seed,
        "counterbalance_rounds": rounds,
        "selected_relations": list(selected_relation_ids),
        "relation_curation": {
            relation_id: RELATION_CURATION[relation_id]
            for relation_id in selected_relation_ids
        },
        "included_policies": included_policies,
        "conditions": list(CONDITIONS),
        "policy_metadata": {policy: POLICY_METADATA[policy] for policy in included_policies},
        "policy_templates": {policy: list(POLICY_TEMPLATES[policy]) for policy in included_policies},
        "response_constraints": list(RESPONSE_CONSTRAINTS),
        "template_bundles": [asdict(bundle) for bundle in TEMPLATE_BUNDLES],
        "semantic_position_definition": {
            "claim_answer_end": "End of the claimed answer string inside the paragraph.",
            "claim_end": "End of the sentence containing the factual claim.",
            "context_end": "End of the full paragraph, after neutral filler.",
            "policy_end": "End of the answer-source instruction; null for neutral prompts.",
            "query_subject_end": "End of the queried subject inside the question.",
            "question_end": "End of the question sentence.",
            "prompt_end": "End of the one-word response constraint before chat templating.",
        },
        "required_downstream_annotations": {
            "facts": ["model_parametric_knowledge_status"],
            "screening": [
                "screening_pass",
                "generated_answer",
                "true_answer_sequence_logprob",
                "false_answer_sequence_logprobs",
            ],
            "experiment": [
                "effective_claim_conflict",
                "effective_query_conflict",
                "generated_answer",
                "parametric_answer_sequence_logprob",
                "context_answer_sequence_logprob",
                "context_minus_parametric_logprob_margin",
                "observed_knowledge_source",
            ],
        },
        "counts": counts,
        "files": {
            "facts.jsonl": {"sha256": file_sha256(fact_path), "records": len(all_fact_records)},
            "screening.jsonl": {
                "sha256": file_sha256(screening_path),
                "records": len(all_screening_records),
            },
            "experiment.jsonl": {
                "sha256": file_sha256(experiment_path),
                "records": len(all_experiment_records),
            },
            "DATASET_CARD.md": {"sha256": file_sha256(card_path)},
        },
        "warnings": [
            "Curated world truth is not equivalent to model parametric knowledge; screening is mandatory.",
            "Do not tune layers, thresholds, or hyperparameters on the element_symbol relation.",
            "Do not treat answer-source behavior as the conflict label; it is a separate outcome.",
            "Use teacher-forced sequence log-probability for all candidate answers, even when an answer is one orthographic word.",
            "Apply the model's canonical chat template downstream; raw prompts intentionally contain no model-specific control tokens.",
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes(manifest_path, manifest_bytes, overwrite=overwrite)
    return manifest


def count_by(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record[field])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conflict_awareness_dataset_v1"),
        help="Directory receiving facts.jsonl, screening.jsonl, experiment.jsonl, and metadata.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--counterbalance-rounds",
        type=int,
        default=1,
        help="Number of independent balanced false-answer/distractor assignments.",
    )
    parser.add_argument(
        "--relations",
        default="country_capital,element_symbol",
        help="Comma-separated relation IDs to generate.",
    )
    parser.add_argument(
        "--exclude-parametric-policy",
        action="store_true",
        help="Omit the explicit ignore-the-paragraph anchor control.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files already present in --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relation_ids = [value.strip() for value in args.relations.split(",") if value.strip()]
    if not relation_ids:
        raise ValueError("At least one relation must be selected")
    manifest = build_dataset(
        selected_relation_ids=relation_ids,
        output_dir=args.output_dir,
        seed=args.seed,
        rounds=args.counterbalance_rounds,
        include_parametric_policy=not args.exclude_parametric_policy,
        overwrite=args.overwrite,
    )
    print(f"Wrote dataset to {args.output_dir.resolve()}")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
