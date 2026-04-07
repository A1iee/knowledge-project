import re


ENTITY_LABELS = {"Person", "Organization", "Theory", "Work", "Event"}


GAZETTEER = {
    "Person": {
        "Alan Turing",
        "Alonzo Church",
        "John von Neumann",
        "Claude Shannon",
    },
    "Organization": {
        "University of Cambridge",
        "Princeton University",
        "Bletchley Park",
        "King's College",
    },
    "Theory": {
        "Turing Machine",
        "Turing Test",
        "Church-Turing thesis",
    },
    "Work": {
        "On Computable Numbers",
    },
    "Event": {
        "World War II",
        "Enigma codebreaking",
    },
}


ALIAS_MAP = {
    "A. M. Turing": "Alan Turing",
    "A. Turing": "Alan Turing",
    "Computing Machinery and Intelligence": "Computing Machinery and Intelligence",
}


LABEL_HINT_PATTERNS = {
    "Organization": [
        re.compile(r"\b(university|college|laboratory|lab|institute|park)\b", re.I),
    ],
    "Theory": [
        re.compile(r"\b(test|machine|thesis|model|theory|concept)\b", re.I),
    ],
    "Work": [
        re.compile(r"\b(paper|article|book|publication|published|journal)\b", re.I),
    ],
    "Event": [
        re.compile(r"\b(war|conference|project|campaign|award|event)\b", re.I),
    ],
}


RELATION_PATTERNS = {
    "STUDIED_AT": [
        re.compile(r"\bstudied at\b", re.I),
        re.compile(r"\beducated at\b", re.I),
        re.compile(r"\bgraduated from\b", re.I),
    ],
    "WORKED_AT": [
        re.compile(r"\bworked at\b", re.I),
        re.compile(r"\bjoined\b", re.I),
        re.compile(r"\bserved at\b", re.I),
        re.compile(r"\bresearcher at\b", re.I),
    ],
    "PROPOSED": [
        re.compile(r"\bproposed\b", re.I),
        re.compile(r"\bintroduced\b", re.I),
        re.compile(r"\bformulated\b", re.I),
    ],
    "PUBLISHED": [
        re.compile(r"\bpublished\b", re.I),
        re.compile(r"\bwrote\b", re.I),
        re.compile(r"\bauthored\b", re.I),
    ],
    "PARTICIPATED_IN": [
        re.compile(r"\bparticipated in\b", re.I),
        re.compile(r"\binvolved in\b", re.I),
        re.compile(r"\btook part in\b", re.I),
    ],
    "INFLUENCED": [
        re.compile(r"\binfluenced\b", re.I),
        re.compile(r"\binspired\b", re.I),
        re.compile(r"\blaid foundation for\b", re.I),
    ],
    "RELATED_TO": [
        re.compile(r"\brelated to\b", re.I),
        re.compile(r"\bconnected to\b", re.I),
    ],
}


RELATION_SCHEMA = {
    "STUDIED_AT": ("Person", "Organization"),
    "WORKED_AT": ("Person", "Organization"),
    "PROPOSED": ("Person", "Theory"),
    "PUBLISHED": ("Person", "Work"),
    "PARTICIPATED_IN": ("Person", "Event"),
    "INFLUENCED": ("Person", "Any"),
    "RELATED_TO": ("Any", "Any"),
}
