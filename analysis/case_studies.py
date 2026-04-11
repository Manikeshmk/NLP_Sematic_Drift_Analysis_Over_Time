"""
analysis/case_studies.py
-------------------------
Pre-defined case studies ready for the report / demo.
Each case study demonstrates a different type of semantic change.
"""

CASE_STUDIES = [
    
    {
        "id": "broadcast",
        "word": "broadcast",
        "decade_a": 1870,
        "decade_b": 1930,
        "type": "broadening",
        "description": (
            "From agriculture (scattering seeds widely) to mass-media transmission "
            "of radio and television signals — a classic domain broadening."
        ),
    },
    {
        "id": "awful",
        "word": "awful",
        "decade_a": 1800,
        "decade_b": 1990,
        "type": "pejoration",
        "description": (
            "18th-century 'awful' meant 'inspiring awe / reverential fear'. "
            "Over two centuries it degraded to mean simply 'very bad'."
        ),
    },
    {
        "id": "computer",
        "word": "computer",
        "decade_a": 1900,
        "decade_b": 1980,
        "type": "narrowing → technical shift",
        "description": (
            "Originally a job title for a human who performs calculations, "
            "'computer' shifted entirely to refer to electronic computing machines."
        ),
    },
    {
        "id": "network",
        "word": "network",
        "decade_a": 1900,
        "decade_b": 1990,
        "type": "broadening",
        "description": (
            "From a physical mesh of threads/wires, 'network' expanded to encompass "
            "social networks, television networks, and computer networks."
        ),
    },
    {
        "id": "virus",
        "word": "virus",
        "decade_a": 1900,
        "decade_b": 1990,
        "type": "metaphorical extension",
        "description": (
            "Originally purely biological, 'virus' acquired a second major sense "
            "— computer virus — driven by the digital revolution."
        ),
    },
    {
        "id": "nice",
        "word": "nice",
        "decade_a": 1800,
        "decade_b": 1990,
        "type": "amelioration",
        "description": (
            "Middle-English 'nice' meant foolish or wanton. "
            "It ameliorated over centuries to mean pleasant or agreeable."
        ),
    },
    {
        "id": "artificial",
        "word": "artificial",
        "decade_a": 1900,
        "decade_b": 1990,
        "type": "domain extension",
        "description": (
            "'artificial' expanded from describing man-made (vs. natural) objects "
            "to encompass Artificial Intelligence as a major new semantic cluster."
        ),
    },
]


def get_case_study(word: str) -> dict | None:
    """Return the first matching case study for a word."""
    for cs in CASE_STUDIES:
        if cs["word"].lower() == word.lower():
            return cs
    return None
