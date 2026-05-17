from typing import Any, Dict, List


Entity = Dict[str, Any]

GENERIC_ORG_UNIT_SUFFIXES = (
    "division",
    "department",
    "team",
    "unit",
    "office",
    "board",
    "panel",
    "committee",
)
LEADING_ORG_ARTICLES = ("the ", "a ", "an ")


def normalize_org_text_for_unit_filter(text: str) -> str:
    normalized = " ".join(text.strip().split()).lower()
    for article in LEADING_ORG_ARTICLES:
        if normalized.startswith(article):
            return normalized[len(article):]
    return normalized


def is_generic_org_unit_detection(text: str) -> bool:
    normalized = normalize_org_text_for_unit_filter(text)
    return any(
        normalized.endswith(f" {suffix}") or normalized == suffix
        for suffix in GENERIC_ORG_UNIT_SUFFIXES
    )


def filter_generic_org_units(entities: List[Entity]) -> List[Entity]:
    return [
        entity
        for entity in entities
        if not (
            entity.get("type") in {"ORG", "ORGANIZATION"}
            and is_generic_org_unit_detection(str(entity.get("text", "")))
        )
    ]
