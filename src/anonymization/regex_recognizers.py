from typing import Dict, List
from presidio_analyzer import Pattern, PatternRecognizer

def build_regex_recognizers(regex_cfg: Dict) -> List[PatternRecognizer]:
    """
    Builds Presidio PatternRecognizers from configs/anonymization.yaml.
    """
    recognizers: List[PatternRecognizer] = []

    for entity_type, cfg in regex_cfg.items():
        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            continue

        patterns = cfg.get("patterns", [])
        presidio_patterns: List[Pattern] = []
        for i, pattern_str in enumerate(patterns):
            presidio_patterns.append(
                Pattern(
                    name=f"{entity_type}_pattern_{i}",
                    regex=pattern_str,
                    score=1.0,  # regex matches treated as high-precision
                )
            )
            
        recognizer = PatternRecognizer(
            supported_entity=entity_type,
            patterns=presidio_patterns,
            name=f"REGEX_{entity_type}",
        )
        recognizers.append(recognizer)

    return recognizers