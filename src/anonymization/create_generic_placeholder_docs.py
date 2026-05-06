import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml


Document = Dict[str, Any]


def read_jsonl(path: str | Path) -> List[Document]:
    docs = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    return docs


def write_jsonl(path: str | Path, docs: List[Document]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def replace_placeholders(
    text: str,
    typed_placeholders: List[str],
    generic_placeholder: str,
) -> str:
    pattern = "|".join(re.escape(placeholder) for placeholder in typed_placeholders)
    return re.sub(pattern, generic_placeholder, text)


def create_generic_placeholder_docs(config: Dict[str, Any]) -> List[Document]:
    input_path = config["input_path"]
    typed_placeholders = config["typed_placeholders"]
    generic_placeholder = config.get("generic_placeholder", "[SENSITIVE_ENTITY]")

    output_docs = []
    for doc in read_jsonl(input_path):
        output_doc = dict(doc)
        anonymized_text = output_doc.get("anonymized_text", "")

        if isinstance(anonymized_text, str):
            output_doc["anonymized_text"] = replace_placeholders(
                text=anonymized_text,
                typed_placeholders=typed_placeholders,
                generic_placeholder=generic_placeholder,
            )

        output_docs.append(output_doc)

    return output_docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to generic placeholder conversion YAML config.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_docs = create_generic_placeholder_docs(config)
    write_jsonl(config["output_path"], output_docs)

    print("Generic placeholder documents created.")
    print(f"Input: {config['input_path']}")
    print(f"Output: {config['output_path']}")


if __name__ == "__main__":
    main()
