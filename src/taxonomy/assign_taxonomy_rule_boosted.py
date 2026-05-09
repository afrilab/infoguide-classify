"""
Rule-boosted hierarchical taxonomy assignment.

This post-processor applies conservative banking-taxonomy rules on top of an
existing embedding/evidence output. The taxonomy levels are:
Level 1: banking domain
Level 2: functional category
Level 3: specific topic
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, Optional, Tuple


TaxonomyPath = Tuple[str, str, str]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_text(doc: Dict[str, Any], max_chars: int = 6000) -> str:
    for key in ("processed_text", "text", "clean_text", "raw_text"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value[:max_chars]
    return ""


def contains_any(text: str, patterns: Iterable[str]) -> Optional[str]:
    for pattern in patterns:
        if re.search(pattern, text):
            return pattern
    return None


def rule_path(doc: Dict[str, Any]) -> Tuple[Optional[TaxonomyPath], Optional[str]]:
    filename = normalize(str(doc.get("filename") or ""))
    source = normalize(str(doc.get("source") or ""))
    text = normalize(extract_text(doc))
    title_zone = f"{filename} {source} {text[:1000]}"
    combined = f"{filename} {source} {text[:2200]}"

    rules: list[Tuple[TaxonomyPath, str, list[str]]] = [
        (
            ("Human Resources", "Employee Management", "Workforce Analytics"),
            "human_resources_workforce",
            [r"\bhr analytics\b", r"\bemployee attrition\b", r"\bworkforce analytics\b"],
        ),
        (
            ("IT & Security", "Data Protection", "Data Quality and Integrity"),
            "data_quality",
            [r"\bdata quality\b", r"\bdata integrity\b", r"\bhow to collaborate effectively.*data quality\b"],
        ),
        (
            ("IT & Security", "Data Protection", "Security Controls"),
            "security_controls",
            [r"\bict risk\b", r"\bit risk\b", r"\bcyber\b", r"\binformation security\b"],
        ),
        (
            ("Risk & Compliance", "AML / KYC", "Beneficial Ownership"),
            "aml_beneficial_ownership",
            [r"\bbeneficial ownership\b", r"\blegal arrangements\b"],
        ),
        (
            ("Risk & Compliance", "AML / KYC", "Virtual Asset Compliance"),
            "aml_virtual_assets",
            [r"\bvirtual assets?\b", r"\bvasp\b", r"\bcryptoasset\b", r"\btargeted update\b"],
        ),
        (
            ("Risk & Compliance", "AML / KYC", "Customer Due Diligence"),
            "aml_customer_due_diligence",
            [r"\bcorrespondent banking\b", r"\bcustomer due diligence\b", r"\bknow your customer\b", r"\bkyc\b"],
        ),
        (
            ("Risk & Compliance", "AML / KYC", "Financial Crime Risk Assessment"),
            "aml_financial_crime_risk",
            [
                r"\bmoney laundering\b",
                r"\bterrorist financing\b",
                r"\bnational risk assessment\b",
                r"\baml/cft\b",
                r"\bfatf\b",
            ],
        ),
        (
            ("Financial Operations", "Reporting & Statements", "Tax Reporting"),
            "tax_reporting",
            [
                r"\bfw-?4\b",
                r"\bw-?4\b",
                r"\bfw-?9\b",
                r"\bw-?9\b",
                r"\bf1040\b",
                r"\bform 1040\b",
                r"\bisora\s+20\d{2}\s+form\b",
                r"\btax administration\b",
                r"\btax revenue\b",
                r"\brevenue authorit",
            ],
        ),
        (
            ("Customer & Accounts", "Account Management", "Account Authorization"),
            "account_authorization",
            [
                r"\bauthorizing resolutions\b",
                r"\bofficial authorization list\b",
                r"\boal\b",
                r"\bcertificate\b",
                r"\bauthorization form\b",
                r"\bcombined euac\b",
            ],
        ),
        (
            ("Customer & Accounts", "Account Management", "Custody and Correspondent Accounts"),
            "custody_correspondent_accounts",
            [r"\bthird[_ -]party[_ -]custodian\b", r"\bcustodian agreement\b", r"\bcorrespondent agreement\b"],
        ),
        (
            ("Financial Operations", "Transactions & Processing", "Deposit Operations"),
            "deposit_operations",
            [r"\bexcess balance account\b", r"\beba\b", r"\bterm deposit\b", r"\bparticipant listing\b"],
        ),
        (
            ("Customer & Accounts", "Customer Communication", "Relationship Correspondence"),
            "relationship_correspondence",
            [r"\bletter of agreement\b", r"\brelationship correspondence\b"],
        ),
        (
            ("Financial Operations", "Reporting & Statements", "Regulatory Reporting"),
            "regulatory_reporting",
            [r"\bffiec\s*\d+", r"\bcall report\b", r"\bform bq-?3\b", r"\bregulatory reporting\b"],
        ),
        (
            ("Financial Operations", "Reporting & Statements", "Securities Disclosures"),
            "securities_disclosures",
            [
                r"\bproxy statement\b",
                r"\bform 8-k\b",
                r"\bpricing supplement\b",
                r"\bpricing term sheet\b",
                r"\bprospectus supplement\b",
                r"\bterms of the notes\b",
            ],
        ),
        (
            ("Financial Operations", "Reporting & Statements", "Financial Statements"),
            "financial_statements",
            [
                r"\bannual ?report\b",
                r"\baudited financial statements\b",
                r"\bmanagement'?s discussion\b",
                r"\bfinancial statements\b",
            ],
        ),
        (
            ("Financial Operations", "Reporting & Statements", "Public Finance Reporting"),
            "public_finance_reporting",
            [
                r"\bpublic financial management\b",
                r"\bpublic expenditure\b",
                r"\bpublic finance\b",
                r"\bpefa\b",
                r"\bfiscal\b",
                r"\beconomic inclusion\b",
                r"\bdevelopment policy financing\b",
                r"\bgovtech\b",
            ],
        ),
        (
            ("Risk & Compliance", "Audit & Monitoring", "Financial Stability Assessment"),
            "financial_stability_assessment",
            [r"\bfsap\b", r"\bfinancial sector assessment\b", r"\bfinancial stability\b", r"\bcapital markets development\b"],
        ),
        (
            ("Risk & Compliance", "Audit & Monitoring", "Regulatory Monitoring"),
            "regulatory_monitoring",
            [r"\bbasel iii monitoring\b", r"\bmonitoring report\b", r"\brcap\b", r"\bregulatory consistency\b"],
        ),
        (
            ("Risk & Compliance", "Incident & Fraud", "Risk Events"),
            "risk_events",
            [r"\bbanking turmoil\b", r"\brisk event\b", r"\bstress event\b"],
        ),
        (
            ("Risk & Compliance", "Audit & Monitoring", "Supervisory Review"),
            "risk_management_review",
            [
                r"\bcounterparty credit risk\b",
                r"\bcredit risk management\b",
                r"\brisk management\b",
                r"\bstress testing\b",
            ],
        ),
        (
            ("Risk & Compliance", "Audit & Monitoring", "Supervisory Review"),
            "supervisory_review",
            [
                r"\bsupervisory review\b",
                r"\bsrep\b",
                r"\bicaap\b",
                r"\bcomprehensive assessment\b",
                r"\brecovery plans?\b",
                r"\bmain findings\b",
            ],
        ),
        (
            ("Governance & Policy", "Internal Policies", "Regulatory Policy"),
            "regulatory_policy",
            [
                r"\bcore principles\b",
                r"\bprinciples for\b",
                r"\bcapital treatment\b",
                r"\btechnical amendment\b",
                r"\bcryptoasset exposures\b",
                r"\bprudential\b",
            ],
        ),
        (
            ("Governance & Policy", "Procedures & Guidelines", "Supervisory Guidelines"),
            "supervisory_guidelines",
            [
                r"\bguidance\b",
                r"\bguidelines\b",
                r"\bmanual\b",
                r"\brisk-based approach\b",
                r"\bsupervisory guide\b",
                r"\bleveraged transactions\b",
            ],
        ),
        (
            ("Governance & Policy", "Procedures & Guidelines", "Implementation Toolkits"),
            "implementation_toolkits",
            [r"\btoolkit\b", r"\bglossary\b", r"\btechnical guide\b", r"\btemplate\b"],
        ),
        (
            ("Governance & Policy", "Internal Policies", "Governance Frameworks"),
            "governance_frameworks",
            [r"\bcorporate governance\b", r"\bgovernance framework\b", r"\bboard oversight\b"],
        ),
        (
            ("Governance & Policy", "Internal Policies", "Contractual Policy"),
            "contractual_policy",
            [r"\bagreement\b", r"\bcontract\b", r"\blegal opinion\b", r"\baccount agreement\b"],
        ),
    ]

    for path, reason, patterns in rules:
        match = contains_any(title_zone, patterns) or contains_any(combined, patterns)
        if match:
            return path, f"{reason}:{match}"

    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", default="data/processed/clean_documents.jsonl")
    ap.add_argument("--base_predictions", default="outputs/taxonomy/taxonomy_assignments_embeddings.jsonl")
    ap.add_argument("--output", default="outputs/taxonomy/taxonomy_assignments_rule_boosted.jsonl")
    args = ap.parse_args()

    docs = {doc["doc_id"]: doc for doc in read_jsonl(args.documents)}
    base_predictions = list(read_jsonl(args.base_predictions))

    output = []
    changed = 0
    for pred in base_predictions:
        doc_id = pred.get("doc_id")
        doc = docs.get(doc_id, {})
        path, reason = rule_path(doc)
        rec = dict(pred)
        debug = rec.get("debug") if isinstance(rec.get("debug"), dict) else {}

        if path:
            old_taxonomy = rec.get("taxonomy") if isinstance(rec.get("taxonomy"), dict) else {}
            rec["taxonomy"] = {"level_1": path[0], "level_2": path[1], "level_3": path[2]}
            rec["status"] = "assigned"
            rec["confidence"] = max(float(rec.get("confidence") or 0.0), 0.85)
            rec["method"] = "rule_boosted_hierarchical"
            debug["rule_boost"] = {
                "applied": True,
                "reason": reason,
                "base_taxonomy": old_taxonomy,
            }
            if old_taxonomy != rec["taxonomy"]:
                changed += 1
        else:
            rec["method"] = "rule_boosted_hierarchical"
            debug["rule_boost"] = {"applied": False}

        rec["debug"] = debug
        output.append(rec)

    write_jsonl(args.output, output)
    print(f"Wrote {len(output)} records to {args.output}")
    print(f"Rule overrides: {changed}")


if __name__ == "__main__":
    main()
