"""Executable script for CVE to ATT&CK domain rule classification."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cve_to_attack_domain.rule_classifier import CVEDomainRuleClassifier
from cve_to_attack_domain.utils import load_cve_records, save_mapping


def main() -> None:
    """Load CVE files, classify domains, and write mapping results."""
    cve_dir = PROJECT_ROOT / "og_data" / "cve"
    output_file = PROJECT_ROOT / "cve_to_attack_domain" / "result" / "cve_domain_mapping.json"

    records = load_cve_records(cve_dir)
    classifier = CVEDomainRuleClassifier()
    mapping = classifier.classify_records(records)
    save_mapping(mapping, output_file)

    print(f"[INFO] Loaded CVEs: {len(records)}")
    print(f"[INFO] Saved mapping: {output_file}")


if __name__ == "__main__":
    main()
