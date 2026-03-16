"""Rule-based classifier for CVE to ATT&CK domain mapping."""

from typing import Dict, Set

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        """Fallback tqdm wrapper when tqdm is not installed."""
        return iterable

from cve_to_attack_domain.keywords import DOMAIN_KEYWORDS
from cve_to_attack_domain.utils import extract_cpe_tokens, tokenize, tokenize_source_identifier


class CVEDomainRuleClassifier:
    """Classify CVE records into ATT&CK domains using keyword intersection rules."""

    def __init__(self, domain_keywords: Dict[str, Set[str]] | None = None) -> None:
        """Initialize classifier with optional custom domain keyword sets."""
        self.domain_keywords = domain_keywords or DOMAIN_KEYWORDS

    def extract_tokens(self, record: Dict) -> Set[str]:
        """Extract normalized tokens from description, CPEs, and sourceIdentifier."""
        description_tokens = tokenize(record.get("description", ""))
        cpe_tokens = extract_cpe_tokens(record.get("cpes", []))
        source_tokens = tokenize_source_identifier(record.get("sourceIdentifier", ""))
        return description_tokens | cpe_tokens | source_tokens

    def classify_record(self, record: Dict) -> str:
        """Classify a single CVE record with priority: ICS -> Mobile -> Enterprise."""
        tokens = self.extract_tokens(record)

        if tokens & self.domain_keywords["ICS"]:
            return "ICS"
        if tokens & self.domain_keywords["Mobile"]:
            return "Mobile"
        return "Enterprise"

    def classify_records(self, records: Dict[str, Dict]) -> Dict[str, str]:
        """Classify a mapping of CVE records and return CVE-to-domain results."""
        result: Dict[str, str] = {}
        for cve_id, record in tqdm(records.items(), desc="Classifying CVEs", unit="CVE"):
            result[cve_id] = self.classify_record(record)
        return result
