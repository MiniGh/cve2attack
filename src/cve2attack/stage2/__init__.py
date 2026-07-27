"""Second-stage attack-graph context extraction.

The stage-1 pipeline produces ranked ATT&CK Technique candidates.  This
package extracts the MulVAL context that will later be used to rerank those
candidates.  Candidate joining and reranking intentionally remain separate
from graph parsing so each step can be tested independently.
"""

from cve2attack.stage2.context_extractor import (
    CONTEXT_SCHEMA_VERSION,
    extract_all_cve_contexts,
    extract_cve_context,
    find_cve_nodes,
)
from cve2attack.stage2.graph_parser import (
    parse_xml_to_graph,
    reverse_for_analysis,
    summarize_graph,
)
from cve2attack.stage2.pipeline import run_stage2_experiment

__all__ = [
    "CONTEXT_SCHEMA_VERSION",
    "extract_all_cve_contexts",
    "extract_cve_context",
    "find_cve_nodes",
    "parse_xml_to_graph",
    "reverse_for_analysis",
    "summarize_graph",
    "run_stage2_experiment",
]
