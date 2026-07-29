"""Deterministic topology-only baseline for stage-2 candidate reranking.

This first baseline deliberately uses no benchmark labels, CVE description,
``remoteExploit`` field or ``expected_impact`` field.  It detects three graph
shapes and gives compatible ATT&CK candidates priority while preserving the
stage-1 order within matched and unmatched groups.  Generic topology may match
at tactic level when it cannot justify one specific Technique.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from cve2attack.stage2.context_extractor import parse_fact


RULESET_VERSION = "topology-rule-priority-v2"


def _collect_facts(value: Any) -> set[str]:
    """Collect unique MulVAL fact strings from a nested context object."""
    facts: set[str] = set()
    if isinstance(value, Mapping):
        fact = value.get("fact")
        if isinstance(fact, str) and fact:
            facts.add(fact)
        for child in value.values():
            facts.update(_collect_facts(child))
    elif isinstance(value, list):
        for child in value:
            facts.update(_collect_facts(child))
    return facts


def _parsed_facts(facts: Iterable[str]) -> list[tuple[str, tuple[str, ...], str]]:
    return [(predicate, arguments, fact) for fact in sorted(facts) for predicate, arguments in [parse_fact(fact)]]


def _normalized_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if str(item)}
    return set()


def _candidate_tactics(candidate: Mapping[str, Any]) -> set[str]:
    metadata = candidate.get("metadata")
    if not isinstance(metadata, Mapping):
        return set()
    return _normalized_values(metadata.get("tactics"))


def _rule_matches_candidate(
    rule: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    """Match a rule at the evidence resolution it explicitly declares."""
    scope = str(rule.get("match_scope") or "")
    match_values = _normalized_values(rule.get("match_values"))
    technique_id = str(candidate.get("technique_id") or "")
    if scope == "technique":
        return technique_id in match_values
    if scope == "tactic":
        candidate_tactics = _candidate_tactics(candidate)
        if candidate_tactics:
            return bool(candidate_tactics & match_values)
        return technique_id in _normalized_values(rule.get("fallback_technique_ids"))
    raise ValueError(f"Unsupported topology rule match scope: {scope!r}")


def detect_topology_rules(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Detect predeclared ATT&CK-compatible shapes from graph topology only."""
    local_context = record.get("local_context")
    graph_context = record.get("graph_context")
    if not isinstance(local_context, Mapping) or not isinstance(graph_context, Mapping):
        raise ValueError("Joined record must contain local_context and graph_context objects")

    target_host = str(local_context.get("target_host") or "")
    facts = _collect_facts(local_context) | _collect_facts(graph_context)
    parsed = _parsed_facts(facts)
    rules: list[dict[str, Any]] = []

    attacker_facts = [
        fact
        for predicate, arguments, fact in parsed
        if predicate == "attackerLocated" and arguments and arguments[0] == "internet"
    ]
    internet_edges = [
        fact
        for predicate, arguments, fact in parsed
        if predicate == "hacl"
        and len(arguments) >= 2
        and arguments[0] == "internet"
        and arguments[1] == target_host
    ]
    service_facts = [
        fact
        for predicate, arguments, fact in parsed
        if predicate == "networkServiceInfo" and arguments and arguments[0] == target_host
    ]
    if attacker_facts and internet_edges and service_facts:
        rules.append(
            {
                "rule_id": "public_facing_service",
                "match_scope": "technique",
                "match_values": ["T1190"],
                "reason": "Internet-origin access reaches a network service on the target host.",
                "evidence": sorted(attacker_facts + internet_edges + service_facts),
            }
        )

    lateral_evidence: list[str] = []
    for predicate, arguments, fact in parsed:
        if predicate != "hacl" or len(arguments) < 2:
            continue
        source_host, destination_host = arguments[0], arguments[1]
        if destination_host != target_host or source_host in {"internet", target_host}:
            continue
        source_exec = [
            exec_fact
            for exec_predicate, exec_arguments, exec_fact in parsed
            if exec_predicate == "execCode"
            and exec_arguments
            and exec_arguments[0] == source_host
        ]
        if source_exec:
            lateral_evidence.extend([fact, *source_exec])
    if lateral_evidence:
        rules.append(
            {
                "rule_id": "lateral_remote_service",
                "match_scope": "technique",
                "match_values": ["T1210"],
                "reason": "Code execution on another host precedes network access to the target service.",
                "evidence": sorted(set(lateral_evidence)),
            }
        )

    direct_consequences = local_context.get("direct_consequences")
    consequence_facts = _collect_facts(direct_consequences)
    root_consequences = [
        fact
        for fact in consequence_facts
        for predicate, arguments in [parse_fact(fact)]
        if predicate == "execCode"
        and len(arguments) >= 2
        and arguments[0] == target_host
        and arguments[1] == "root"
    ]
    prior_local_exec = [
        fact
        for predicate, arguments, fact in parsed
        if predicate == "execCode"
        and len(arguments) >= 2
        and arguments[0] == target_host
        and arguments[1] != "root"
    ]
    if root_consequences and prior_local_exec:
        rules.append(
            {
                "rule_id": "local_privilege_transition",
                "match_scope": "tactic",
                "match_values": ["privilege-escalation"],
                "fallback_technique_ids": ["T1068"],
                "reason": (
                    "Existing non-root execution precedes root execution on the same host; "
                    "the topology establishes privilege escalation but not its specific mechanism."
                ),
                "evidence": sorted(set(root_consequences + prior_local_exec)),
            }
        )

    return rules


def rerank_joined_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Prioritize candidates matched by topology rules without changing the set."""
    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Joined record must contain a candidates list")

    rules = detect_topology_rules(record)
    decorated: list[tuple[bool, int, dict[str, Any]]] = []
    for original_rank, raw_candidate in enumerate(raw_candidates, start=1):
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("Each candidate must be a JSON object")
        candidate = deepcopy(dict(raw_candidate))
        technique_id = str(candidate.get("technique_id") or "")
        if not technique_id:
            raise ValueError("Each candidate must contain a technique_id")
        matched_rules = [rule for rule in rules if _rule_matches_candidate(rule, candidate)]
        metadata = candidate.get("metadata")
        metadata_copy = deepcopy(dict(metadata)) if isinstance(metadata, Mapping) else {}
        metadata_copy["stage2"] = {
            "original_rank": original_rank,
            "topology_match": bool(matched_rules),
            "matched_rules": deepcopy(matched_rules),
            "ruleset_version": RULESET_VERSION,
        }
        candidate["metadata"] = metadata_copy
        decorated.append((bool(matched_rules), original_rank, candidate))

    decorated.sort(key=lambda item: (not item[0], item[1]))
    reranked_candidates: list[dict[str, Any]] = []
    for reranked_rank, (_matched, _original_rank, candidate) in enumerate(decorated, start=1):
        candidate["metadata"]["stage2"]["reranked_rank"] = reranked_rank
        reranked_candidates.append(candidate)

    result = deepcopy(dict(record))
    result["candidates"] = reranked_candidates
    result["reranker"] = {
        "strategy": RULESET_VERSION,
        "uses_target_semantics": False,
        "detected_rules": rules,
    }
    return result


def rerank_joined_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [rerank_joined_record(record) for record in records]
