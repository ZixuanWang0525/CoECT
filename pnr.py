"""
Progressive Narrative Revision (PNR)
"""

from __future__ import annotations
__VERSION__ = "v20260430_globalpath_v3_nonumeric"

import argparse
import copy
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def strip_reasoning_prefixes(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^```(?:text|markdown|json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    changed = True
    while changed:
        changed = False
        for prefix in ("analysis", "reasoning", "thought", "assistant", "final"):
            m = re.match(rf"^{prefix}\s*[:\-\]]*\s*", s, flags=re.IGNORECASE)
            if m:
                s = s[m.end():].lstrip()
                changed = True
                break
    return s.strip()


META_POLLUTION_PATTERNS = [
    r"\bassistant\b",
    r"\banalysis\b",
    r"\bfinal\b",
    r"\bwe need to\b",
    r"\blet'?s\b",
    r"\braw prompt\b",
    r"\breference bundle\b",
    r"\bprevious event description\b",
    r"\bcurrent event description\b",
    r"\bevent_description\b",
    r"\bpositive_prompt\b",
    r"\bnegative_prompt\b",
    r"\bstructured payload\b",
    r"\bexactly in this plain-text format\b",
    r"\btask:\b",
]

PLACEHOLDER_PATTERNS = [
    r"\bthe earliest physical stage is constrained by\b",
    r"\ba later physical stage remains constrained by\b",
    r"\bremains constrained by\b",
    r"\bstring\s*<=\s*\d+\s*words\b",
    r"\bevent_description\b",
    r"\bdescription\s*placeholder\b",
    r"^\s*\.\.\.\s*$",
]


def contains_meta_pollution(text: str) -> bool:
    s = norm_text(text).lower()
    if not s:
        return False
    return any(re.search(p, s) for p in META_POLLUTION_PATTERNS)


def is_placeholder_summary(text: str) -> bool:
    s = norm_text(text).lower()
    if not s:
        return False
    return any(re.search(p, s) for p in PLACEHOLDER_PATTERNS)


def trim_sentence(text: str, max_len: int = 260) -> str:
    s = norm_text(text)
    if len(s) <= max_len:
        return s
    parts = re.split(r"(?<=[.!?])\s+", s)
    if parts and parts[0]:
        return norm_text(parts[0])
    return s[:max_len].strip()


NUMERIC_CHAR_PATTERN = re.compile(r"\d")


def contains_numeric_content(text: str) -> bool:
    return bool(NUMERIC_CHAR_PATTERN.search(norm_text(text)))


def strip_numeric_content(text: str) -> str:
    s = norm_text(text)
    if not s:
        return s

    s = re.sub(r"\([^)]*\d[^)]*\)", "", s)

    s = re.sub(r"\b\d+(?:\.\d+)?(?:\s*[A-Za-z°%/\^²³\-]+)?\b", "", s)

    s = re.sub(r"\b[A-Za-z]*\d[A-Za-z0-9°%/\^²³\-]*\b", "", s)

    s = re.sub(r"\b(?:about|around|approximately|roughly|nearly)\b", "", s, flags=re.IGNORECASE)

    s = re.sub(r"\b(?:at|to|of|with|by|under|over)\s*(?=[,.;:!?]|$)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"[,;:]\s*[,;:]+", ", ", s)
    s = re.sub(r"\d", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-")

    return s


def sanitize_prompt_text(text: str, max_len: int = 260) -> str:
    s = strip_numeric_content(text)
    s = trim_sentence(s, max_len=max_len)
    return norm_text(s)


def extract_last_block(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), flags=re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def extract_last_labeled_value(text: str, label: str) -> Optional[str]:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|\Z)", flags=re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    return norm_text(matches[-1])


def parse_tagged_output(text: str) -> Dict[str, str]:
    s = strip_reasoning_prefixes(text)
    block = extract_last_block(s, "<PNR_OUTPUT>", "</PNR_OUTPUT>") or s
    out: Dict[str, str] = {}

    ev = extract_last_labeled_value(block, "EVENT_DESCRIPTION")
    pos = extract_last_labeled_value(block, "POSITIVE_PROMPT")
    neg = extract_last_labeled_value(block, "NEGATIVE_PROMPT")
    ts = extract_last_labeled_value(block, "TRANSITION_SCORE")

    if ev:
        out["event_description"] = ev
    if pos:
        out["positive_prompt"] = pos
    if neg:
        out["negative_prompt"] = neg
    if ts:
        out["transition_score"] = ts

    if not out:
        first_line = block.splitlines()[0].strip() if block.strip() else ""
        if first_line and not re.match(r"^[A-Z_]+\s*:", first_line):
            out["event_description"] = norm_text(first_line)
    return out



class LocalHFChatClient:
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        dtype: str = "bfloat16",
        gpu_id: int = 0,
    ) -> None:
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = "auto"

        print(f"[PNR] version={__VERSION__} gpu_id={gpu_id}", flush=True)
        print(f"[PNR] loading tokenizer from: {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"[PNR] loading model from: {model_path} on {self.device}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.model.eval()

    def _build_inputs(self, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self.tokenizer(rendered, return_tensors="pt")
        else:
            rendered = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
            inputs = self.tokenizer(rendered, return_tensors="pt")
        return inputs

    def chat_text(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                inputs = self._build_inputs(system_prompt, user_prompt)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                prompt_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.temperature > 0,
                        temperature=self.temperature if self.temperature > 0 else None,
                        top_p=self.top_p if self.temperature > 0 else None,
                        repetition_penalty=self.repetition_penalty,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                gen_ids = output_ids[0][prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                return strip_reasoning_prefixes(text)
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.2 * attempt)
                else:
                    raise RuntimeError(f"Local LLM text generation failed after {max_retries} retries: {last_err}")
        raise RuntimeError(f"Unreachable LLM state: {last_err}")



@dataclass
class Node:
    id: str
    type: str = "object"
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    subject: str
    relation: str
    object: str


@dataclass
class PhysicalCondition:
    physical_parameter_vector: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    derived_quantities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    condition_summary: str = ""


@dataclass
class SceneGraph:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    graph_summary: str = ""


@dataclass
class Event:
    event_id: int
    physical_condition: PhysicalCondition
    scene_graph: SceneGraph


@dataclass
class PFGContext:
    index: int
    input_text: str
    category: Optional[str] = None
    category_name: Optional[str] = None
    physical_law_name: Optional[str] = None
    physical_law_reason: Optional[str] = None
    formula_name: Optional[str] = None
    equation_latex: Optional[str] = None
    formula_role_in_scene: Optional[str] = None
    variable_semantics: Dict[str, str] = field(default_factory=dict)
    parameter_table: List[Dict[str, Any]] = field(default_factory=list)




def parse_pfg_item(item: Dict[str, Any]) -> PFGContext:
    grounded = item.get("grounded_formula", {}) or {}
    law = item.get("physical_law", {}) or {}
    return PFGContext(
        index=item.get("index"),
        input_text=norm_text(item.get("input_text", "")),
        category=item.get("category"),
        category_name=item.get("category_name"),
        physical_law_name=law.get("name"),
        physical_law_reason=law.get("reason"),
        formula_name=grounded.get("formula_name"),
        equation_latex=grounded.get("equation_latex"),
        formula_role_in_scene=grounded.get("formula_role_in_scene"),
        variable_semantics=grounded.get("variable_semantics", {}) or {},
        parameter_table=item.get("parameter_table", []) or [],
    )


def sanitize_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = norm_text(n.get("id", ""))
        if not nid:
            continue
        attrs = n.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        out.append({
            "id": nid,
            "type": n.get("type", "object") or "object",
            "attributes": attrs,
        })
    return out


def sanitize_event_dict(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    ed = copy.deepcopy(event_dict)
    pc = ed.setdefault("physical_condition", {})
    sg = ed.setdefault("scene_graph", {})
    pc["condition_summary"] = norm_text(pc.get("condition_summary", ""))
    sg["graph_summary"] = norm_text(sg.get("graph_summary", ""))
    sg["nodes"] = sanitize_nodes(sg.get("nodes", []) or [])
    sg["edges"] = sg.get("edges", []) or []

    if is_placeholder_summary(pc.get("condition_summary", "")) and sg.get("graph_summary"):
        pc["condition_summary"] = sg["graph_summary"]
    return ed


def parse_event(event_dict: Dict[str, Any]) -> Event:
    event_dict = sanitize_event_dict(event_dict)
    pc = event_dict.get("physical_condition", {}) or {}
    sg = event_dict.get("scene_graph", {}) or {}
    return Event(
        event_id=event_dict.get("event_id", -1),
        physical_condition=PhysicalCondition(
            physical_parameter_vector=pc.get("physical_parameter_vector", {}) or {},
            derived_quantities=pc.get("derived_quantities", {}) or {},
            condition_summary=norm_text(pc.get("condition_summary", "")),
        ),
        scene_graph=SceneGraph(
            nodes=[Node(**n) for n in (sg.get("nodes", []) or [])],
            edges=[Edge(**e) for e in (sg.get("edges", []) or [])],
            graph_summary=norm_text(sg.get("graph_summary", "")),
        ),
    )


def _node_map(nodes: List[Node]) -> Dict[str, Node]:
    return {n.id: n for n in nodes}


def _edge_key(e: Edge) -> Tuple[str, str, str]:
    return (e.subject, e.relation, e.object)


def _edge_set(edges: List[Edge]) -> set:
    return {_edge_key(e) for e in edges}


def _dict_diff(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    keys = sorted(set(prev.keys()) | set(curr.keys()))
    for k in keys:
        pv = prev.get(k)
        cv = curr.get(k)
        if pv != cv:
            out[k] = {"prev": pv, "curr": cv}
    return out


def extract_delta(prev_event: Optional[Event], curr_event: Event) -> Dict[str, Any]:
    curr_pc = curr_event.physical_condition
    curr_sg = curr_event.scene_graph
    if prev_event is None:
        return {
            "mode": "bootstrap",
            "event_id": curr_event.event_id,
            "condition_summary": curr_pc.condition_summary,
            "graph_summary": curr_sg.graph_summary,
            "physical_parameter_changes": copy.deepcopy(curr_pc.physical_parameter_vector),
            "derived_quantity_changes": copy.deepcopy(curr_pc.derived_quantities),
            "node_attribute_changes": {n.id: copy.deepcopy(n.attributes) for n in curr_sg.nodes if n.attributes},
            "edge_additions": [asdict(e) for e in curr_sg.edges],
            "edge_removals": [],
            "node_additions": [n.id for n in curr_sg.nodes],
            "node_removals": [],
        }

    prev_pc = prev_event.physical_condition
    prev_sg = prev_event.scene_graph

    ppv_delta: Dict[str, Any] = {}
    for k in sorted(set(prev_pc.physical_parameter_vector) | set(curr_pc.physical_parameter_vector)):
        pv = prev_pc.physical_parameter_vector.get(k)
        cv = curr_pc.physical_parameter_vector.get(k)
        if pv != cv:
            ppv_delta[k] = {"prev": pv, "curr": cv}

    dq_delta: Dict[str, Any] = {}
    for k in sorted(set(prev_pc.derived_quantities) | set(curr_pc.derived_quantities)):
        pv = prev_pc.derived_quantities.get(k)
        cv = curr_pc.derived_quantities.get(k)
        if pv != cv:
            dq_delta[k] = {"prev": pv, "curr": cv}

    prev_nodes = _node_map(prev_sg.nodes)
    curr_nodes = _node_map(curr_sg.nodes)
    node_additions = sorted(set(curr_nodes) - set(prev_nodes))
    node_removals = sorted(set(prev_nodes) - set(curr_nodes))

    node_attr_delta: Dict[str, Any] = {}
    for nid in sorted(set(prev_nodes) & set(curr_nodes)):
        diff = _dict_diff(prev_nodes[nid].attributes, curr_nodes[nid].attributes)
        if diff:
            node_attr_delta[nid] = diff

    prev_edges = _edge_set(prev_sg.edges)
    curr_edges = _edge_set(curr_sg.edges)
    edge_additions = [
        {"subject": s, "relation": r, "object": o}
        for (s, r, o) in sorted(curr_edges - prev_edges)
    ]
    edge_removals = [
        {"subject": s, "relation": r, "object": o}
        for (s, r, o) in sorted(prev_edges - curr_edges)
    ]

    return {
        "mode": "incremental",
        "event_id": curr_event.event_id,
        "condition_summary": curr_pc.condition_summary,
        "graph_summary": curr_sg.graph_summary,
        "physical_parameter_changes": ppv_delta,
        "derived_quantity_changes": dq_delta,
        "node_attribute_changes": node_attr_delta,
        "edge_additions": edge_additions,
        "edge_removals": edge_removals,
        "node_additions": node_additions,
        "node_removals": node_removals,
    }


def build_soft_reference_bundle(pfg: PFGContext, prev_event: Optional[Event], curr_event: Event) -> Dict[str, Any]:
    delta = extract_delta(prev_event, curr_event)
    return {
        "raw_prompt": pfg.input_text,
        "physical_law": {
            "name": pfg.physical_law_name,
            "reason": pfg.physical_law_reason,
            "category": pfg.category_name,
        },
        "formula": {
            "name": pfg.formula_name,
            "equation_latex": pfg.equation_latex,
            "role_in_scene": pfg.formula_role_in_scene,
            "variable_semantics": pfg.variable_semantics,
        },
        "parameter_reference": pfg.parameter_table,
        "current_event": {
            "event_id": curr_event.event_id,
            "physical_condition": asdict(curr_event.physical_condition),
            "scene_graph": asdict(curr_event.scene_graph),
        },
        "delta_from_previous": delta,
    }


def build_reference_bundles_from_events(pfg: PFGContext, events: List[Event]) -> List[Dict[str, Any]]:
    bundles: List[Dict[str, Any]] = []
    prev_event: Optional[Event] = None
    for curr_event in events:
        bundles.append(build_soft_reference_bundle(pfg, prev_event, curr_event))
        prev_event = curr_event
    return bundles



STOPWORDS = {
    'a','an','the','of','in','on','into','to','from','with','and','or','as','at','by','for','is','are','was','were',
    'be','being','been','that','this','it','its','their','his','her','inside','outside','within','filled','slowly','gently','very','now',
    'then','first','finally','while','during','under','over','up','down','across','through','remains','remain',
    'stage','physical','current','event','law','formula'
}


def _tokenize_content(text: str) -> set:
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", norm_text(text).lower())
    return {t for t in toks if len(t) > 2 and t not in STOPWORDS}


def _bundle_nodes(bundle: Dict[str, Any]) -> set:
    nodes = bundle.get('current_event', {}).get('scene_graph', {}).get('nodes', []) or []
    return {norm_text(n.get('id', '')).lower() for n in nodes if norm_text(n.get('id', ''))}


def _bundle_edges(bundle: Dict[str, Any]) -> set:
    edges = bundle.get('current_event', {}).get('scene_graph', {}).get('edges', []) or []
    return {(norm_text(e.get('subject', '')).lower(), norm_text(e.get('relation', '')).lower(), norm_text(e.get('object', '')).lower()) for e in edges}


def _delta_edges(bundle: Dict[str, Any], field: str) -> set:
    arr = bundle.get('delta_from_previous', {}).get(field, []) or []
    return {(norm_text(e.get('subject', '')).lower(), norm_text(e.get('relation', '')).lower(), norm_text(e.get('object', '')).lower()) for e in arr}


def _delta_size(bundle: Dict[str, Any]) -> int:
    d = bundle.get('delta_from_previous', {}) or {}
    return (
        len(d.get('physical_parameter_changes', {}) or {}) +
        len(d.get('derived_quantity_changes', {}) or {}) +
        len(d.get('node_attribute_changes', {}) or {}) +
        len(d.get('edge_additions', []) or []) +
        len(d.get('edge_removals', []) or []) +
        len(d.get('node_additions', []) or []) +
        len(d.get('node_removals', []) or [])
    )


def bundle_quality_score(bundle: Dict[str, Any], raw_prompt: str) -> float:
    summary = reference_core_summary(bundle)
    score = 0.0
    if summary:
        score += 1.2
    else:
        score -= 2.0
    if is_placeholder_summary(summary):
        score -= 1.25
    if contains_meta_pollution(summary):
        score -= 1.5

    nodes = _bundle_nodes(bundle)
    edges = _bundle_edges(bundle)
    score += min(0.6, 0.12 * len(nodes))
    score += min(0.6, 0.08 * len(edges))
    score += min(0.8, 0.06 * _delta_size(bundle))

    ptoks = _tokenize_content(raw_prompt)
    stoks = _tokenize_content(summary) | nodes
    if ptoks:
        overlap = len(ptoks & stoks) / max(1, len(ptoks))
        score += 1.0 * overlap
        if overlap < 0.15:
            score -= 0.4

 
    mode = bundle.get('delta_from_previous', {}).get('mode')
    if mode == 'incremental' and _delta_size(bundle) == 0:
        score -= 0.7
    return score


def transition_score(prev_bundle: Dict[str, Any], curr_bundle: Dict[str, Any], raw_prompt: str) -> float:
    score = 0.0

    prev_nodes = _bundle_nodes(prev_bundle)
    curr_nodes = _bundle_nodes(curr_bundle)
    if prev_nodes or curr_nodes:
        jacc = len(prev_nodes & curr_nodes) / max(1, len(prev_nodes | curr_nodes))
        score += 1.4 * jacc

    prev_edges = _bundle_edges(prev_bundle)
    curr_edges = _bundle_edges(curr_bundle)
    if prev_edges or curr_edges:
        edge_jacc = len(prev_edges & curr_edges) / max(1, len(prev_edges | curr_edges))
        score += 0.5 * edge_jacc

    prev_added = _delta_edges(prev_bundle, 'edge_additions')
    curr_removed = _delta_edges(curr_bundle, 'edge_removals')
    curr_added = _delta_edges(curr_bundle, 'edge_additions')
    prev_removed = _delta_edges(prev_bundle, 'edge_removals')

    rollback = len(prev_added & curr_removed)
    readd = len(prev_removed & curr_added)
    score -= 0.45 * rollback
    score -= 1.6 * readd

    removed_pairs = {(s, o) for (s, _, o) in curr_removed}
    added_pairs = {(s, o) for (s, _, o) in curr_added}
    replacement_credit = len(removed_pairs & added_pairs)
    score += 0.7 * replacement_credit

    prev_summary = reference_core_summary(prev_bundle)
    curr_summary = reference_core_summary(curr_bundle)
    if norm_text(prev_summary).lower() == norm_text(curr_summary).lower() and norm_text(curr_summary):
        score -= 0.6

    curr_delta_size = _delta_size(curr_bundle)
    if len(curr_removed) > len(curr_added) and curr_delta_size <= len(curr_removed) + 1:
        score -= 0.8
    if len(curr_removed) >= 2 and len(curr_added) <= 1:
        score -= 0.5 * (len(curr_removed) - len(curr_added))

    ptoks = _tokenize_content(raw_prompt)
    curr_toks = _tokenize_content(curr_summary) | curr_nodes
    if ptoks:
        overlap = len(ptoks & curr_toks) / max(1, len(ptoks))
        score += 0.6 * overlap

    if is_placeholder_summary(curr_summary):
        score -= 0.8
    if contains_meta_pollution(curr_summary):
        score -= 1.2

    return score


def select_best_event_path(reference_bundles: List[Dict[str, Any]], raw_prompt: str, client: Optional[LocalHFChatClient] = None, physical_law_name: Optional[str] = None, judge_mode: str = "hybrid", allowed_k: Tuple[int, ...] = (1, 2, 4)) -> Tuple[List[int], List[Dict[str, Any]]]:
    if not reference_bundles:
        return [], []

    n = len(reference_bundles)
    valid_k = [k for k in allowed_k if 1 <= k <= n]
    if not valid_k:
        valid_k = [1]

    node_scores = [bundle_quality_score(b, raw_prompt) for b in reference_bundles]
    cache: Dict[Tuple[int, int], float] = {}
    JUDGE_WEIGHT = 4.0
    GAP_PENALTY = 1.3
    MIN_TRANSITION_SCORE = 0.25

    def judge_bonus(j: int, i: int) -> float:
        if judge_mode == "heuristic":
            return 0.0
        val = judge_transition_plausibility(client, raw_prompt, physical_law_name, reference_bundles[j], reference_bundles[i], cache)
        return JUDGE_WEIGHT * (val - 0.5)

    def start_score(i: int) -> float:
        pos_bonus = 0.6 * (1.0 - (i / max(1, n - 1))) if n > 1 else 0.6
        return 0.75 * node_scores[i] + pos_bonus

    best_candidate: Optional[Tuple[float, List[int]]] = None

    for K in valid_k:
        NEG = -1e18
        dp = [[NEG] * (K + 1) for _ in range(n)]
        parent = [[-1] * (K + 1) for _ in range(n)]

        for i in range(n):
            dp[i][1] = start_score(i)

        for k in range(2, K + 1):
            for i in range(n):
                best_val = NEG
                best_j = -1
                for j in range(i):
                    if dp[j][k - 1] <= NEG / 2:
                        continue
                    tscore = transition_score(reference_bundles[j], reference_bundles[i], raw_prompt)
                    if judge_mode in {"llm", "hybrid"}:
                        tscore += judge_bonus(j, i)
                    gap = i - j - 1
                    if gap > 0:
                        tscore -= GAP_PENALTY * gap
                    if tscore <= MIN_TRANSITION_SCORE:
                        continue
                    cand = dp[j][k - 1] + tscore + 0.35 * node_scores[i]
                    if cand > best_val:
                        best_val = cand
                        best_j = j
                dp[i][k] = best_val
                parent[i][k] = best_j

        end = max(range(n), key=lambda i: dp[i][K])
        if dp[end][K] <= NEG / 2:
            continue

        path_idx: List[int] = []
        cur_i, cur_k = end, K
        while cur_i != -1 and cur_k >= 1:
            path_idx.append(cur_i)
            cur_i = parent[cur_i][cur_k]
            cur_k -= 1
        path_idx.reverse()

        if len(path_idx) != K:
            continue

        total_score = dp[end][K] + 0.12 * K
        if best_candidate is None or total_score > best_candidate[0]:
            best_candidate = (total_score, path_idx)

    if best_candidate is None:
        best_i = max(range(n), key=start_score)
        path_idx = [best_i]
    else:
        path_idx = best_candidate[1]

    selected = [reference_bundles[i] for i in path_idx]
    return path_idx, selected


def rebuild_selected_reference_bundles(pfg: PFGContext, events: List[Event], selected_indices: List[int]) -> List[Dict[str, Any]]:
    selected_events = [events[i] for i in selected_indices]
    return build_reference_bundles_from_events(pfg, selected_events)



def judge_transition_plausibility(client: Optional[LocalHFChatClient], raw_prompt: str, physical_law_name: Optional[str], prev_bundle: Dict[str, Any], curr_bundle: Dict[str, Any], cache: Optional[Dict[Tuple[int,int], float]] = None) -> float:
    if client is None:
        return 0.5
    prev_id = int(prev_bundle.get("current_event", {}).get("event_id", -1))
    curr_id = int(curr_bundle.get("current_event", {}).get("event_id", -1))
    key = (prev_id, curr_id)
    if cache is not None and key in cache:
        return cache[key]
    try:
        raw_out = client.chat_text(
            TRANSITION_JUDGE_SYSTEM,
            build_transition_judge_user_prompt(raw_prompt, physical_law_name, prev_bundle, curr_bundle),
        )
        tagged = parse_tagged_output(raw_out)
        text = tagged.get("transition_score", raw_out)
        m = re.search(r"([0-3])", norm_text(text))
        if m:
            val = int(m.group(1)) / 3.0
        else:
            val = 0.5
    except Exception:
        val = 0.5
    if cache is not None:
        cache[key] = val
    return val



BOOTSTRAP_SYSTEM = """
You are implementing Progressive Narrative Revision (PNR) in a physics-aware video generation pipeline.

Write the first event description w1.
Rules:
1. Use the original text prompt as the main semantic source.
2. Treat PECR metadata as reference evidence, not absolute truth.
3. Keep object identities stable and visually grounded.
4. Prefer visible scene changes and interactions over abstract formula language.
5. Do not invent future outcomes.
6. If PECR metadata contains local noise, prefer raw-prompt faithfulness and physical plausibility.
7. Output exactly one concise event description.
8. Do not include any digits, numbers, units, equations, magnitudes, angles, or symbolic parameters.
9. Describe only the observable phenomenon itself.
10. Do not output reasoning.
11. Return exactly this format:
<PNR_OUTPUT>
EVENT_DESCRIPTION: ...
</PNR_OUTPUT>
""".strip()


REVISION_SYSTEM = """
You are implementing Progressive Narrative Revision (PNR) in a physics-aware video generation pipeline.

Produce the current event description wt by minimally revising the previous event description w_{t-1}.
Rules:
1. The previous event description is the continuity anchor.
2. The raw prompt remains the main semantic source.
3. Treat PECR metadata as reference evidence, not absolute truth.
4. Revise only the parts that need to change for the current event.
5. Preserve unchanged objects, identities, and stable context.
6. Reflect the current physical phase using visible and causal language.
7. If PECR metadata is locally noisy, prefer continuity, raw-prompt faithfulness, and physical plausibility.
8. Do not include any digits, numbers, units, equations, magnitudes, angles, or symbolic parameters.
9. Describe only the observable phenomenon itself.
10. Do not output reasoning.
11. Return exactly this format:
<PNR_OUTPUT>
EVENT_DESCRIPTION: ...
</PNR_OUTPUT>
""".strip()


REPAIR_EVENT_SYSTEM = """
You are repairing a corrupted event description output.
Rules:
1. Remove all reasoning, meta text, and protocol text.
2. Keep only one clean event description.
3. Preserve continuity with the previous event description.
4. Stay faithful to the raw prompt and current physical phase.
5. Remove all digits, numbers, units, equations, magnitudes, angles, and symbolic parameters.
6. Keep only a clean natural-language description of the visible phenomenon.
7. Output exactly this format:
<PNR_OUTPUT>
EVENT_DESCRIPTION: ...
</PNR_OUTPUT>
""".strip()


CONDENSE_SYSTEM = """
You are implementing semantic condensation in TCP.

Merge multiple event descriptions into one causally consistent positive semantic prompt.
Rules:
1. Preserve chronological order.
2. Use causal connectives such as first, then, as, while, causing, finally.
3. Remove redundancy.
4. Keep object identities consistent.
5. Make the positive prompt concise and suitable for a video diffusion model.
6. Do not include any digits, numbers, units, equations, magnitudes, angles, or symbolic parameters in the positive prompt.
7. Describe only visible phenomena and temporal progression.
8. The negative prompt must stay generic and safe.
9. Do not output reasoning.
10. Return exactly this format:
<PNR_OUTPUT>
POSITIVE_PROMPT: ...
NEGATIVE_PROMPT: blurry, artifacts, duplicate objects, identity inconsistency, abrupt discontinuity, broken geometry, implausible motion
</PNR_OUTPUT>
""".strip()


REPAIR_CONDENSE_SYSTEM = """
You are repairing a corrupted condensed prompt output.
Rules:
1. Remove all reasoning, meta text, and protocol text.
2. Preserve the chronological event progression.
3. Remove all digits, numbers, units, equations, magnitudes, angles, and symbolic parameters from the positive prompt.
4. Output exactly this format:
<PNR_OUTPUT>
POSITIVE_PROMPT: ...
NEGATIVE_PROMPT: blurry, artifacts, duplicate objects, identity inconsistency, abrupt discontinuity, broken geometry, implausible motion
</PNR_OUTPUT>
""".strip()


TRANSITION_JUDGE_SYSTEM = """
You are scoring whether a candidate current event can plausibly follow a previous event under the same raw prompt and physical law.
Rules:
1. Judge causal continuity and forward progression.
2. Treat PECR metadata as reference evidence, not absolute truth.
3. Penalize local rollback, direction reversal, or identity inconsistency.
4. Use this scale only: 0=implausible, 1=weak, 2=plausible, 3=strongly plausible.
5. Do not output reasoning.
6. Return exactly this format:
<PNR_OUTPUT>
TRANSITION_SCORE: 0
</PNR_OUTPUT>
""".strip()


def build_bootstrap_user_prompt(ref_bundle: Dict[str, Any]) -> str:
    return "Write the first event description.\n\nReference bundle:\n" + json.dumps(ref_bundle, ensure_ascii=False, indent=2)


def build_revision_user_prompt(prev_description: str, ref_bundle: Dict[str, Any]) -> str:
    return (
        "Minimally revise the previous event description to obtain the current event description.\n\n"
        f"Previous event description:\n{prev_description}\n\n"
        f"Reference bundle:\n{json.dumps(ref_bundle, ensure_ascii=False, indent=2)}"
    )


def build_repair_event_user_prompt(raw_output: str, raw_prompt: str, prev_description: str, ref_bundle: Dict[str, Any], issues: List[str]) -> str:
    return (
        f"Corrupted output:\n{raw_output}\n\n"
        f"Raw prompt:\n{raw_prompt}\n\n"
        f"Previous event description:\n{prev_description}\n\n"
        f"Reference bundle:\n{json.dumps(ref_bundle, ensure_ascii=False, indent=2)}\n\n"
        f"Issues to fix:\n- " + "\n- ".join(issues)
    )


def build_condense_user_prompt(raw_prompt: str, event_descriptions: List[str], physical_law_name: Optional[str]) -> str:
    numbered = "\n".join([f"Event {i+1}: {d}" for i, d in enumerate(event_descriptions)])
    payload = {
        "raw_prompt": raw_prompt,
        "physical_law_name": physical_law_name,
        "event_descriptions": event_descriptions,
    }
    return (
        f"Original prompt: {raw_prompt}\n"
        f"Physical law: {physical_law_name}\n"
        f"Event descriptions:\n{numbered}\n\n"
        f"Structured payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_repair_condense_user_prompt(raw_output: str, raw_prompt: str, event_descriptions: List[str], issues: List[str]) -> str:
    numbered = "\n".join([f"Event {i+1}: {d}" for i, d in enumerate(event_descriptions)])
    return (
        f"Corrupted output:\n{raw_output}\n\n"
        f"Original prompt:\n{raw_prompt}\n\n"
        f"Event descriptions:\n{numbered}\n\n"
        f"Issues to fix:\n- " + "\n- ".join(issues)
    )


def build_transition_judge_user_prompt(raw_prompt: str, physical_law_name: Optional[str], prev_bundle: Dict[str, Any], curr_bundle: Dict[str, Any]) -> str:
    payload = {
        "raw_prompt": raw_prompt,
        "physical_law_name": physical_law_name,
        "previous_event": {
            "event_id": prev_bundle.get("current_event", {}).get("event_id"),
            "summary": reference_core_summary(prev_bundle),
            "delta_from_previous": prev_bundle.get("delta_from_previous", {}),
        },
        "current_event": {
            "event_id": curr_bundle.get("current_event", {}).get("event_id"),
            "summary": reference_core_summary(curr_bundle),
            "delta_from_previous": curr_bundle.get("delta_from_previous", {}),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)



def validate_event_description(desc: str, raw_prompt: str, prev_desc: Optional[str], event_id: int, ref_bundle: Optional[Dict[str, Any]] = None) -> List[str]:
    issues: List[str] = []
    s = norm_text(desc)

    if not s:
        issues.append("empty event description")
        return issues
    if contains_meta_pollution(s):
        issues.append("contains reasoning or protocol pollution")
    if contains_numeric_content(s):
        issues.append("contains numeric details")
    if len(s) < 8:
        issues.append("too short to be a usable event description")
    if len(s) > 320:
        issues.append("too long and likely not concise")
    if not re.search(r"[A-Za-z]", s):
        issues.append("contains no alphabetic content")

    return issues


def validate_condensed_prompts(positive_prompt: str, negative_prompt: str) -> List[str]:
    issues: List[str] = []
    if not positive_prompt:
        issues.append("empty positive prompt")
    if contains_meta_pollution(positive_prompt):
        issues.append("positive prompt contains reasoning or protocol pollution")
    if contains_numeric_content(positive_prompt):
        issues.append("positive prompt contains numeric details")
    if positive_prompt and not re.search(r"\b(first|then|finally|as|while|causing)\b", positive_prompt.lower()):
        issues.append("positive prompt lacks clear causal connective")
    if negative_prompt and contains_meta_pollution(negative_prompt):
        issues.append("negative prompt contains reasoning or protocol pollution")
    return issues


ENTITY_COMPARE_SYSTEM = """
You are a semantic consistency checker for a video generation prompt.

Your task is to compare the ORIGINAL_PROMPT and the GENERATED_PROMPT.

Check only whether the concrete visible entities explicitly mentioned in ORIGINAL_PROMPT
are still preserved in GENERATED_PROMPT.

Rules:
1. Do not judge physical correctness.
2. Do not judge grammar or style.
3. Do not require the same sentence structure.
4. An entity is preserved if it is clearly present with the same meaning.
5. Mark fail if an original entity is omitted, renamed, merged into another entity, or replaced.
6. Return ONE valid JSON object only.

Schema:
{
  "status": "pass|fail",
  "original_entities": ["string"],
  "missing_or_changed_entities": ["string"],
  "reason": "string"
}
""".strip()


ENTITY_REPAIR_SYSTEM = """
You are a prompt repair assistant.

You will receive:
- the original prompt
- the generated positive prompt
- a semantic consistency report

Your task is to minimally repair the generated positive prompt so that all concrete visible entities
from the original prompt are preserved.

Rules:
1. Preserve the existing temporal and causal progression as much as possible.
2. Restore missing or changed entities using their original wording.
3. Do not introduce new physical outcomes that are not supported by the original prompt or generated prompt.
4. Do not add numbers, units, equations, magnitudes, angles, or symbolic parameters.
5. Return ONE valid JSON object only.

Schema:
{
  "positive_prompt": "string"
}
""".strip()


def extract_json_object_from_text(text: str) -> Dict[str, Any]:
    text = strip_reasoning_prefixes(text or "")
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    try:
        return json.loads(text)
    except Exception:
        pass

    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for st in starts:
        depth = 0
        for i in range(st, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = text[st:i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        continue

    return {
        "status": "checker_failed",
        "reason": "json_parse_failed"
    }


def compare_positive_prompt_with_original(
    client: LocalHFChatClient,
    raw_prompt: str,
    positive_prompt: str,
) -> Dict[str, Any]:
    user_prompt = json.dumps({
        "original_prompt": raw_prompt,
        "generated_prompt": positive_prompt,
    }, ensure_ascii=False, indent=2)

    raw_out = client.chat_text(
        ENTITY_COMPARE_SYSTEM,
        user_prompt,
        max_retries=2,
    )

    report = extract_json_object_from_text(raw_out)

    if report.get("status") not in {"pass", "fail"}:
        return {
            "status": "checker_failed",
            "original_entities": [],
            "missing_or_changed_entities": [],
            "reason": "checker output could not be parsed"
        }

    if not isinstance(report.get("original_entities", []), list):
        report["original_entities"] = []
    if not isinstance(report.get("missing_or_changed_entities", []), list):
        report["missing_or_changed_entities"] = []

    return report


def repair_positive_prompt_by_original_comparison(
    client: LocalHFChatClient,
    raw_prompt: str,
    positive_prompt: str,
    consistency_report: Dict[str, Any],
) -> str:
    user_prompt = json.dumps({
        "original_prompt": raw_prompt,
        "generated_positive_prompt": positive_prompt,
        "consistency_report": consistency_report,
    }, ensure_ascii=False, indent=2)

    raw_out = client.chat_text(
        ENTITY_REPAIR_SYSTEM,
        user_prompt,
        max_retries=2,
    )

    obj = extract_json_object_from_text(raw_out)
    repaired = norm_text(obj.get("positive_prompt", ""))

    if not repaired:
        return positive_prompt

    return sanitize_prompt_text(repaired, max_len=500)


def final_positive_prompt_consistency_check(
    client: LocalHFChatClient,
    raw_prompt: str,
    positive_prompt: str,
    max_rounds: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    current = sanitize_prompt_text(positive_prompt, max_len=500)
    last_report: Dict[str, Any] = {}

    for _ in range(max_rounds):
        report = compare_positive_prompt_with_original(
            client=client,
            raw_prompt=raw_prompt,
            positive_prompt=current,
        )
        last_report = report

        if report.get("status") in {"pass", "checker_failed"}:
            return current, report

        missing = report.get("missing_or_changed_entities", [])
        if not missing:
            return current, report

        repaired = repair_positive_prompt_by_original_comparison(
            client=client,
            raw_prompt=raw_prompt,
            positive_prompt=current,
            consistency_report=report,
        )

        if norm_text(repaired) == norm_text(current):
            return current, report

        current = repaired

    return current, last_report


def reference_core_summary(ref_bundle: Dict[str, Any]) -> str:
    event = ref_bundle.get("current_event", {})
    pc = (event.get("physical_condition") or {})
    sg = (event.get("scene_graph") or {})
    cond = norm_text(pc.get("condition_summary", ""))
    graph = norm_text(sg.get("graph_summary", ""))
    if cond and not is_placeholder_summary(cond):
        return cond
    if graph:
        return graph
    return ref_bundle.get("raw_prompt", "")


def fallback_event_description(raw_prompt: str, ref_bundle: Dict[str, Any], prev_desc: Optional[str]) -> str:
    summary = reference_core_summary(ref_bundle)
    if prev_desc:
        if summary:
            return sanitize_prompt_text(summary)
        return sanitize_prompt_text(prev_desc)
    return sanitize_prompt_text(summary or raw_prompt)


def fallback_condense(raw_prompt: str, event_descriptions: List[str]) -> Dict[str, str]:
    cleaned = [sanitize_prompt_text(d) for d in event_descriptions if norm_text(d)]
    if not cleaned:
        positive = sanitize_prompt_text(raw_prompt, max_len=500)
    else:
        lead = ["First", "Then", "Then", "Finally"]
        parts = []
        for i, d in enumerate(cleaned):
            prefix = lead[i] if i < len(lead) else "Then"
            d2 = d[0].lower() + d[1:] if len(d) > 1 else d.lower()
            parts.append(f"{prefix}, {d2}")
        positive = sanitize_prompt_text(" ".join(parts), max_len=500)
    negative = "blurry, artifacts, duplicate objects, identity inconsistency, abrupt discontinuity, broken geometry, implausible motion"
    return {"positive_prompt": positive, "negative_prompt": negative}



def generate_one_event_description(
    client: LocalHFChatClient,
    raw_prompt: str,
    prev_desc: Optional[str],
    ref_bundle: Dict[str, Any],
) -> str:
    event_id = int(ref_bundle.get("current_event", {}).get("event_id", -1))
    if prev_desc is None:
        system_prompt = BOOTSTRAP_SYSTEM
        user_prompt = build_bootstrap_user_prompt(ref_bundle)
    else:
        system_prompt = REVISION_SYSTEM
        user_prompt = build_revision_user_prompt(prev_desc, ref_bundle)

    raw_out = client.chat_text(system_prompt, user_prompt)
    tagged = parse_tagged_output(raw_out)
    candidate = sanitize_prompt_text(tagged.get("event_description", raw_out))
    issues = validate_event_description(candidate, raw_prompt, prev_desc, event_id, ref_bundle)

    for _ in range(2):
        if not issues:
            return candidate
        repair_out = client.chat_text(
            REPAIR_EVENT_SYSTEM,
            build_repair_event_user_prompt(raw_out, raw_prompt, prev_desc or "", ref_bundle, issues),
        )
        repair_tagged = parse_tagged_output(repair_out)
        repaired = sanitize_prompt_text(repair_tagged.get("event_description", repair_out))
        new_issues = validate_event_description(repaired, raw_prompt, prev_desc, event_id, ref_bundle)
        if not new_issues:
            return repaired
        raw_out = repair_out
        candidate = repaired
        issues = new_issues

    return fallback_event_description(raw_prompt, ref_bundle, prev_desc)


def generate_event_descriptions(client: LocalHFChatClient, pfg_ctx: PFGContext, selected_ref_bundles: List[Dict[str, Any]]) -> List[str]:
    event_descriptions: List[str] = []
    prev_desc: Optional[str] = None

    for ref_bundle in selected_ref_bundles:
        desc = generate_one_event_description(client, pfg_ctx.input_text, prev_desc, ref_bundle)
        event_descriptions.append(desc)
        prev_desc = desc
    return event_descriptions


def condense_prompts(client: LocalHFChatClient, pfg_ctx: PFGContext, event_descriptions: List[str]) -> Dict[str, Any]:
    raw_out = client.chat_text(
        CONDENSE_SYSTEM,
        build_condense_user_prompt(
            raw_prompt=pfg_ctx.input_text,
            event_descriptions=event_descriptions,
            physical_law_name=pfg_ctx.physical_law_name,
        ),
    )
    tagged = parse_tagged_output(raw_out)
    positive = sanitize_prompt_text(tagged.get("positive_prompt", ""), max_len=500)
    negative = norm_text(tagged.get("negative_prompt", ""))
    issues = validate_condensed_prompts(positive, negative)

    for _ in range(2):
        if not issues:
            if not negative:
                negative = "blurry, artifacts, duplicate objects, identity inconsistency, abrupt discontinuity, broken geometry, implausible motion"

            positive, entity_report = final_positive_prompt_consistency_check(
                client=client,
                raw_prompt=pfg_ctx.input_text,
                positive_prompt=positive,
            )

            return {
                "positive_prompt": positive,
                "negative_prompt": negative,
                "entity_consistency_report": entity_report,
            }

        repair_out = client.chat_text(
            REPAIR_CONDENSE_SYSTEM,
            build_repair_condense_user_prompt(raw_out, pfg_ctx.input_text, event_descriptions, issues),
        )
        repair_tagged = parse_tagged_output(repair_out)
        positive = sanitize_prompt_text(repair_tagged.get("positive_prompt", ""), max_len=500)
        negative = norm_text(repair_tagged.get("negative_prompt", ""))
        issues = validate_condensed_prompts(positive, negative)
        raw_out = repair_out

    fallback = fallback_condense(pfg_ctx.input_text, event_descriptions)
    positive, entity_report = final_positive_prompt_consistency_check(
        client=client,
        raw_prompt=pfg_ctx.input_text,
        positive_prompt=fallback["positive_prompt"],
    )

    return {
        "positive_prompt": positive,
        "negative_prompt": fallback["negative_prompt"],
        "entity_consistency_report": entity_report,
    }


def build_iks_minimal_events(selected_ref_bundles: List[Dict[str, Any]], event_descriptions: List[str]) -> List[Dict[str, Any]]:
    if len(selected_ref_bundles) != len(event_descriptions):
        raise ValueError(f"Length mismatch: {len(selected_ref_bundles)} bundles vs {len(event_descriptions)} descriptions")

    minimal_events: List[Dict[str, Any]] = []
    for bundle, desc in zip(selected_ref_bundles, event_descriptions):
        curr = bundle.get("current_event", {}) or {}
        minimal_events.append({
            "event_id": curr.get("event_id"),
            "event_description": desc,
            "delta_from_previous": bundle.get("delta_from_previous", {}),
        })
    return minimal_events


def run_sample(client: LocalHFChatClient, pfg_item: Dict[str, Any], ppd_item: Dict[str, Any]) -> Dict[str, Any]:
    pfg_ctx = parse_pfg_item(pfg_item)
    events = [parse_event(e) for e in (ppd_item.get("events", []) or [])]
    if not events:
        raise RuntimeError(f"No events found for sample index={pfg_ctx.index}")

    all_ref_bundles = build_reference_bundles_from_events(pfg_ctx, events)
    selected_indices, selected_ref_bundles = select_best_event_path(all_ref_bundles, pfg_ctx.input_text, client=client, physical_law_name=pfg_ctx.physical_law_name, judge_mode="hybrid")
    selected_ref_bundles = rebuild_selected_reference_bundles(pfg_ctx, events, selected_indices)

    event_descriptions = generate_event_descriptions(client, pfg_ctx, selected_ref_bundles)
    prompts = condense_prompts(client, pfg_ctx, event_descriptions)
    all_event_ids = [e.event_id for e in events]
    selected_event_ids = [events[i].event_id for i in selected_indices]
    dropped_event_ids = [eid for eid in all_event_ids if eid not in selected_event_ids]

    minimal_events = build_iks_minimal_events(selected_ref_bundles, event_descriptions)

    return {
        "index": pfg_ctx.index,
        "positive_prompt": prompts["positive_prompt"],
        "negative_prompt": prompts["negative_prompt"],
        "entity_consistency_report": prompts.get("entity_consistency_report", {}),
        "events": minimal_events,
    }



def align_items_by_index(pfg_json: Dict[str, Any], ppd_json: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pfg_items = pfg_json.get("items", []) or []
    ppd_items = ppd_json.get("items", []) or []
    ppd_by_index = {item.get("index"): item for item in ppd_items}
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for pfg_item in pfg_items:
        idx = pfg_item.get("index")
        if idx not in ppd_by_index:
            raise KeyError(f"PPD item missing for index={idx}")
        pairs.append((pfg_item, ppd_by_index[idx]))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PNR for TCP using PECR outputs as reference.")
    parser.add_argument("--pfg_path", type=str, default="")
    parser.add_argument("--ppd_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    if not os.path.exists(args.pfg_path):
        raise FileNotFoundError(f"PFG path not found: {args.pfg_path}")
    if not os.path.exists(args.ppd_path):
        raise FileNotFoundError(f"PPD path not found: {args.ppd_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    pfg_json = load_json(args.pfg_path)
    ppd_json = load_json(args.ppd_path)
    pairs = align_items_by_index(pfg_json, ppd_json)
    if args.sample_index is not None:
        pairs = [pair for pair in pairs if pair[0].get("index") == args.sample_index]
        if not pairs:
            raise KeyError(f"No sample found for index={args.sample_index}")

    client = LocalHFChatClient(
        model_path=args.model_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        dtype=args.dtype,
        gpu_id=args.gpu_id,
    )

    outputs: List[Dict[str, Any]] = []
    for i, (pfg_item, ppd_item) in enumerate(pairs, start=1):
        idx = pfg_item.get("index")
        print(f"[PNR] processing {i}/{len(pairs)} | index={idx}", flush=True)
        result = run_sample(client, pfg_item, ppd_item)
        outputs.append(result)

    final = {
        "module": "PNR_minimal_for_IKS",
        "for_module": "IKS",
        "num_items": len(outputs),
        "items": outputs,
    }
    dump_json(final, args.output_path)
    print(f"[PNR] saved to: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()