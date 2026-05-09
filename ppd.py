"""
Physics Phenomena Decomposition (PPD)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


DEFAULT_SNAP_DIR = ""
MAX_NEW_TOKENS = 1400
LLM_MAX_INPUT_TOKENS = 6000
TAU_P = 0.08
MAX_RETRIES = 2
MAX_RAW_EVENTS = 8
VERBOSE = True

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def vprint(*args: Any, **kwargs: Any) -> None:
    if VERBOSE:
        print(*args, **kwargs)

def hprint(title: str) -> None:
    if VERBOSE:
        print("\n" + "=" * 20 + f" {title} " + "=" * 20)

def shorten_text(s: Any, limit: int = 180) -> str:
    s = str(s or "").replace("\n", " ").strip()
    return s if len(s) <= limit else s[:limit] + " ..."

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()

def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []

def canonical_unit(u: Any) -> str:
    return str(u or "").strip() or "dimensionless"


_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```", re.IGNORECASE)
_NON_JSON_CHARS_RE = re.compile(r"^[^\{\[]*|[^\}\]]*$", re.S)

def strip_code_fences(text: str) -> str:
    m = _CODE_FENCE_RE.search(text or "")
    return m.group(1) if m else (text or "")

def extract_balanced_json(text: str) -> str:
    s = _NON_JSON_CHARS_RE.sub("", text or "")
    starts = [i for i, ch in enumerate(s) if ch == "{"]
    for st in starts:
        depth = 0
        for i in range(st, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[st:i + 1]
                    try:
                        json.loads(cand)
                        return cand
                    except Exception:
                        continue
    return ""

def extract_json(text: str) -> Dict[str, Any]:
    text = text or ""
    try:
        return json.loads(text)
    except Exception:
        pass

    t = strip_code_fences(text)
    j = extract_balanced_json(t)
    if j:
        return json.loads(j)

    head = shorten_text(t, 300)
    raise ValueError(f"No valid JSON object found in model output. output_head={head}")


def clip_text(s: str, max_chars: int = 1200) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s[:max_chars]

def _shrink_for_llm(obj: Any, *, max_str: int, max_list: int) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str):
                out[k] = clip_text(v, max_str)
            else:
                out[k] = _shrink_for_llm(v, max_str=max_str, max_list=max_list)
        return out
    if isinstance(obj, list):
        return [_shrink_for_llm(v, max_str=max_str, max_list=max_list) for v in obj[:max_list]]
    if isinstance(obj, str):
        return clip_text(obj, max_str)
    return obj

def estimate_chat_tokens(tokenizer: Any, system_prompt: str, user_prompt: str) -> int:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(ids)
    except Exception:
        try:
            return len(tokenizer(system_prompt + "\n\n" + user_prompt)["input_ids"])
        except Exception:
            return -1

def compact_user_prompt_to_budget(tokenizer: Any, system_prompt: str, user_prompt: str, max_input_tokens: int = LLM_MAX_INPUT_TOKENS) -> str:
    n_tok = estimate_chat_tokens(tokenizer, system_prompt, user_prompt)
    if n_tok != -1 and n_tok <= max_input_tokens:
        return user_prompt

    try:
        payload = json.loads(user_prompt)
    except Exception:
        return clip_text(user_prompt, 8000)

    schedules = [
        {"max_str": 520, "max_list": 8},
        {"max_str": 320, "max_list": 6},
        {"max_str": 220, "max_list": 4},
    ]
    for sch in schedules:
        shrunk = _shrink_for_llm(payload, max_str=sch["max_str"], max_list=sch["max_list"])
        candidate = json.dumps(shrunk, ensure_ascii=False, indent=2)
        n_tok = estimate_chat_tokens(tokenizer, system_prompt, candidate)
        if n_tok == -1 or n_tok <= max_input_tokens:
            return candidate

    return json.dumps(_shrink_for_llm(payload, max_str=160, max_list=3), ensure_ascii=False, indent=2)


class LocalLLM:
    def __init__(
        self,
        model_path: str = DEFAULT_SNAP_DIR,
        max_new_tokens: int = MAX_NEW_TOKENS,
        device: str = "cuda:0",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = Mxfp4Config(dequantize=True)
        device = (device or "cuda:0").strip()
        device_map = {"": device} if device != "auto" else "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device_map,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _build_inputs(self, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raw_tokens = estimate_chat_tokens(self.tokenizer, system_prompt, user_prompt)
        packed_user_prompt = compact_user_prompt_to_budget(
            self.tokenizer, system_prompt, user_prompt, LLM_MAX_INPUT_TOKENS
        )
        packed_tokens = estimate_chat_tokens(self.tokenizer, system_prompt, packed_user_prompt)

        if VERBOSE:
            print(f"[LLM] prompt_tokens raw={raw_tokens} packed={packed_tokens} budget={LLM_MAX_INPUT_TOKENS}")
            print(f"[LLM] user_prompt_head={shorten_text(packed_user_prompt, 180)}")

        try:
            inputs = self._build_inputs(system_prompt, packed_user_prompt)
            target_device = self.model.get_input_embeddings().weight.device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as e:
            print(f"[LLM] GENERATE_EXCEPTION={type(e).__name__}: {e}")
            raise

        completion = outputs[0][inputs["input_ids"].shape[-1]:]
        content = self.tokenizer.decode(completion, skip_special_tokens=True)

        if VERBOSE:
            print(f"[LLM] raw_output_head={shorten_text(content, 260)}")

        try:
            return extract_json(content)
        except Exception:
            repair_system = "Return ONE valid JSON object only. No prose. No markdown. No explanation."
            repair_user = (
                "Convert the following raw model output into one valid JSON object only. "
                "If it cannot be repaired, return {\"status\":\"failed\",\"reason\":\"model_output_not_json\"}.\n\n"
                f"RAW OUTPUT:\n{content[:4000]}"
            )
            try:
                repair_inputs = self._build_inputs(repair_system, repair_user)
                target_device = self.model.get_input_embeddings().weight.device
                repair_inputs = {k: v.to(target_device) for k, v in repair_inputs.items()}
                with torch.no_grad():
                    repair_outputs = self.model.generate(
                        **repair_inputs,
                        max_new_tokens=min(512, self.max_new_tokens),
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                repair_completion = repair_outputs[0][repair_inputs["input_ids"].shape[-1]:]
                repair_content = self.tokenizer.decode(repair_completion, skip_special_tokens=True)
                if VERBOSE:
                    print(f"[LLM] repair_output_head={shorten_text(repair_content, 260)}")
                return extract_json(repair_content)
            except Exception:
                return {"status": "failed", "reason": "model_output_not_json"}

def validate_pfg_minimal_item(item: Dict[str, Any]) -> Tuple[bool, str]:
    need = {"index", "input_text", "physical_law", "grounded_formula", "parameter_table", "ready_for_ppd"}
    if not isinstance(item, dict):
        return False, "item is not an object"
    miss = [k for k in need if k not in item]
    if miss:
        return False, f"missing keys: {miss}"
    if not item.get("ready_for_ppd", False):
        return False, "ready_for_ppd is false"

    gf = item.get("grounded_formula", {})
    if not isinstance(gf, dict):
        return False, "grounded_formula is not an object"
    for k in ["formula_name", "equation_sympy", "variables", "variable_semantics"]:
        if k not in gf:
            return False, f"grounded_formula missing key: {k}"
    return True, "ok"


def parameter_table_to_dict(parameter_table: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in parameter_table:
        if isinstance(row, dict) and row.get("name"):
            out[str(row["name"])] = {
                "value": row.get("value"),
                "unit": row.get("unit", ""),
                "rationale": row.get("rationale", ""),
            }
    return out

def build_ppd_input(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": item["index"],
        "input_text": item["input_text"],
        "physical_law": item["physical_law"],
        "grounded_formula": item["grounded_formula"],
        "parameters": parameter_table_to_dict(item["parameter_table"]),
    }


PPD_INITIAL_CONDITION_SYSTEM = """You are a physics reasoning engine for Physics Phenomena Decomposition (PPD).

Your task is to infer the FIRST physical event condition C1.

CORE REQUIREMENTS:
1. The FIRST event must describe an OBSERVABLE physical configuration of the scene.
2. It must NOT describe laws, rules, prompts, or abstract constraints.
3. It must correspond to a real physical state of objects, positions, contact, immersion, refraction, or force configuration.
4. It must obey the grounded physical law and formula.
5. All formula variables must be present and physically plausible.
6. Do NOT output placeholders such as:
   - "string <= 40 words"
   - "..."
   - "event_description"

Return ONE valid JSON object only.

Schema:
{
  "event_id": 1,
  "physical_parameter_vector": {
    "var_name": {"value": 0.0, "unit": "string"}
  },
  "derived_quantities": {
    "name": {"value": 0.0, "unit": "string", "meaning": "string"}
  },
  "condition_summary": "concrete physical state, <= 40 words"
}
"""

PPD_NEXT_CONDITION_SYSTEM = """You are a physics reasoning engine for Physics Phenomena Decomposition (PPD).

Your task is to infer the NEXT physical event condition Ct from the previous event.

STRICT REQUIREMENTS:
1. Events MUST form a strictly forward physical progression.
2. The new event must represent a later physical state than the previous one.
3. NEVER revert to an earlier state.
4. NEVER oscillate between states.
5. A later event MUST NOT restore previous parameter values or previous spatial relations.
6. The new event must introduce a real physical change.
7. If no further distinct event exists, return terminal.
8. All outputs must obey the grounded physical law and formula.
9. All formula variables must be present and physically plausible.
10. Do NOT output placeholders such as:
   - "string <= 40 words"
   - "..."
   - "event_description"

Return ONE valid JSON object only.

Schema:
{
  "status": "ok|terminal",
  "event_id": 2,
  "physical_parameter_vector": {
    "var_name": {"value": 0.0, "unit": "string"}
  },
  "derived_quantities": {
    "name": {"value": 0.0, "unit": "string", "meaning": "string"}
  },
  "condition_summary": "concrete physical state, <= 40 words",
  "terminal_reason": "string"
}
"""

PPD_CONDITION_REPAIR_SYSTEM = """You are a repair assistant for PPD condition inference.

Repair the candidate current physical condition so that:
- it remains causally later than the previous event
- it does not violate physical continuity
- it stays constrained by the same grounded formula and parameters
- it represents a distinct later event
- it does NOT revert to any earlier state
- it does NOT repeat the same state with different wording
- it does NOT describe laws or abstract constraints instead of physical states

Return ONE valid JSON object only using the same schema as the next physical condition.
"""

PPD_INITIAL_GRAPH_SYSTEM = """You are the PPD scene-graph initialization assistant.

Infer the dynamic scene graph G1 from:
- the original text
- the grounded formula
- the current physical condition

GRAPH REQUIREMENTS:
1. Nodes must correspond to tangible physical entities:
   objects, materials, forces, media, fields, containers, surfaces.
2. INVALID nodes include:
   - verbs (e.g., "poured", "inserted")
   - adjectives (e.g., "crystal-clear")
   - abstract words
3. Use physically meaningful relations such as:
   in, on, enters, exits, displaces, refracts_into, contacts, immersed_in
4. The graph must describe the current event only.

Return ONE valid JSON object only.

Schema:
{
  "event_id": 1,
  "nodes": [
    {"id": "string", "type": "string", "attributes": {"key": "value"}}
  ],
  "edges": [
    {"subject": "string", "relation": "string", "object": "string"}
  ],
  "graph_summary": "concrete graph summary, <= 35 words"
}
"""

PPD_NEXT_GRAPH_SYSTEM = """You are the PPD scene-graph update assistant.

Infer Gt from:
- the previous graph
- the current physical condition

GRAPH REQUIREMENTS:
1. Update the previous graph minimally.
2. Nodes must correspond to tangible physical entities:
   objects, materials, forces, media, fields, containers, surfaces.
3. INVALID nodes include:
   - verbs (e.g., "poured", "inserted")
   - adjectives (e.g., "crystal-clear")
   - abstract words
4. Use physically meaningful relations such as:
   in, on, enters, exits, displaces, refracts_into, contacts, immersed_in
5. The graph must describe the current event only.

Return ONE valid JSON object only.

Schema:
{
  "event_id": 2,
  "nodes": [
    {"id": "string", "type": "string", "attributes": {"key": "value"}}
  ],
  "edges": [
    {"subject": "string", "relation": "string", "object": "string"}
  ],
  "graph_summary": "concrete graph summary, <= 35 words"
}
"""

PPD_EVENT_DESCRIPTION_SYSTEM = """You are the PPD event description assistant.

Generate one concise description for the CURRENT event only.

REQUIREMENTS:
1. The description must state a concrete physical state or change.
2. It must match the current physical_condition and scene_graph.
3. It must mention every required entity from the original prompt.
4. Do not remove, rename, merge, or replace any required entity.
5. It must NOT describe laws, prompts, schemas, or placeholders.
6. It must be <= 45 words.

Return ONE valid JSON object only.

Schema:
{
  "event_id": 1,
  "event_description": "concrete physical description, <= 40 words"
}
"""


def compact_pfg_for_prompt(ppd_input: Dict[str, Any]) -> Dict[str, Any]:
    gf = ppd_input["grounded_formula"]
    return {
        "index": ppd_input["index"],
        "input_text": ppd_input["input_text"],
        "physical_law": ppd_input["physical_law"],
        "grounded_formula": {
            "formula_name": gf.get("formula_name", ""),
            "equation_latex": gf.get("equation_latex", ""),
            "equation_sympy": gf.get("equation_sympy", ""),
            "variables": gf.get("variables", []),
            "variable_semantics": gf.get("variable_semantics", {}),
            "formula_role_in_scene": gf.get("formula_role_in_scene", ""),
        },
        "parameters": ppd_input["parameters"],
    }

def build_initial_condition_prompt(ppd_input: Dict[str, Any]) -> str:
    return json.dumps({
        "task": "infer C1 for the earliest observable physical event",
        "pfg_result": compact_pfg_for_prompt(ppd_input),
        "strict_requirements": [
            "The first event must be a real observable physical state.",
            "It must not describe laws, rules, or abstract constraints.",
            "It must obey the grounded formula.",
            "All formula variables must be present and physically plausible."
        ],
    }, ensure_ascii=False, indent=2)

def build_next_condition_prompt(ppd_input: Dict[str, Any], prev_event: Dict[str, Any], event_id: int) -> str:
    return json.dumps({
        "task": f"infer C{event_id} from the previous event, or return terminal if no further distinct event exists",
        "pfg_result": compact_pfg_for_prompt(ppd_input),
        "previous_event": {
            "event_id": prev_event["event_id"],
            "physical_condition": prev_event["physical_condition"],
            "scene_graph": prev_event["scene_graph"],
            "event_description": prev_event["event_description"],
        },
        "strict_requirements": [
            "The sequence must be temporally monotonic.",
            "The new event must be strictly later than the previous event.",
            "Do not revert to any previous state.",
            "Do not restore previous parameter values or previous spatial relations.",
            "The new event must introduce a real physical change.",
            "If no further distinct event exists, return terminal."
        ],
        "physical_reasoning_guide": [
            "Think in terms of force balance changes.",
            "Think in terms of motion initiation or continuation.",
            "Think in terms of interaction changes between objects.",
            "Think in terms of medium transitions."
        ],
    }, ensure_ascii=False, indent=2)

def build_condition_repair_prompt(ppd_input: Dict[str, Any], prev_event: Dict[str, Any], cand: Dict[str, Any], reason: str, event_id: int) -> str:
    return json.dumps({
        "task": f"repair C{event_id} inferred from the previous event",
        "pfg_result": compact_pfg_for_prompt(ppd_input),
        "previous_event": {
            "event_id": prev_event["event_id"],
            "physical_condition": prev_event["physical_condition"],
            "scene_graph": prev_event["scene_graph"],
            "event_description": prev_event["event_description"],
        },
        "candidate_condition": cand,
        "failure_reason": reason,
        "instruction": "Repair the current event condition so it becomes physically continuous, formula-consistent, and distinct from the previous event.",
    }, ensure_ascii=False, indent=2)

def build_initial_graph_prompt(ppd_input: Dict[str, Any], c1: Dict[str, Any]) -> str:
    return json.dumps({
        "task": "infer G1 for event 1",
        "pfg_result": compact_pfg_for_prompt(ppd_input),
        "physical_condition": c1,
    }, ensure_ascii=False, indent=2)

def build_next_graph_prompt(ppd_input: Dict[str, Any], prev_event: Dict[str, Any], curr_condition: Dict[str, Any], event_id: int) -> str:
    return json.dumps({
        "task": f"infer G{event_id} from previous graph and current condition",
        "pfg_result": compact_pfg_for_prompt(ppd_input),
        "previous_event": {
            "event_id": prev_event["event_id"],
            "scene_graph": prev_event["scene_graph"],
        },
        "current_physical_condition": curr_condition,
    }, ensure_ascii=False, indent=2)

def build_event_description_prompt(ppd_input: Dict[str, Any], event: Dict[str, Any]) -> str:
    return json.dumps({
        "task": f"generate event description for event {event['event_id']}",
        "input_text": ppd_input["input_text"],
        "required_entities": build_entity_pool(ppd_input),
        "hard_entity_rule": (
            "The event_description must explicitly mention every required entity. "
            "Do not omit, rename, merge, or replace any required entity."
        ),
        "physical_condition": event["physical_condition"],
        "scene_graph": event["scene_graph"],
    }, ensure_ascii=False, indent=2)


def validate_condition_schema(obj: Dict[str, Any], event_id: int, required_formula_vars: Sequence[str], allow_terminal: bool = False) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "condition is not an object"

    if allow_terminal and obj.get("status") == "terminal":
        return True, "terminal"

    need = {"event_id", "physical_parameter_vector", "derived_quantities", "condition_summary"}
    miss = [k for k in need if k not in obj]
    if miss:
        return False, f"condition missing keys: {miss}"

    if obj.get("event_id") != event_id:
        return False, f"event_id must be {event_id}"

    ppv = obj.get("physical_parameter_vector")
    if not isinstance(ppv, dict):
        return False, "physical_parameter_vector is not a dict"

    for v in required_formula_vars:
        if v not in ppv:
            return False, f"physical_parameter_vector missing formula variable: {v}"
        row = ppv[v]
        if not isinstance(row, dict):
            return False, f"parameter row for {v} is not an object"
        if "value" not in row or "unit" not in row:
            return False, f"parameter row for {v} missing value/unit"
        try:
            float(row["value"])
        except Exception:
            return False, f"parameter value for {v} is not numeric"

    if not isinstance(obj.get("derived_quantities"), dict):
        return False, "derived_quantities is not a dict"

    if not str(obj.get("condition_summary", "")).strip():
        return False, "condition_summary is empty"

    return True, "ok"

def validate_graph_schema(obj: Dict[str, Any], event_id: int) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "graph is not an object"

    need = {"event_id", "nodes", "edges", "graph_summary"}
    miss = [k for k in need if k not in obj]
    if miss:
        return False, f"graph missing keys: {miss}"

    if obj.get("event_id") != event_id:
        return False, f"event_id must be {event_id}"

    if not isinstance(obj.get("nodes"), list):
        return False, "nodes is not a list"
    if not isinstance(obj.get("edges"), list):
        return False, "edges is not a list"

    for n in obj["nodes"]:
        if not isinstance(n, dict):
            return False, "one node is not an object"
        for k in ["id", "type", "attributes"]:
            if k not in n:
                return False, f"node missing key: {k}"

    for e in obj["edges"]:
        if not isinstance(e, dict):
            return False, "one edge is not an object"
        for k in ["subject", "relation", "object"]:
            if k not in e:
                return False, f"edge missing key: {k}"

    return True, "ok"

def validate_event_desc_schema(obj: Dict[str, Any], event_id: int) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "event description is not an object"
    if obj.get("event_id") != event_id:
        return False, f"event_id must be {event_id}"
    if not str(obj.get("event_description", "")).strip():
        return False, "event_description is empty"
    return True, "ok"


def condition_vector(cond: Dict[str, Any]) -> Dict[str, Tuple[float, str]]:
    out: Dict[str, Tuple[float, str]] = {}
    for k, row in (cond.get("physical_parameter_vector", {}) or {}).items():
        try:
            out[k] = (float(row.get("value", 0.0)), str(row.get("unit", "")))
        except Exception:
            continue
    return out

def normalized_parameter_delta(prev_cond: Dict[str, Any], curr_cond: Dict[str, Any]) -> float:
    prev = condition_vector(prev_cond)
    curr = condition_vector(curr_cond)
    keys = sorted(set(prev.keys()) & set(curr.keys()))
    if not keys:
        return 0.0
    vals = []
    for k in keys:
        pv, _ = prev[k]
        cv, _ = curr[k]
        denom = abs(pv) + 1e-6
        vals.append(abs(cv - pv) / denom)
    return float(sum(vals) / max(1, len(vals)))

def scene_graph_change_score(prev_graph: Dict[str, Any], curr_graph: Dict[str, Any]) -> float:
    prev_nodes = {(n["id"], n["type"], json.dumps(n.get("attributes", {}), sort_keys=True, ensure_ascii=False))
                  for n in prev_graph.get("nodes", []) if isinstance(n, dict)}
    curr_nodes = {(n["id"], n["type"], json.dumps(n.get("attributes", {}), sort_keys=True, ensure_ascii=False))
                  for n in curr_graph.get("nodes", []) if isinstance(n, dict)}
    prev_edges = {(e["subject"], e["relation"], e["object"])
                  for e in prev_graph.get("edges", []) if isinstance(e, dict)}
    curr_edges = {(e["subject"], e["relation"], e["object"])
                  for e in curr_graph.get("edges", []) if isinstance(e, dict)}

    denom = max(1, len(prev_nodes | curr_nodes | prev_edges | curr_edges))
    diff = len(prev_nodes ^ curr_nodes) + len(prev_edges ^ curr_edges)
    return float(diff) / float(denom)

def relation_change_score(prev_graph: Dict[str, Any], curr_graph: Dict[str, Any]) -> float:
    prev_edges = {
        (e["subject"], e["relation"], e["object"])
        for e in prev_graph.get("edges", [])
        if isinstance(e, dict) and {"subject", "relation", "object"} <= set(e.keys())
    }
    curr_edges = {
        (e["subject"], e["relation"], e["object"])
        for e in curr_graph.get("edges", [])
        if isinstance(e, dict) and {"subject", "relation", "object"} <= set(e.keys())
    }
    denom = max(1, len(prev_edges | curr_edges))
    diff = len(prev_edges ^ curr_edges)
    return float(diff) / float(denom)

def summary_delta(prev_summary: str, curr_summary: str) -> float:
    p = set(re.findall(r"[a-zA-Z]+", normalize_text(prev_summary)))
    c = set(re.findall(r"[a-zA-Z]+", normalize_text(curr_summary)))
    if not (p or c):
        return 0.0
    inter = len(p & c)
    union = len(p | c)
    return 1.0 - inter / max(1, union)

def has_abrupt_jump(prev_cond: Dict[str, Any], curr_cond: Dict[str, Any]) -> Tuple[bool, str]:
    prev = condition_vector(prev_cond)
    curr = condition_vector(curr_cond)
    keys = sorted(set(prev.keys()) & set(curr.keys()))
    for k in keys:
        pv, pu = prev[k]
        cv, cu = curr[k]
        if pu == cu:
            if abs(pv) > 1e-6 and abs(cv) > 1e-6:
                ratio = max(abs(cv), abs(pv)) / max(1e-6, min(abs(cv), abs(pv)))
                if ratio > 20.0:
                    return True, f"abrupt numeric jump detected on {k}"
    return False, "ok"

def validate_continuity(prev_event: Dict[str, Any], cand_cond: Dict[str, Any]) -> Tuple[bool, str]:
    abrupt, reason = has_abrupt_jump(prev_event["physical_condition"], cand_cond)
    if abrupt:
        return False, reason
    return True, "ok"

def boundary_is_meaningful(
    prev_event: Dict[str, Any],
    cand_cond: Dict[str, Any],
    cand_graph: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    param_delta = normalized_parameter_delta(prev_event["physical_condition"], cand_cond)
    rel_delta = relation_change_score(prev_event["scene_graph"], cand_graph)
    ok = (param_delta > TAU_P) or (rel_delta > 0.0)
    return ok, {
        "param_delta": round(param_delta, 6),
        "relation_delta": round(rel_delta, 6),
        "tau_p": TAU_P,
        "rule": "param_change_or_relation_change_only",
    }

def event_distance(e1: Dict[str, Any], e2: Dict[str, Any]) -> float:
    param_delta = normalized_parameter_delta(e1["physical_condition"], e2["physical_condition"])
    graph_delta = scene_graph_change_score(e1["scene_graph"], e2["scene_graph"])
    sem_delta = summary_delta(e1.get("event_description", ""), e2.get("event_description", ""))
    return 0.55 * param_delta + 0.25 * graph_delta + 0.20 * sem_delta


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "with", "into", "from", "for",
    "and", "or", "as", "that", "this", "it", "its", "their", "his", "her",
    "gently", "slowly", "filled", "clear", "crystal", "plastic"
}

def default_formula_params(ppd_input: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    gf = safe_dict(ppd_input.get("grounded_formula"))
    params = safe_dict(ppd_input.get("parameters"))
    for var in safe_list(gf.get("variables")):
        p = safe_dict(params.get(var))
        out[var] = {
            "value": to_float(p.get("value", 0.0), 0.0),
            "unit": canonical_unit(p.get("unit", "dimensionless")),
        }
    return out

def fallback_condition_summary(ppd_input: Dict[str, Any], event_id: int, prev_event: Optional[Dict[str, Any]] = None) -> str:
    formula_name = safe_dict(ppd_input.get("grounded_formula")).get("formula_name", "the grounded formula")
    law_name = safe_dict(ppd_input.get("physical_law")).get("name", "the physical law")
    if prev_event is None:
        return f"The earliest physical stage is constrained by {law_name} and {formula_name}."
    return f"A later physical stage remains constrained by {law_name} and {formula_name}."

def build_fallback_condition(ppd_input: Dict[str, Any], event_id: int, prev_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "physical_parameter_vector": default_formula_params(ppd_input),
        "derived_quantities": {},
        "condition_summary": fallback_condition_summary(ppd_input, event_id, prev_event),
    }

def normalize_condition_object(
    ppd_input: Dict[str, Any],
    obj: Dict[str, Any],
    event_id: int,
    prev_event: Optional[Dict[str, Any]] = None,
    allow_terminal: bool = False,
) -> Dict[str, Any]:
    obj = safe_dict(obj)

    if allow_terminal and obj.get("status") == "terminal":
        return {
            "status": "terminal",
            "event_id": event_id,
            "terminal_reason": str(obj.get("terminal_reason", "no further distinct event")).strip() or "no further distinct event",
        }

    if obj.get("status") == "failed" or obj.get("reason") == "model_output_not_json":
        return build_fallback_condition(ppd_input, event_id, prev_event)

    defaults = default_formula_params(ppd_input)
    ppv_raw = safe_dict(obj.get("physical_parameter_vector"))
    ppv_norm: Dict[str, Dict[str, Any]] = {}

    for var, d in defaults.items():
        row = safe_dict(ppv_raw.get(var))
        value = row.get("value", d["value"])
        unit = row.get("unit", d["unit"])
        try:
            value = float(value)
        except Exception:
            value = d["value"]
        unit = canonical_unit(unit or d["unit"])
        ppv_norm[var] = {"value": value, "unit": unit}

    derived = obj.get("derived_quantities")
    if not isinstance(derived, dict):
        derived = {}

    summary = str(obj.get("condition_summary", "")).strip()
    if not summary:
        summary = fallback_condition_summary(ppd_input, event_id, prev_event)

    return {
        "event_id": event_id,
        "physical_parameter_vector": ppv_norm,
        "derived_quantities": derived,
        "condition_summary": summary,
    }

def extract_object_candidates(text: str, max_objects: int = 5) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text or "")
    out: List[str] = []
    for w in words:
        lw = w.lower()
        if lw in STOPWORDS:
            continue
        if lw not in out:
            out.append(lw)
        if len(out) >= max_objects:
            break
    if not out:
        out = ["scene"]
    return out

def extract_entity_pool(text: str, max_entities: int = 8) -> List[str]:
    text = (text or "").strip().lower()

                           
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    entities: List[str] = []

                                                         
                                                                    
    chunk_patterns = [
        r"\b(?:the|a|an)\s+([a-z][a-z-]*)\b",
        r"\b(?:in|into|on|onto|of|inside)\s+(?:the|a|an)\s+([a-z][a-z-]*)\b",
    ]
    for pat in chunk_patterns:
        for m in re.finditer(pat, text):
            cand = m.group(1).strip()
            if not cand or cand in STOPWORDS:
                continue
            if cand not in entities:
                entities.append(cand)
            if len(entities) >= max_entities:
                return entities

                                                                                        
    banned_lexemes = {
        "poured", "placed", "inserted", "filled", "floating", "sinking",
        "clear", "crystal", "slowly", "gently"
    }
    for w in re.findall(r"[a-z][a-z-]*", text):
        if w in STOPWORDS or w in banned_lexemes:
            continue
        if w.endswith("ed") or w.endswith("ing"):
            continue
        if w not in entities:
            entities.append(w)
        if len(entities) >= max_entities:
            break

    return entities

def build_entity_pool(ppd_input: Dict[str, Any]) -> List[str]:
    pool = extract_entity_pool(ppd_input.get("input_text", ""))
    out: List[str] = []
    for x in pool:
        if x and x not in out:
            out.append(x)
    return out

def build_fallback_graph(ppd_input: Dict[str, Any], event_id: int, cond: Dict[str, Any], prev_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if prev_event is not None:
        g = copy.deepcopy(prev_event["scene_graph"])
        g["event_id"] = event_id
        g["graph_summary"] = shorten_text(cond.get("condition_summary", ""), 120)
                                                           
        g["nodes"] = [n for n in safe_list(g.get("nodes")) if str(safe_dict(n).get("id", "")).strip().lower() != "scene"]
        g["edges"] = [
            e for e in safe_list(g.get("edges"))
            if str(safe_dict(e).get("subject", "")).strip().lower() != "scene"
            and str(safe_dict(e).get("object", "")).strip().lower() != "scene"
        ]
        return g

    entity_pool = build_entity_pool(ppd_input)
    nodes = [{"id": ent, "type": "object", "attributes": {}} for ent in entity_pool]
    edges = []

    return {
        "event_id": event_id,
        "nodes": nodes,
        "edges": edges,
        "graph_summary": shorten_text(cond.get("condition_summary", ""), 120),
    }

def normalize_graph_object(
    ppd_input: Dict[str, Any],
    obj: Dict[str, Any],
    event_id: int,
    cond: Dict[str, Any],
    prev_event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    obj = safe_dict(obj)

    if obj.get("status") == "failed" or obj.get("reason") == "model_output_not_json":
        return build_fallback_graph(ppd_input, event_id, cond, prev_event)

    entity_pool = set(build_entity_pool(ppd_input))
    nodes = safe_list(obj.get("nodes"))
    edges = safe_list(obj.get("edges"))
    graph_summary = str(obj.get("graph_summary", "")).strip() or shorten_text(cond.get("condition_summary", ""), 120)

    norm_nodes = []
    allowed_ids = set()

    for n in nodes:
        n = safe_dict(n)
        nid = str(n.get("id", "")).strip().lower()
        if not nid or nid == "scene":
            continue
        if nid not in entity_pool:
            continue
        attrs = n.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        norm_nodes.append({"id": nid, "type": "object", "attributes": attrs})
        allowed_ids.add(nid)

    norm_edges = []
    for e in edges:
        e = safe_dict(e)
        s = str(e.get("subject", "")).strip().lower()
        r = str(e.get("relation", "")).strip()
        o = str(e.get("object", "")).strip().lower()
        if not (s and r and o):
            continue
        if s == "scene" or o == "scene":
            continue
        if s in allowed_ids and o in allowed_ids:
            norm_edges.append({"subject": s, "relation": r, "object": o})

    for ent in sorted(entity_pool):
        if ent not in allowed_ids:
            norm_nodes.append({
                "id": ent,
                "type": "object",
                "attributes": {}
            })
            allowed_ids.add(ent)

    if len(norm_nodes) == 0:
        return build_fallback_graph(ppd_input, event_id, cond, prev_event)

    return {
        "event_id": event_id,
        "nodes": norm_nodes,
        "edges": norm_edges,
        "graph_summary": graph_summary,
    }

def _is_bad_event_description(desc: str) -> bool:
    """
    Hard-rule post-cleaning is allowed ONLY for event descriptions.
    """
    s = str(desc or "").strip()
    if not s:
        return True

    low = normalize_text(s)

    bad_exact = {
        "string <= 40 words",
        "...",
        "…",
        "event_description",
        "description",
        "event description",
        "n/a",
        "none",
        "null",
        "todo",
    }
    if low in bad_exact:
        return True

    bad_substrings = [
        "string <=",
        "event_description",
        "schema:",
        '"event_description"',
    ]
    if any(b in low for b in bad_substrings):
        return True

    if not re.search(r"[a-zA-Z0-9]", s):
        return True

    return False

def normalize_event_description_object(
    obj: Dict[str, Any],
    event_id: int,
    cond: Dict[str, Any],
    graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    obj = safe_dict(obj)
    raw_desc = str(obj.get("event_description", "")).strip()

    if not _is_bad_event_description(raw_desc):
        desc = raw_desc
    else:
        graph_summary = ""
        if isinstance(graph, dict):
            graph_summary = str(graph.get("graph_summary", "")).strip()

        cond_summary = str(cond.get("condition_summary", "")).strip()

        if graph_summary and not _is_bad_event_description(graph_summary):
            desc = graph_summary
        elif cond_summary and not _is_bad_event_description(cond_summary):
            desc = cond_summary
        else:
            desc = f"Event {event_id}"

    return {"event_id": event_id, "event_description": desc}


def infer_initial_condition(llm: LocalLLM, ppd_input: Dict[str, Any]) -> Dict[str, Any]:
    return llm.generate_json(PPD_INITIAL_CONDITION_SYSTEM, build_initial_condition_prompt(ppd_input))

def infer_next_condition(llm: LocalLLM, ppd_input: Dict[str, Any], prev_event: Dict[str, Any], event_id: int) -> Dict[str, Any]:
    return llm.generate_json(PPD_NEXT_CONDITION_SYSTEM, build_next_condition_prompt(ppd_input, prev_event, event_id))

def repair_condition(llm: LocalLLM, ppd_input: Dict[str, Any], prev_event: Dict[str, Any], cand: Dict[str, Any], reason: str, event_id: int) -> Dict[str, Any]:
    return llm.generate_json(PPD_CONDITION_REPAIR_SYSTEM, build_condition_repair_prompt(ppd_input, prev_event, cand, reason, event_id))

def infer_initial_graph(llm: LocalLLM, ppd_input: Dict[str, Any], c1: Dict[str, Any]) -> Dict[str, Any]:
    return llm.generate_json(PPD_INITIAL_GRAPH_SYSTEM, build_initial_graph_prompt(ppd_input, c1))

def infer_next_graph(llm: LocalLLM, ppd_input: Dict[str, Any], prev_event: Dict[str, Any], curr_condition: Dict[str, Any], event_id: int) -> Dict[str, Any]:
    return llm.generate_json(PPD_NEXT_GRAPH_SYSTEM, build_next_graph_prompt(ppd_input, prev_event, curr_condition, event_id))

def infer_event_description(llm: LocalLLM, ppd_input: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    return llm.generate_json(PPD_EVENT_DESCRIPTION_SYSTEM, build_event_description_prompt(ppd_input, event))


def build_event_record(
    event_id: int,
    cond: Dict[str, Any],
    graph: Dict[str, Any],
    desc: str,
    boundary_info: Optional[Dict[str, Any]] = None,
    continuity_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "physical_condition": cond,
        "scene_graph": graph,
        "event_description": desc,
        "boundary_check": boundary_info or {},
        "continuity_check": continuity_info or {},
    }

def decompose_one_item_raw(ppd_input: Dict[str, Any], llm: LocalLLM) -> Dict[str, Any]:
    gf = ppd_input["grounded_formula"]
    required_formula_vars = list(gf.get("variables", []))

    hprint(f"PPD sample {ppd_input['index']}")
    vprint(f"[PPD] text={ppd_input['input_text']}")
    vprint(f"[PPD] formula={gf.get('formula_name', '')} | vars={required_formula_vars}")

    events: List[Dict[str, Any]] = []

                                   
    c1_raw = infer_initial_condition(llm, ppd_input)
    c1 = normalize_condition_object(ppd_input, c1_raw, 1, prev_event=None, allow_terminal=False)
    ok, reason = validate_condition_schema(c1, 1, required_formula_vars)
    if not ok:
        raise RuntimeError(f"Event 1 condition invalid: {reason}")

    g1_raw = infer_initial_graph(llm, ppd_input, c1)
    g1 = normalize_graph_object(ppd_input, g1_raw, 1, c1, prev_event=None)
    ok, reason = validate_graph_schema(g1, 1)
    if not ok:
        raise RuntimeError(f"Event 1 scene graph invalid: {reason}")

    d1_raw = infer_event_description(llm, ppd_input, {"event_id": 1, "physical_condition": c1, "scene_graph": g1})
    d1 = normalize_event_description_object(d1_raw, 1, c1, g1)
    ok, reason = validate_event_desc_schema(d1, 1)
    desc1 = d1["event_description"] if ok else c1.get("condition_summary", "")

    events.append(build_event_record(1, c1, g1, desc1))

                                      
    next_event_id = 2
    while next_event_id <= MAX_RAW_EVENTS:
        prev_event = events[-1]

        cand_raw = infer_next_condition(llm, ppd_input, prev_event, next_event_id)
        cand = normalize_condition_object(ppd_input, cand_raw, next_event_id, prev_event=prev_event, allow_terminal=True)

        ok, reason = validate_condition_schema(cand, next_event_id, required_formula_vars, allow_terminal=True)
        if not ok:
            repaired_raw = repair_condition(llm, ppd_input, prev_event, cand, f"schema_invalid: {reason}", next_event_id)
            cand = normalize_condition_object(ppd_input, repaired_raw, next_event_id, prev_event=prev_event, allow_terminal=True)
            ok, reason = validate_condition_schema(cand, next_event_id, required_formula_vars, allow_terminal=True)
            if not ok:
                vprint(f"[PPD] stop expansion at event {next_event_id} after invalid repair: {reason}")
                break

        if cand.get("status") == "terminal":
            break

                                                                                                     
        graph_raw = infer_next_graph(llm, ppd_input, prev_event, cand, next_event_id)
        graph = normalize_graph_object(ppd_input, graph_raw, next_event_id, cand, prev_event=prev_event)
        ok, reason = validate_graph_schema(graph, next_event_id)
        if not ok:
            raise RuntimeError(f"Event {next_event_id} scene graph invalid: {reason}")

        boundary_ok, boundary_info = boundary_is_meaningful(prev_event, cand, graph)
        continuity_ok, continuity_reason = validate_continuity(prev_event, cand)

        retries = 0
        while (not boundary_ok or not continuity_ok) and retries < MAX_RETRIES:
            fail_reason = (
                f"boundary_ok={boundary_ok}, boundary_info={boundary_info}, "
                f"continuity_ok={continuity_ok}, continuity_reason={continuity_reason}"
            )
            vprint(f"[PPD] repairing event {next_event_id} condition -> {fail_reason}")

            repaired_raw = repair_condition(llm, ppd_input, prev_event, cand, fail_reason, next_event_id)
            cand = normalize_condition_object(ppd_input, repaired_raw, next_event_id, prev_event=prev_event, allow_terminal=True)
            ok, reason = validate_condition_schema(cand, next_event_id, required_formula_vars, allow_terminal=True)
            if not ok:
                retries += 1
                continue

            if cand.get("status") == "terminal":
                break

            graph_raw = infer_next_graph(llm, ppd_input, prev_event, cand, next_event_id)
            graph = normalize_graph_object(ppd_input, graph_raw, next_event_id, cand, prev_event=prev_event)
            ok, reason = validate_graph_schema(graph, next_event_id)
            if not ok:
                raise RuntimeError(f"Event {next_event_id} scene graph invalid: {reason}")

            boundary_ok, boundary_info = boundary_is_meaningful(prev_event, cand, graph)
            continuity_ok, continuity_reason = validate_continuity(prev_event, cand)
            retries += 1

        if cand.get("status") == "terminal":
            break

        if not boundary_ok:
            vprint(f"[PPD] stop expansion at event {next_event_id} because no meaningful new boundary")
            break

        continuity_info = {"ok": continuity_ok, "reason": continuity_reason}

        desc_raw = infer_event_description(llm, ppd_input, {"event_id": next_event_id, "physical_condition": cand, "scene_graph": graph})
        desc_obj = normalize_event_description_object(desc_raw, next_event_id, cand, graph)
        ok, reason = validate_event_desc_schema(desc_obj, next_event_id)
        desc = desc_obj["event_description"] if ok else cand.get("condition_summary", "")

        events.append(build_event_record(next_event_id, cand, graph, desc, boundary_info, continuity_info))
        next_event_id += 1

    return {
        "index": ppd_input["index"],
        "input_text": ppd_input["input_text"],
        "physical_law": ppd_input["physical_law"],
        "grounded_formula": ppd_input["grounded_formula"],
        "raw_event_number": len(events),
        "raw_events": events,
    }

def _representative_merged_event(
    left: Dict[str, Any],
    right: Dict[str, Any],
    preserve_first: bool = False,
    preserve_last: bool = False,
) -> Dict[str, Any]:
    if preserve_first:
        rep = copy.deepcopy(left)
    elif preserve_last:
        rep = copy.deepcopy(right)
    else:
        rep = copy.deepcopy(right)

    rep.setdefault("merge_info", {})
    rep["merge_info"] = {
        "merged_from_event_ids": [left["event_id"], right["event_id"]],
        "left_event_description": left.get("event_description", ""),
        "right_event_description": right.get("event_description", ""),
    }
    return rep

def merge_closest_adjacent_pair(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(events) <= 1:
        return events

    best_idx = 0
    best_dist = float("inf")
    for i in range(len(events) - 1):
        d = event_distance(events[i], events[i + 1])
        if d < best_dist:
            best_dist = d
            best_idx = i

    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(events):
        if i == best_idx:
            preserve_first = (i == 0)
            preserve_last = (i + 1 == len(events) - 1)
            out.append(_representative_merged_event(events[i], events[i + 1], preserve_first=preserve_first, preserve_last=preserve_last))
            i += 2
        else:
            out.append(copy.deepcopy(events[i]))
            i += 1

    return out

def quantize_events_to_124(events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_n = len(events)
    info: Dict[str, Any] = {
        "raw_event_number": raw_n,
        "target_event_number": raw_n,
        "rule": "",
        "merge_steps": [],
    }

    if raw_n <= 1:
        info["target_event_number"] = 1
        info["rule"] = "1->1"
        return events[:1], info

    if raw_n == 2:
        info["target_event_number"] = 2
        info["rule"] = "2->2"
        return copy.deepcopy(events), info

    work = copy.deepcopy(events)
    if raw_n == 3:
        info["target_event_number"] = 2
        info["rule"] = "3->2"
        before = len(work)
        work = merge_closest_adjacent_pair(work)
        info["merge_steps"].append({"before": before, "after": len(work)})
    else:
        info["target_event_number"] = 4
        info["rule"] = "4+->4"
        while len(work) > 4:
            before = len(work)
            work = merge_closest_adjacent_pair(work)
            info["merge_steps"].append({"before": before, "after": len(work)})

    for i, e in enumerate(work, start=1):
        e["event_id"] = i
        e["physical_condition"]["event_id"] = i
        e["scene_graph"]["event_id"] = i

    return work, info


def _self_check_entity_pool_filter() -> None:
    ppd_input = {
        "input_text": "The oil in the beaker was poured into the water in the beaker.",
        "physical_law": {"name": "Archimedes' principle"},
        "grounded_formula": {"variables": ["f_a", "rho", "g", "v"]},
        "parameters": {},
    }
    obj = {
        "event_id": 1,
        "nodes": [
            {"id": "scene", "type": "scene", "attributes": {}},
            {"id": "oil", "type": "object", "attributes": {}},
            {"id": "beaker", "type": "object", "attributes": {}},
            {"id": "water", "type": "object", "attributes": {}},
            {"id": "poured", "type": "object", "attributes": {}},
        ],
        "edges": [
            {"subject": "oil", "relation": "in", "object": "beaker"},
            {"subject": "oil", "relation": "poured_into", "object": "water"},
            {"subject": "scene", "relation": "contains", "object": "oil"},
            {"subject": "scene", "relation": "contains", "object": "poured"},
        ],
        "graph_summary": "test",
    }
    out = normalize_graph_object(ppd_input, obj, 1, {"condition_summary": "test"}, None)
    kept = {n["id"] for n in out["nodes"]}
    assert "scene" not in kept
    assert "oil" in kept and "beaker" in kept and "water" in kept
    assert "poured" not in kept
    assert all(e["subject"] != "scene" and e["object"] != "scene" for e in out["edges"])

def build_tcp_minimal_event(event: Dict[str, Any]) -> Dict[str, Any]:
    pc = safe_dict(event.get("physical_condition"))
    sg = safe_dict(event.get("scene_graph"))
    return {
        "event_id": event.get("event_id"),
        "physical_condition": {
            "physical_parameter_vector": safe_dict(pc.get("physical_parameter_vector")),
            "derived_quantities": safe_dict(pc.get("derived_quantities")),
            "condition_summary": pc.get("condition_summary", ""),
        },
        "scene_graph": {
            "nodes": safe_list(sg.get("nodes")),
            "edges": safe_list(sg.get("edges")),
            "graph_summary": sg.get("graph_summary", ""),
        },
    }

def build_tcp_minimal_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": item.get("index"),
        "input_text": item.get("input_text", ""),
        "events": [build_tcp_minimal_event(ev) for ev in safe_list(item.get("events"))],
    }


def run_ppd(pfg_json_path: str, out_file: str, model_path: str, device: str) -> Dict[str, Any]:
    pfg = load_json(pfg_json_path)
    if not isinstance(pfg, dict) or not isinstance(pfg.get("items"), list):
        raise ValueError("PFG json must be an object with key 'items'")

    llm = LocalLLM(model_path=model_path, device=device)
    outputs = []

    for item in pfg["items"]:
        ok, reason = validate_pfg_minimal_item(item)
        if not ok:
            outputs.append({
                "index": item.get("index", -1),
                "status": "failed",
                "reason": reason,
            })
            continue

        ppd_input = build_ppd_input(item)
        try:
            raw_out = decompose_one_item_raw(ppd_input, llm)
            quantized_events, quant_info = quantize_events_to_124(raw_out["raw_events"])
            outputs.append({
                "index": raw_out["index"],
                "input_text": raw_out["input_text"],
                "physical_law": raw_out["physical_law"],
                "grounded_formula": raw_out["grounded_formula"],
                "raw_event_number": raw_out["raw_event_number"],
                "event_number": len(quantized_events),
                "event_quantization": quant_info,
                "events": quantized_events,
                "status": "ok",
            })
        except Exception as e:
            outputs.append({
                "index": item.get("index", -1),
                "status": "failed",
                "reason": str(e),
            })

    ok_items = [item for item in outputs if item.get("status") == "ok"]
    final = {
        "module": "Physics Phenomena Decomposition",
        "for_module": "Trajectory Control Planning",
        "num_items": len(ok_items),
        "items": [build_tcp_minimal_item(item) for item in ok_items],
    }
    dump_json(final, Path(out_file))
    return final


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPD from PFG, event number quantized to {1,2,4}")
    p.add_argument("--pfg_json", type=str, required=True, help="Path to pfg_minimal_for_ppd.json")
    p.add_argument("--out_file", type=str, default="./ppd_event_metadata_quantized124.json")
    p.add_argument("--model_path", type=str, default=DEFAULT_SNAP_DIR)
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Single-device placement, e.g. cuda:0 / cuda:4 / cpu. Use 'auto' to enable sharding.",
    )
    p.add_argument("--quiet", action="store_true")
    return p

def main() -> None:
    global VERBOSE
    args = build_argparser().parse_args()
    VERBOSE = not args.quiet

    _self_check_entity_pool_filter()

    vprint(f"[Init] pfg_json={args.pfg_json}")
    vprint(f"[Init] out_file={args.out_file}")
    vprint(f"[Init] model_path={args.model_path}")
    vprint(f"[Init] device={args.device}")

    final = run_ppd(args.pfg_json, args.out_file, args.model_path, args.device)
    print(json.dumps({
        "module": final["module"],
        "for_module": final["for_module"],
        "num_items": final["num_items"],
        "out_file": args.out_file,
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()