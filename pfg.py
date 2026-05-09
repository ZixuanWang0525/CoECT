#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PECR full pipeline, paper-aligned single-file version (v2)
==========================================================
Process:
1) Text Prompt -> physical law + formula name
2) Explicitly generate a Query bundle
3) Retrieve candidate pages online
4) Extract formula candidate set F_L from candidate pages
5) Perform TopK sorting on F_L; if direct matching fails, regenerate the formula_name and retrieve again
6) The LLM selects one concrete formula from the TopK formula candidates and provides variable semantics
7) The LLM generates a calculation-ready parameter table

python /mnt/43t/YixinHu/CVPR2026_code/pfg.py \
  --mode all \
  --debug_api
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from sympy.parsing.sympy_parser import parse_expr
from transformers import pipeline


DEFAULT_TIMEOUT = 12
DEFAULT_UA = "PECR-Paper-Aligned/2.0 (+academic use)"
WIKI_ACTION = "https://{lang}.wikipedia.org/w/api.php"
WIKIDATA_ACTION = "https://www.wikidata.org/w/api.php"
LANGS = ("en", "zh")
MAX_NEW_TOKENS = 1500
DEFAULT_SNAP_DIR = ""
SAMPLE_PROMPTS = [

]

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EVIDENCE_MAX_CHARS = 1200
CAND_TEXT_MAX_CHARS = 320
FORMULA_PROMPT_MAX_PAGES = 3
FORMULA_PROMPT_MAX_CANDS = 6
PARAM_PROMPT_MAX_CANDS = 3
LLM_MAX_INPUT_TOKENS = 6000
VERBOSE = True
DEBUG_API = False

def shorten_text(s: Any, limit: int = 180) -> str:
    s = str(s or "").replace("\n", " ").strip()
    return s if len(s) <= limit else s[:limit] + " ..."

def vprint(*args: Any, **kwargs: Any) -> None:
    if VERBOSE:
        print(*args, **kwargs)

def hprint(title: str) -> None:
    if VERBOSE:
        print("\n" + "=" * 18 + f" {title} " + "=" * 18)

def print_page_preview(tag: str, pages: Sequence[Dict[str, Any]], limit: int = 5) -> None:
    if not VERBOSE:
        return
    print(f"{tag}: {len(list(pages))} page(s)")
    for i, p in enumerate(list(pages)[:limit], start=1):
        print(f"  - [{i}] {p.get('lang','')}::{p.get('title','')} | src={p.get('source_type','')} | q={shorten_text(p.get('query_text',''), 60)} | page_rank={p.get('page_rank_score','')} | lexical={p.get('lexical_score','')} | title_match={p.get('title_match_score','')}")

def print_formula_preview(tag: str, cands: Sequence[Dict[str, Any]], limit: int = 8) -> None:
    if not VERBOSE:
        return
    print(f"{tag}: {len(list(cands))} formula candidate(s)")
    for i, c in enumerate(list(cands)[:limit], start=1):
        sp = c.get('source_page', {})
        print(f"  - [{i}] {c.get('candidate_id','')} | score={c.get('candidate_score','')} | page={sp.get('lang','')}::{sp.get('title','')} | text={shorten_text(c.get('candidate_text',''), 100)}")



STEP1_SYSTEM_PROMPT = """You are a physics formula grounding assistant.
Given one short physical phenomenon description, do exactly these tasks:

1) choose one category from {A Force, B Light, C Heat, D Material}
2) name the underlying physical law
3) name one concrete formula/equation that would be queried from a knowledge base

Hard rules:
- Return ONE JSON object only.
- Use standard, verifiable names.
- The formula_name should be more formula-like than the law name when possible.
- confidence must be in [0,1] with two decimals.

Return schema:
{
  "input_text": "string",
  "category": "A|B|C|D",
  "category_name": "Force|Light|Heat|Material",
  "physical_law": {
    "name": "string",
    "reason": "string <= 30 words"
  },
  "formula_name": {
    "name": "string",
    "reason": "string <= 30 words"
  },
  "confidence": 0.00
}
"""

STEP1_REGEN_TEMPLATE = """Your previous output failed because: {reason}
Regenerate ONE JSON object only.
- Start with '{{' and end with '}}'
- No prose, no markdown, no code fence
- Keep the same schema
- Use a standard physical law and a concrete formula/equation name
"""

FORMULA_NAME_REGEN_SYSTEM_PROMPT = """You are a formula-name regeneration assistant.
You will receive:
- the original text prompt
- the physical law
- the previous formula_name
- the query bundle and the extracted formula candidate set F_L

Your job:
1) decide whether the previous formula_name failed to directly match the retrieved evidence
2) if it failed, regenerate ONE better formula_name for the same physical law
3) keep the new name standard, queryable, and close to what a knowledge base page would use

Hard rules:
- Return ONE JSON object only.
- Do not change the physical law.
- The regenerated name must be a formula/equation name, not a page title guess.

Return schema:
{
  "status": "regenerated|keep_original",
  "formula_name": {
    "name": "string",
    "reason": "string <= 30 words"
  },
  "regeneration_reason": "string <= 40 words"
}
"""

FORMULA_SYSTEM_PROMPT = """You are a formula grounding assistant.
You will receive:
- the original text prompt
- the physical law
- the active formula_name
- a query bundle with retrieved pages
- a formula candidate set F_L with TopK-ranked candidates

Your job:
1) choose the single best formula candidate from F_L
2) output one concrete formula supported by that candidate and its source page
3) explain every variable in natural language
4) return a short supporting_span copied from the candidate evidence

Hard rules:
- The first character of your answer must be { and the last character must be }.
- Do NOT invent a candidate id that is not present in F_L.
- supporting_span must be copied from the provided candidate/page evidence as literally as possible.
- Do not output analysis or prose. Return ONE JSON object only.

Return schema:
{
  "status": "ok|not_found",
  "formula_name": "string",
  "source_formula_candidate_id": "string",
  "source_page": {
    "title": "string",
    "lang": "en|zh",
    "source_type": "exact|search|wikidata"
  },
  "supporting_span": "string",
  "formula": {
    "equation_latex": "string",
    "equation_sympy": "string",
    "variables": ["string", "..."],
    "variable_semantics": {"var": "meaning", "...": "..."},
    "formula_role_in_scene": "string"
  },
  "confidence": 0.00
}
"""

FORMULA_SCHEMA_REPAIR_SYSTEM_PROMPT = """You are a schema repair assistant for formula grounding.
You will receive:
- the original step1 output
- the compact query bundle
- a partial or malformed formula_grounding object

Your job:
1) convert the partial object into EXACTLY the required formula grounding schema
2) preserve any correct fields already present
3) if some fields are missing, infer them only from the provided candidate/page evidence
4) if no grounded formula can be supported, return a complete not_found object with all required fields present

Hard rules:
- Return ONE JSON object only.
- Do not output analysis.
- source_formula_candidate_id must match one candidate_id from topk_formula_candidates when status is ok.
- source_page must match that candidate's source_page when status is ok.
- supporting_span must be copied literally from the provided evidence when status is ok.
- formula.variables must be a list, and variable_semantics must be a dict.
"""

PARAM_SYSTEM_PROMPT = """You are a parameter initialization assistant.
You will receive:
- the original text prompt
- the physical law and formula name
- the grounded formula with variable semantics
- the retrieved evidence

Your job is to produce a calculation-ready parameter table.
That means every variable in the formula must receive one direct numeric value and one explicit unit.
When the text does not provide the value, infer a plausible scene-consistent numeric value by commonsense reasoning.

Hard rules:
- No symbolic / interval / relation outputs.
- Each variable must appear exactly once.
- Return ONE JSON object only.

Return schema:
{
  "status": "ready|not_ready",
  "is_calculation_ready": true,
  "parameter_table": [
    {"name": "string", "value": 0.0, "unit": "string", "rationale": "string <= 25 words"}
  ],
  "assumptions": ["string", "..."],
  "notes": "string"
}
"""


_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```", re.IGNORECASE)
NON_JSON_CHARS_RE = re.compile(r"^[^\{\[]*|[^\}\]]*$", re.S)


def strip_code_fences(text: str) -> str:
    m = _CODE_FENCE_RE.search(text or "")
    return m.group(1) if m else (text or "")


def extract_balanced_json(text: str) -> str:
    s = NON_JSON_CHARS_RE.sub("", text or "")
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


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_prompts_from_json(path: str | Path) -> Iterable[str]:
    data = load_json(path)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                txt = item.get("caption") or item.get("text") or item.get("prompt") or item.get("orig_text")
            else:
                txt = str(item)
            if txt:
                yield txt
        return
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for item in data["data"]:
            txt = item.get("caption") or item.get("text") or item.get("prompt") or item.get("orig_text")
            if txt:
                yield txt
        return
    raise ValueError("Unsupported prompts JSON structure")


def request_json(url: str, params: Dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    try:
        if DEBUG_API:
            print(f"[HTTP] GET {url} | params={shorten_text(json.dumps(params, ensure_ascii=False), 220)}")
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": DEFAULT_UA})
        if DEBUG_API:
            print(f"[HTTP] status={r.status_code} url={shorten_text(r.url, 220)}")
        r.raise_for_status()
        data = r.json()
        if DEBUG_API:
            print(f"[HTTP] json_head={shorten_text(json.dumps(data, ensure_ascii=False), 220)}")
        return data
    except Exception as e:
        vprint(f"[HTTP-ERR] {type(e).__name__}: {e} | url={url} | params={shorten_text(json.dumps(params, ensure_ascii=False), 220)}")
        raise


def wiki_title_query(title: str, lang: str) -> Dict[str, Any]:
    return request_json(WIKI_ACTION.format(lang=lang), {
        "action": "query",
        "format": "json",
        "prop": "extracts|revisions",
        "explaintext": 1,
        "exintro": 1,
        "rvprop": "content",
        "titles": title,
    })


def wiki_search(query: str, lang: str, limit: int = 5) -> List[Dict[str, Any]]:
    data = request_json(WIKI_ACTION.format(lang=lang), {
        "action": "query",
        "format": "json",
        "list": "search",
        "srlimit": limit,
        "srsearch": query,
    })
    return data.get("query", {}).get("search", [])


def wikidata_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    data = request_json(WIKIDATA_ACTION, {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "uselang": "en",
        "type": "item",
        "search": query,
        "limit": limit,
    })
    return data.get("search", [])


def clip_text(s: str, max_chars: int = EVIDENCE_MAX_CHARS) -> str:
    s = normalize_text(s)
    return s[:max_chars]


def compact_page_for_prompt(page: Dict[str, Any]) -> Dict[str, Any]:
    evidence = []
    for key in ['extract', 'search_snippet', 'revision_text']:
        val = normalize_text(page.get(key, ''))
        if val:
            evidence.append(val)
    merged = ' | '.join(evidence)
    return {
        'title': page.get('title', ''),
        'lang': page.get('lang', ''),
        'source_type': page.get('source_type', ''),
        'score': page.get('score', 0.0),
        'evidence_excerpt': clip_text(merged, EVIDENCE_MAX_CHARS),
    }


def compact_formula_candidate_for_prompt(cand: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'candidate_id': cand.get('candidate_id', ''),
        'score': cand.get('score', 0.0),
        'source_page': cand.get('source_page', {}),
        'candidate_text': clip_text(cand.get('candidate_text', ''), CAND_TEXT_MAX_CHARS),
        'source_query_text': clip_text(cand.get('source_query_text', ''), 120),
    }


def compact_query_bundle_for_formula_prompt(bundle: Dict[str, Any]) -> Dict[str, Any]:
    pages = bundle.get('retrieved_pages', [])[:FORMULA_PROMPT_MAX_PAGES]
    cands = bundle.get('topk_formula_candidates', [])[:FORMULA_PROMPT_MAX_CANDS]
    return {
        'law_name': bundle.get('law_name', ''),
        'formula_name_used': bundle.get('formula_name_used', ''),
        'query_variants': bundle.get('query_variants', [])[:4],
        'selected_page': bundle.get('selected_page', None),
        'retrieved_pages': [compact_page_for_prompt(p) for p in pages],
        'topk_formula_candidates': [compact_formula_candidate_for_prompt(c) for c in cands],
        'formula_name_regeneration': bundle.get('formula_name_regeneration', {}),
    }


def compact_query_bundle_for_param_prompt(bundle: Dict[str, Any], formula_obj: Dict[str, Any]) -> Dict[str, Any]:
    src_page = (formula_obj or {}).get('source_page', {})
    src_cand_id = (formula_obj or {}).get('source_formula_candidate_id', '')

    matched_page = None
    for p in bundle.get('retrieved_pages', []):
        if p.get('title') == src_page.get('title') and p.get('lang') == src_page.get('lang'):
            matched_page = p
            break

    matched_cand = None
    for c in bundle.get('formula_candidate_set_F_L', []):
        if c.get('candidate_id') == src_cand_id:
            matched_cand = c
            break

    top_cands = []
    if matched_cand is not None:
        top_cands.append(compact_formula_candidate_for_prompt(matched_cand))
    for c in bundle.get('topk_formula_candidates', [])[:PARAM_PROMPT_MAX_CANDS]:
        if matched_cand is None or c.get('candidate_id') != matched_cand.get('candidate_id'):
            top_cands.append(compact_formula_candidate_for_prompt(c))
        if len(top_cands) >= PARAM_PROMPT_MAX_CANDS:
            break

    return {
        'law_name': bundle.get('law_name', ''),
        'formula_name_used': bundle.get('formula_name_used', ''),
        'selected_page': src_page or bundle.get('selected_page', None),
        'selected_page_evidence': compact_page_for_prompt(matched_page) if matched_page else None,
        'topk_formula_candidates': top_cands,
    }


def _shrink_for_llm(obj: Any, *, max_str: int, max_list: int) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # 超长原文类字段直接强裁剪
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
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(ids)
    except Exception:
        try:
            return len(tokenizer(system_prompt + '\n\n' + user_prompt)['input_ids'])
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
        {'max_str': 520, 'max_list': 6},
        {'max_str': 320, 'max_list': 4},
        {'max_str': 220, 'max_list': 3},
    ]
    for sch in schedules:
        shrunk = _shrink_for_llm(payload, max_str=sch['max_str'], max_list=sch['max_list'])
        candidate = json.dumps(shrunk, ensure_ascii=False, indent=2)
        n_tok = estimate_chat_tokens(tokenizer, system_prompt, candidate)
        if n_tok == -1 or n_tok <= max_input_tokens:
            return candidate

    return json.dumps(_shrink_for_llm(payload, max_str=160, max_list=2), ensure_ascii=False, indent=2)


class LocalLLM:
    def __init__(self, model_path: str = DEFAULT_SNAP_DIR, max_new_tokens: int = MAX_NEW_TOKENS, device: str = "cuda:0"):
        if device == "auto":
            device_map = "auto"
        else:
            device_map = {"": device}
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.max_new_tokens = max_new_tokens

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raw_tokens = estimate_chat_tokens(self.pipe.tokenizer, system_prompt, user_prompt)
        packed_user_prompt = compact_user_prompt_to_budget(self.pipe.tokenizer, system_prompt, user_prompt, LLM_MAX_INPUT_TOKENS)
        packed_tokens = estimate_chat_tokens(self.pipe.tokenizer, system_prompt, packed_user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": packed_user_prompt},
        ]
        if VERBOSE:
            print(f"[LLM] prompt_tokens raw={raw_tokens} packed={packed_tokens} budget={LLM_MAX_INPUT_TOKENS}")
            print(f"[LLM] user_prompt_head={shorten_text(packed_user_prompt, 160)}")

        out = self.pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=0.0, top_p=1.0)
        gen = out[0].get("generated_text")
        content = gen[-1].get("content", "") if isinstance(gen, list) else str(gen)
        if VERBOSE:
            print(f"[LLM] raw_output_head={shorten_text(content, 300)}")

        try:
            return extract_json(content)
        except Exception as e:
            if VERBOSE:
                print(f"[LLM] first JSON parse failed -> {type(e).__name__}: {e}")
                print("[LLM] retry with strict JSON-only repair prompt")

            repair_prompt = (
                "Return ONE valid JSON object only. No prose. No markdown. No explanation. "
                "Do not output analysis. If the raw output contains no usable JSON, return a minimal failure JSON object.\n\n"
                f"SYSTEM PROMPT TYPE:\n{system_prompt[:400]}\n\n"
                f"RAW OUTPUT:\n{content[:4000]}"
            )
            repair_messages = [
                {"role": "system", "content": "Return ONE valid JSON object only. No prose. No markdown. No explanation."},
                {"role": "user", "content": repair_prompt},
            ]
            out2 = self.pipe(repair_messages, max_new_tokens=min(512, self.max_new_tokens), do_sample=False, temperature=0.0, top_p=1.0)
            gen2 = out2[0].get("generated_text")
            content2 = gen2[-1].get("content", "") if isinstance(gen2, list) else str(gen2)
            if VERBOSE:
                print(f"[LLM] repair_output_head={shorten_text(content2, 300)}")
            try:
                return extract_json(content2)
            except Exception as e2:
                if VERBOSE:
                    print(f"[LLM] repair JSON parse failed -> {type(e2).__name__}: {e2}")
                sp = system_prompt.lower()
                if 'formula grounding assistant' in sp:
                    return {
                        "status": "not_found",
                        "formula_name": "",
                        "source_formula_candidate_id": "",
                        "source_page": {"title": "", "lang": "", "source_type": ""},
                        "supporting_span": "",
                        "formula": {
                            "equation_latex": "",
                            "equation_sympy": "",
                            "variables": [],
                            "variable_semantics": {},
                            "formula_role_in_scene": ""
                        },
                        "confidence": 0.0,
                        "reason": "model_output_not_json"
                    }
                if 'parameter initialization assistant' in sp:
                    return {
                        "status": "not_ready",
                        "is_calculation_ready": False,
                        "parameter_table": [],
                        "assumptions": [],
                        "notes": "model_output_not_json"
                    }
                return {"status": "failed", "reason": "model_output_not_json"}


def build_step1_prompt(text: str, reason: str = "") -> str:
    user = f"Original text: {text}"
    if reason:
        user += "\n\n" + STEP1_REGEN_TEMPLATE.format(reason=reason)
    return user


def validate_step1_obj(obj: Dict[str, Any]) -> Tuple[bool, str]:
    need = {"input_text", "category", "category_name", "physical_law", "formula_name", "confidence"}
    if not isinstance(obj, dict):
        return False, "result is not an object"
    miss = [k for k in need if k not in obj]
    if miss:
        return False, f"missing keys: {miss}"
    for key in ["physical_law", "formula_name"]:
        sub = obj.get(key)
        if not isinstance(sub, dict):
            return False, f"{key} is not an object"
        if not sub.get("name"):
            return False, f"{key}.name is empty"
    return True, "basic structure ok"


def wiki_exists(name: str, langs: Sequence[str] = LANGS) -> Tuple[bool, str]:
    for lang in langs:
        try:
            data = wiki_title_query(name, lang)
            pages = data.get("query", {}).get("pages", {})
            if pages:
                page = next(iter(pages.values()))
                if "missing" not in page:
                    return True, f"exact hit on {lang}"
        except Exception:
            pass
        try:
            if wiki_search(name, lang, limit=1):
                return True, f"search hit on {lang}"
        except Exception:
            pass
    return False, "no hit on en/zh Wikipedia"


def validate_step1_against_wiki(obj: Dict[str, Any], langs: Sequence[str] = LANGS) -> Tuple[bool, Dict[str, Any]]:
    ok0, reason0 = validate_step1_obj(obj)
    if not ok0:
        return False, {"ok": False, "reason": reason0}
    law_ok, law_reason = wiki_exists(obj["physical_law"]["name"], langs)
    formula_ok, formula_reason = wiki_exists(obj["formula_name"]["name"], langs)
    ok = law_ok and formula_ok
    info = {
        "ok": ok,
        "law_check": {"ok": law_ok, "reason": law_reason},
        "formula_name_check": {"ok": formula_ok, "reason": formula_reason},
    }
    if not ok:
        info["reason"] = "physical_law or formula_name not found on en/zh Wikipedia"
    return ok, info


def run_step1_generation(prompts: List[str], llm: LocalLLM, out_dir: Path, langs: Sequence[str], save_intermediate: bool = False) -> List[Dict[str, Any]]:
    rows = []
    total = len(prompts)
    for idx, text in enumerate(prompts, start=1):
        hprint(f"Step1 {idx}/{total}")
        print(f"[Step1] input={text}")
        last_reason = ""
        obj = None
        check = None
        for attempt in range(1, 4):
            print(f"[Step1] attempt={attempt} regenerate_reason={shorten_text(last_reason, 120) if last_reason else '<none>'}")
            obj = llm.generate_json(STEP1_SYSTEM_PROMPT, build_step1_prompt(text, last_reason))
            print(f"[Step1] physical_law={obj.get('physical_law', {}).get('name', '')} | formula_name={obj.get('formula_name', {}).get('name', '')}")
            ok, check = validate_step1_against_wiki(obj, langs)
            print(f"[Step1] validation_ok={ok} | detail={check}")
            if ok:
                break
            last_reason = json.dumps(check, ensure_ascii=False)
        rows.append({
            "index": idx,
            "step1": obj,
            "step1_validation": check,
        })
    if save_intermediate:
        dump_json(rows, out_dir / "step1_verified.json")
    return rows


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\n", " ")).strip().lower()


def simple_keywords(text: str, top_k: int = 8) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    stop = {"the", "a", "an", "of", "in", "into", "on", "to", "with", "and", "was", "were", "is", "are"}
    out = []
    for t in toks:
        if t not in stop and t not in out:
            out.append(t)
    return out[:top_k]


def build_query_variants(step1_obj: Dict[str, Any], formula_name_override: Optional[str] = None) -> List[str]:
    law = step1_obj["physical_law"]["name"]
    formula = formula_name_override or step1_obj["formula_name"]["name"]
    text = step1_obj["input_text"]
    kws = " ".join(simple_keywords(text, top_k=6))
    queries = [
        formula,
        law,
        f"{formula} {law}",
        f"{formula} {kws}",
        f"{law} {kws}",
        f"{formula} physical formula",
    ]
    seen, uniq = set(), []
    for q in queries:
        if q and q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq


def lexical_score(query_text: str, page_text: str) -> float:
    q = set(re.findall(r"[a-zA-Z]+", query_text.lower()))
    p = set(re.findall(r"[a-zA-Z]+", page_text.lower()))
    return float(len(q & p)) / float(max(1, len(q)))


def title_match_score(name: str, title: str) -> float:
    name_n = normalize_text(name)
    title_n = normalize_text(title)
    if not name_n or not title_n:
        return 0.0
    if name_n == title_n:
        return 1.0
    q = set(re.findall(r"[a-zA-Z]+", name_n))
    t = set(re.findall(r"[a-zA-Z]+", title_n))
    return float(len(q & t)) / float(max(1, len(q)))


def fetch_page_payload(title: str, lang: str) -> Optional[Dict[str, Any]]:
    try:
        data = wiki_title_query(title, lang)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        if "missing" in page:
            return None
        revs = page.get("revisions", [])
        rev_text = ""
        if revs:
            slots = revs[0].get("slots", {})
            if isinstance(slots, dict):
                rev_text = slots.get("main", {}).get("*") or ""
            if not rev_text:
                rev_text = revs[0].get("*") or ""
        return {
            "title": page.get("title", title),
            "lang": lang,
            "pageid": page.get("pageid"),
            "extract": page.get("extract", ""),
            "revision_text": rev_text,
        }
    except Exception:
        return None


def build_page_query_bundle(step1_obj: Dict[str, Any], languages: Sequence[str], formula_name_override: Optional[str] = None) -> Dict[str, Any]:
    variants = build_query_variants(step1_obj, formula_name_override=formula_name_override)
    active_formula_name = formula_name_override or step1_obj["formula_name"]["name"]
    candidates: List[Dict[str, Any]] = []

    hprint("Step2A / Query bundle")
    print(f"[Query] law={step1_obj['physical_law']['name']}")
    print(f"[Query] formula_name_used={active_formula_name}")
    print(f"[Query] variants={variants}")

    for lang in languages:
        for q in variants[:2]:
            vprint(f"[ExactTitle] lang={lang} query={q}")
            payload = fetch_page_payload(q, lang)
            if payload:
                payload.update({"query_text": q, "source_type": "exact"})
                candidates.append(payload)
                vprint(f"[ExactTitle] HIT -> {lang}::{payload.get('title','')} | extract_head={shorten_text(payload.get('extract',''), 90)}")
            else:
                vprint(f"[ExactTitle] MISS -> {lang}::{q}")

    for lang in languages:
        for q in variants:
            try:
                items = wiki_search(q, lang, limit=3)
                vprint(f"[Search] lang={lang} query={q} -> {len(items)} hit(s)")
                for item in items:
                    payload = fetch_page_payload(item.get("title", ""), lang)
                    if payload:
                        payload.update({
                            "query_text": q,
                            "source_type": "search",
                            "search_snippet": item.get("snippet", ""),
                        })
                        candidates.append(payload)
                        vprint(f"[Search] keep page={lang}::{payload.get('title','')} | snippet={shorten_text(item.get('snippet',''), 80)}")
            except Exception as e:
                vprint(f"[Search-ERR] lang={lang} query={q} -> {type(e).__name__}: {e}")
                continue

    for q in variants[:3]:
        try:
            items = wikidata_search(q, limit=3)
            vprint(f"[Wikidata] query={q} -> {len(items)} hit(s)")
            for item in items:
                title = item.get("label")
                if not title:
                    continue
                payload = fetch_page_payload(title, "en")
                if payload:
                    payload.update({
                        "query_text": q,
                        "source_type": "wikidata",
                        "wikidata_id": item.get("id"),
                        "wikidata_desc": item.get("description", ""),
                    })
                    candidates.append(payload)
                    vprint(f"[Wikidata] keep page=en::{payload.get('title','')} | item={item.get('id','')} | desc={shorten_text(item.get('description',''), 80)}")
        except Exception as e:
            vprint(f"[Wikidata-ERR] query={q} -> {type(e).__name__}: {e}")
            continue

    print(f"[Query] raw_candidate_pages={len(candidates)}")
    uniq = {}
    ranking_query = " ".join(variants + [step1_obj["input_text"]])
    for item in candidates:
        key = (item.get("lang"), item.get("title"))
        text = " ".join([item.get("extract", ""), item.get("revision_text", ""), item.get("search_snippet", "")])
        item["lexical_score"] = lexical_score(ranking_query, text)
        item["title_match_score"] = title_match_score(active_formula_name, item.get("title", ""))
        item["page_rank_score"] = round(0.7 * item["lexical_score"] + 0.3 * item["title_match_score"], 4)
        if key not in uniq or item["page_rank_score"] > uniq[key]["page_rank_score"]:
            uniq[key] = item
    ranked = sorted(uniq.values(), key=lambda x: x.get("page_rank_score", 0.0), reverse=True)
    print_page_preview("[Query] ranked_pages", ranked, limit=8)

    return {
        "law_name": step1_obj["physical_law"]["name"],
        "formula_name_original": step1_obj["formula_name"]["name"],
        "formula_name_used": active_formula_name,
        "query_variants": variants,
        "retrieved_pages": ranked[:12],
        "selected_page": ranked[0] if ranked else None,
    }


def extract_equation_like_spans(text: str) -> List[str]:
    spans = []
    spans.extend(re.findall(r"<math>(.*?)</math>", text or "", flags=re.I | re.S))
    spans.extend(re.findall(r"\$([^\$]{1,120}=[^\$]{1,120})\$", text or "", flags=re.S))
    spans.extend(re.findall(r"([A-Za-z0-9_\\\^\{\}\(\)\[\]\+\-\*/·\. ]{1,120}=[A-Za-z0-9_\\\^\{\}\(\)\[\]\+\-\*/·\. ]{1,120})", text or ""))
    clean = []
    seen = set()
    for s in spans:
        s2 = re.sub(r"\s+", " ", s).strip()
        if len(s2) >= 5 and s2 not in seen:
            clean.append(s2)
            seen.add(s2)
    return clean


def sentence_chunks(text: str, max_len: int = 240) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    parts = re.split(r'(?<=[\.\!\?。！？；;:])\s+', text)

    chunks: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(part) <= max_len:
            chunks.append(part)
            continue

        for i in range(0, len(part), max_len):
            sub = part[i:i + max_len].strip()
            if sub:
                chunks.append(sub)

    return chunks


def clean_wiki_markup_text(text: str) -> str:
    s = text or ""
    s = re.sub(r"<ref[^>]*?>.*?</ref>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<ref[^/]*/>", " ", s, flags=re.I | re.S)
    s = re.sub(r"\{\{[^\{\}]{0,400}\}\}", " ", s)
    s = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\|\s*(doi|isbn|publisher|access-date|archive-url|archive-date)\s*=.*?(?=\||$)", " ", s, flags=re.I)
    s = re.sub(r"==+[^=]+==+", " ", s)
    s = s.replace('&nbsp;', ' ').replace('&quot;', ' ').replace('&amp;', '&')
    s = re.sub(r"[{}|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def page_is_relevant_for_formula(step1_obj: Dict[str, Any], page: Dict[str, Any]) -> bool:
    law = normalize_text(step1_obj["physical_law"]["name"])
    formula = normalize_text(step1_obj["formula_name"]["name"])
    title = normalize_text(page.get("title", ""))
    text = normalize_text(" ".join([page.get("extract", ""), page.get("search_snippet", "")]))

    if title_match_score(step1_obj["physical_law"]["name"], page.get("title", "")) >= 0.65:
        return True
    if title_match_score(step1_obj["formula_name"]["name"], page.get("title", "")) >= 0.65:
        return True
    if law and law in text:
        return True
    if formula and formula in text:
        return True
    if page.get("page_rank_score", 0.0) >= 0.75 and title not in {'english law', 'scientific law', '法律'}:
        return True
    return False


def is_formula_like_span(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if '=' in text:
        return True
    keywords = ['equation', 'formula', 'refractive index', 'refraction', 'incident angle', 'refracted angle', 'snell']
    return any(k in t for k in keywords)


def is_bad_formula_candidate_text(text: str) -> bool:
    t = normalize_text(text)

    bad_patterns = [
        '#redirect', 'redirect category shell', '[[category:', '{{reflist', '{{authority control',
        '{{short description', '{{other uses', 'isbn=', 'access-date', 'publisher=',
        'archive-url', 'archive-date', 'doi=', '/meta', 'paper/view/', 'work=', 'title=',
        'url=', 'bibcode=', 'pmid', 's2cid', 'cite web', 'cite book', 'cite journal'
    ]
    if any(p in t for p in bad_patterns):
        return True
    if len(t) < 20:
        return True
    if text.count('|') >= 3 or text.count('{') >= 2 or text.count('[') >= 4:
        return True
    if re.search(r"\b(year|isbn|doi|publisher|access-date|archive-date|archive-url)\b", t):
        return True
    return False


def build_formula_candidates_from_pages(step1_obj: Dict[str, Any], query_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    law = step1_obj["physical_law"]["name"]
    formula_name = query_bundle["formula_name_used"]
    prompt_kws = simple_keywords(step1_obj["input_text"], top_k=8)
    formula_candidates: List[Dict[str, Any]] = []
    cid = 1

    hprint("Step2B / Build formula candidate set F_L")
    print(f"[F_L] retrieved_pages={len(query_bundle.get('retrieved_pages', []))}")

    for page in query_bundle.get("retrieved_pages", []):
        if not page_is_relevant_for_formula(step1_obj, page):
            vprint(f"[F_L] skip irrelevant page={page.get('lang','')}::{page.get('title','')}")
            continue

        cleaned_revision = clean_wiki_markup_text(page.get("revision_text", ""))
        page_text = " ".join([page.get("extract", ""), cleaned_revision, page.get("search_snippet", "")])
        page_title = page.get("title", "")
        page_lang = page.get("lang", "")
        candidate_spans: List[Tuple[str, str]] = []

        for eq in extract_equation_like_spans(page_text):
            candidate_spans.append((eq, 'equation'))

        for sent in sentence_chunks(page_text):
            sent_n = normalize_text(sent)
            if (
                '=' in sent
                or 'equation' in sent_n
                or 'formula' in sent_n
                or normalize_text(formula_name) in sent_n
                or normalize_text(law) in sent_n
            ):
                candidate_spans.append((sent.strip(), 'support_sentence'))

        vprint(f"[F_L] page={page_lang}::{page_title} | raw_candidate_spans={len(candidate_spans)}")
        seen_local = set()
        for span, cand_type in candidate_spans:
            span = re.sub(r"\s+", " ", span).strip()
            if len(span) < 5 or span in seen_local:
                continue
            if is_bad_formula_candidate_text(span):
                continue
            if not is_formula_like_span(span):
                continue
            seen_local.add(span)

            local_score = 0.0
            local_score += 0.28 * lexical_score(formula_name, span)
            local_score += 0.18 * lexical_score(law, span)
            local_score += 0.08 * lexical_score(" ".join(prompt_kws), span)
            local_score += 0.18 * title_match_score(formula_name, page_title)
            local_score += 0.08 * title_match_score(law, page_title)
            local_score += 0.10 * page.get("page_rank_score", 0.0)
            if page.get("source_type") == "exact":
                local_score += 0.06
            if cand_type == 'equation':
                local_score += 0.12
            if '=' in span:
                local_score += 0.10
            if 'refractive indices' in normalize_text(span) or 'incident angle' in normalize_text(span) or 'refracted angle' in normalize_text(span):
                local_score += 0.08

            formula_candidates.append({
                "candidate_id": f"F{cid:03d}",
                "candidate_formula_name": formula_name,
                "candidate_type": cand_type,
                "candidate_text": span[:400],
                "source_page": {
                    "title": page_title,
                    "lang": page_lang,
                    "source_type": page.get("source_type", "search"),
                },
                "source_query_text": page.get("query_text", ""),
                "candidate_score": round(local_score, 4),
                "page_rank_score": page.get("page_rank_score", 0.0),
            })
            cid += 1

    uniq = {}
    for cand in formula_candidates:
        key = normalize_text(cand.get('candidate_text', ''))
        if key not in uniq or cand.get('candidate_score', 0.0) > uniq[key].get('candidate_score', 0.0):
            uniq[key] = cand
    formula_candidates = sorted(uniq.values(), key=lambda x: x.get("candidate_score", 0.0), reverse=True)
    print_formula_preview("[F_L] ranked_formula_candidates", formula_candidates, limit=10)
    return formula_candidates



def has_direct_formula_match(step1_obj: Dict[str, Any], formula_candidates: Sequence[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    formula_name = step1_obj["formula_name"]["name"]
    inspected = []
    for cand in formula_candidates[:5]:
        name_score = lexical_score(formula_name, cand.get("candidate_text", ""))
        title_score = title_match_score(formula_name, cand.get("source_page", {}).get("title", ""))
        inspected.append({
            "candidate_id": cand.get("candidate_id"),
            "candidate_score": cand.get("candidate_score", 0.0),
            "name_score": round(name_score, 4),
            "title_score": round(title_score, 4),
            "page_title": cand.get("source_page", {}).get("title", ""),
        })
        if cand.get("candidate_score", 0.0) >= 0.45 or name_score >= 0.45 or title_score >= 0.80:
            vprint(f"[DirectMatch] HIT -> {inspected[-1]}")
            return True, {
                "direct_match": True,
                "reason": "top formula candidates already align with formula_name",
                "matched_candidate_id": cand.get("candidate_id"),
                "matched_candidate_score": cand.get("candidate_score", 0.0),
                "inspected_top_candidates": inspected,
            }
    vprint(f"[DirectMatch] MISS | inspected={inspected}")
    return False, {
        "direct_match": False,
        "reason": "no strong formula-level match found in top candidates",
        "inspected_top_candidates": inspected,
    }


def build_formula_name_regen_user_prompt(step1_obj: Dict[str, Any], query_bundle: Dict[str, Any], formula_candidates: Sequence[Dict[str, Any]], direct_match_check: Dict[str, Any]) -> str:
    return json.dumps({
        "task": "regenerate formula_name when direct formula-level match fails",
        "step1_output": step1_obj,
        "query_bundle": {
            "formula_name_original": query_bundle.get("formula_name_original"),
            "formula_name_used": query_bundle.get("formula_name_used"),
            "query_variants": query_bundle.get("query_variants", []),
            "retrieved_pages": query_bundle.get("retrieved_pages", [])[:5],
        },
        "formula_candidate_set_F_L": list(formula_candidates[:8]),
        "direct_match_check": direct_match_check,
        "instruction": "If the current formula_name does not directly match the retrieved evidence, output a better formula_name for the same physical law.",
    }, ensure_ascii=False, indent=2)


def validate_formula_name_regen(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "regen result is not an object"
    if obj.get("status") not in {"regenerated", "keep_original"}:
        return False, "status must be regenerated or keep_original"
    sub = obj.get("formula_name")
    if not isinstance(sub, dict) or not sub.get("name"):
        return False, "formula_name.name is empty"
    return True, "ok"


def clone_step1_with_formula_name(step1_obj: Dict[str, Any], new_formula_name: str, reason: str) -> Dict[str, Any]:
    cloned = json.loads(json.dumps(step1_obj, ensure_ascii=False))
    cloned["formula_name"]["name"] = new_formula_name
    if reason:
        cloned["formula_name"]["reason"] = reason
    return cloned


def build_retrieval_package(step1_obj: Dict[str, Any], llm: LocalLLM, languages: Sequence[str]) -> Dict[str, Any]:
    hprint("Step2 / Retrieval package")
    print(f"[Retrieval] input_text={step1_obj['input_text']}")
    print(f"[Retrieval] law={step1_obj['physical_law']['name']} | formula_name={step1_obj['formula_name']['name']}")

    print("[Retrieval] attempt=1 using original formula_name")
    qb1 = build_page_query_bundle(step1_obj, languages)
    fl1 = build_formula_candidates_from_pages(step1_obj, qb1)
    direct_ok, direct_info = has_direct_formula_match(step1_obj, fl1)
    print(f"[Retrieval] attempt=1 direct_match={direct_ok} | detail={direct_info}")

    if direct_ok:
        qb1["formula_candidate_set_F_L"] = fl1[:12]
        qb1["topk_formula_candidates"] = fl1[:5]
        qb1["formula_name_regeneration"] = {
            "status": "not_needed",
            "reason": direct_info.get("reason", ""),
        }
        return {
            "step1_effective": step1_obj,
            "query_bundle": qb1,
            "formula_name_regeneration": qb1["formula_name_regeneration"],
        }

    print("[Retrieval] attempt=2 regenerate formula_name")
    regen_raw = llm.generate_json(
        FORMULA_NAME_REGEN_SYSTEM_PROMPT,
        build_formula_name_regen_user_prompt(step1_obj, qb1, fl1, direct_info),
    )
    print(f"[Retrieval] regen_raw={regen_raw}")
    regen_ok, regen_reason = validate_formula_name_regen(regen_raw)
    print(f"[Retrieval] regen_ok={regen_ok} | regen_reason={regen_reason}")
    if not regen_ok:
        qb1["formula_candidate_set_F_L"] = fl1[:12]
        qb1["topk_formula_candidates"] = fl1[:5]
        qb1["formula_name_regeneration"] = {
            "status": "failed",
            "reason": regen_reason,
        }
        return {
            "step1_effective": step1_obj,
            "query_bundle": qb1,
            "formula_name_regeneration": qb1["formula_name_regeneration"],
        }

    if regen_raw.get("status") == "keep_original":
        print("[Retrieval] regeneration says keep_original")
        qb1["formula_candidate_set_F_L"] = fl1[:12]
        qb1["topk_formula_candidates"] = fl1[:5]
        qb1["formula_name_regeneration"] = regen_raw
        return {
            "step1_effective": step1_obj,
            "query_bundle": qb1,
            "formula_name_regeneration": regen_raw,
        }

    new_formula_name = regen_raw["formula_name"]["name"]
    step1_effective = clone_step1_with_formula_name(step1_obj, new_formula_name, regen_raw["formula_name"].get("reason", ""))
    print(f"[Retrieval] regenerated_formula_name={new_formula_name}")

    qb2 = build_page_query_bundle(step1_effective, languages, formula_name_override=new_formula_name)
    fl2 = build_formula_candidates_from_pages(step1_effective, qb2)
    direct_ok2, direct_info2 = has_direct_formula_match(step1_effective, fl2)
    print(f"[Retrieval] attempt=2 direct_match={direct_ok2} | detail={direct_info2}")

    qb2["formula_candidate_set_F_L"] = fl2[:12]
    qb2["topk_formula_candidates"] = fl2[:5]
    qb2["formula_name_regeneration"] = regen_raw
    qb2["attempt_history"] = [
        {
            "attempt": 1,
            "formula_name_used": step1_obj["formula_name"]["name"],
            "direct_match_check": direct_info,
        },
        {
            "attempt": 2,
            "formula_name_used": new_formula_name,
            "direct_match_check": direct_info2,
        },
    ]
    return {
        "step1_effective": step1_effective,
        "query_bundle": qb2,
        "formula_name_regeneration": regen_raw,
    }


def build_formula_user_prompt(step1_obj: Dict[str, Any], query_bundle: Dict[str, Any]) -> str:
    compact_bundle = compact_query_bundle_for_formula_prompt(query_bundle)
    return json.dumps({
        "task": "ground one concrete formula from TopK-ranked formula candidates",
        "step1_output": step1_obj,
        "query_bundle": compact_bundle,
        "instruction": (
            "Return ONE JSON object only, and the first character of your answer must be '{'. "
            "Choose exactly one candidate from topk_formula_candidates. "
            "The field source_formula_candidate_id MUST copy the candidate_id exactly, character by character. "
            "source_page must be the same page as that candidate's source_page. "
            "supporting_span must be copied literally from the chosen candidate/page evidence. "
            "Do not output analysis. Do not output markdown. Do not copy wiki templates, redirects, categories, references, or raw page text outside JSON fields. "
            "If all candidates are bad, return a not_found JSON object with the required schema."
        ),
    }, ensure_ascii=False, indent=2)


def page_text_from_bundle(bundle: Dict[str, Any], title: str, lang: str) -> str:
    for p in bundle.get("retrieved_pages", []):
        if p.get("title") == title and p.get("lang") == lang:
            return " ".join([p.get("extract", ""), p.get("revision_text", ""), p.get("search_snippet", "")])
    return ""


def candidate_text_from_bundle(bundle: Dict[str, Any], candidate_id: str) -> str:
    for cand in bundle.get("formula_candidate_set_F_L", []):
        if cand.get("candidate_id") == candidate_id:
            return " ".join([cand.get("candidate_text", ""), cand.get("source_query_text", "")])
    return ""


def normalize_equation_sympy_text(expr_text: str) -> str:
    s = str(expr_text or '').strip()
    repl = {
        '−': '-', '–': '-', '—': '-',
        '×': '*', '·': '*',
        '^': '**',
        'θ': 'theta', 'Θ': 'theta',
        'ρ': 'rho', 'Ρ': 'rho',
        'γ': 'gamma', 'Γ': 'gamma',
        'δ': 'delta', 'Δ': 'delta',
        'μ': 'mu', 'Μ': 'mu',
        'λ': 'lambda', 'Λ': 'lambda',
        'φ': 'phi', 'Φ': 'phi',
        'ω': 'omega', 'Ω': 'omega',
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"\\(left|right)", "", s)
    s = s.replace('{', '').replace('}', '')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_equation_symbols(expr_text: str) -> Tuple[bool, List[str], str]:
    if not expr_text:
        return False, [], "equation_sympy is empty"

    expr_text = normalize_equation_sympy_text(expr_text)
    try:
        if '=' in expr_text and '==' not in expr_text and '>=' not in expr_text and '<=' not in expr_text:
            lhs, rhs = expr_text.split('=', 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if not lhs or not rhs:
                return False, [], 'equation_sympy contains an incomplete equality'
            left_expr = parse_expr(lhs, evaluate=False)
            right_expr = parse_expr(rhs, evaluate=False)
            syms = sorted({str(s) for s in left_expr.free_symbols} | {str(s) for s in right_expr.free_symbols})
            return True, syms, 'ok'

        expr = parse_expr(expr_text, evaluate=False)
        syms = sorted(str(s) for s in expr.free_symbols)
        return True, syms, 'ok'
    except Exception as e:
        return False, [], f"sympy parse failed: {e}"


def normalize_identifier(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").casefold())


def normalize_page_title(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).casefold()


def bind_formula_candidate(query_bundle: Dict[str, Any], grounded_obj: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    raw_id = grounded_obj.get("source_formula_candidate_id", "")
    if not raw_id:
        return None, "source_formula_candidate_id is empty"

    cid_norm = normalize_identifier(raw_id)
    if not cid_norm:
        return None, "source_formula_candidate_id is empty after normalization"

    cand_map: Dict[str, Dict[str, Any]] = {}
    for cand in query_bundle.get("formula_candidate_set_F_L", []):
        k = normalize_identifier(cand.get("candidate_id", ""))
        if k:
            cand_map[k] = cand

    matched = cand_map.get(cid_norm)
    if matched is None:
        return None, "source_formula_candidate_id is not present in formula_candidate_set_F_L"

    return matched, "ok"


def bind_source_page(query_bundle: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    cand_page = candidate.get("source_page", {}) or {}
    cand_title = normalize_page_title(cand_page.get("title", ""))
    cand_lang = str(cand_page.get("lang", "") or "").strip().lower()

    for page in query_bundle.get("retrieved_pages", []):
        page_title = normalize_page_title(page.get("title", ""))
        page_lang = str(page.get("lang", "") or "").strip().lower()
        if page_title == cand_title and page_lang == cand_lang:
            return page, "ok"

    return None, "candidate source page is not present in query_bundle.retrieved_pages"


def validate_formula_grounding(obj: Dict[str, Any], query_bundle: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    need = {"status", "formula_name", "source_formula_candidate_id", "source_page", "supporting_span", "formula", "confidence"}
    if not isinstance(obj, dict):
        return False, {"ok": False, "reason": "result is not an object"}

    miss = [k for k in need if k not in obj]
    if miss:
        return False, {"ok": False, "reason": f"missing keys: {miss}"}

    if obj.get("status") != "ok":
        return False, {"ok": False, "reason": "status is not ok"}

    matched_cand, bind_reason = bind_formula_candidate(query_bundle, obj)
    if matched_cand is None:
        return False, {"ok": False, "reason": bind_reason}

    obj["source_formula_candidate_id"] = matched_cand.get("candidate_id", "")
    obj["source_page"] = json.loads(json.dumps(matched_cand.get("source_page", {}), ensure_ascii=False))

    matched_page, page_reason = bind_source_page(query_bundle, matched_cand)
    if matched_page is None:
        return False, {"ok": False, "reason": page_reason}

    formula = obj.get("formula", {})
    if not isinstance(formula, dict):
        return False, {"ok": False, "reason": "formula is not an object"}

    vars_decl = formula.get("variables", [])
    var_sem = formula.get("variable_semantics", {})
    if not isinstance(vars_decl, list) or not vars_decl:
        return False, {"ok": False, "reason": "formula.variables is empty"}

    if not isinstance(var_sem, dict):
        return False, {"ok": False, "reason": "variable_semantics is not a dict"}

    missing_sem = [v for v in vars_decl if v not in var_sem or not str(var_sem.get(v, "")).strip()]
    if missing_sem:
        return False, {"ok": False, "reason": f"missing variable semantics for: {missing_sem}"}

    ok_expr, expr_syms, expr_reason = parse_equation_symbols(formula.get("equation_sympy", ""))
    if not ok_expr:
        return False, {"ok": False, "reason": expr_reason}

    undeclared = [s for s in expr_syms if s not in vars_decl]
    if undeclared:
        return False, {"ok": False, "reason": f"equation contains undeclared symbols: {undeclared}"}

    span = normalize_text(obj.get("supporting_span", ""))
    if not span:
        return False, {"ok": False, "reason": "supporting_span is empty"}

    cand_text = normalize_text(candidate_text_from_bundle(query_bundle, matched_cand.get("candidate_id", "")))
    page_text = normalize_text(page_text_from_bundle(
        query_bundle,
        matched_cand.get("source_page", {}).get("title", ""),
        matched_cand.get("source_page", {}).get("lang", "")
    ))

    if span not in cand_text and span not in page_text:
        return False, {"ok": False, "reason": "supporting_span is not supported by selected candidate/page text"}

    grounded_formula_name = normalize_text(obj.get("formula_name", ""))
    cand_formula_name = normalize_text(matched_cand.get("candidate_formula_name", ""))
    active_formula_name = normalize_text(query_bundle.get("formula_name_used", ""))

    if grounded_formula_name and grounded_formula_name not in {cand_formula_name, active_formula_name}:
        return False, {
            "ok": False,
            "reason": "formula_name is inconsistent with selected candidate and active formula_name"
        }

    return True, {
        "ok": True,
        "reason": "formula grounding is supported by a selected formula candidate and passes structural checks",
        "selected_candidate_id": matched_cand.get("candidate_id", ""),
        "selected_page": matched_cand.get("source_page", {}),
        "parsed_symbols": expr_syms,
    }


def formula_grounding_required_keys() -> List[str]:
    return [
        'status', 'formula_name', 'source_formula_candidate_id',
        'source_page', 'supporting_span', 'formula', 'confidence'
    ]


def get_formula_schema_issue(obj: Dict[str, Any]) -> str:
    if not isinstance(obj, dict):
        return 'result is not an object'
    miss = [k for k in formula_grounding_required_keys() if k not in obj]
    if miss:
        return f'missing keys: {miss}'
    if not isinstance(obj.get('formula', {}), dict):
        return 'formula is not an object'
    return ''


def formula_grounding_stub(query_bundle: Dict[str, Any], status: str = 'not_found', reason: str = '') -> Dict[str, Any]:
    top = (query_bundle.get('topk_formula_candidates') or [{}])[0]
    page = top.get('source_page', {}) if isinstance(top, dict) else {}
    span = clip_text(top.get('candidate_text', ''), 160) if isinstance(top, dict) else ''
    return {
        'status': status,
        'formula_name': query_bundle.get('formula_name_used', ''),
        'source_formula_candidate_id': top.get('candidate_id', '') if status == 'ok' else '',
        'source_page': {
            'title': page.get('title', '') if status == 'ok' else '',
            'lang': page.get('lang', '') if status == 'ok' else '',
            'source_type': page.get('source_type', '') if status == 'ok' else '',
        },
        'supporting_span': span if status == 'ok' else '',
        'formula': {
            'equation_latex': '',
            'equation_sympy': '',
            'variables': [],
            'variable_semantics': {},
            'formula_role_in_scene': reason or '',
        },
        'confidence': 0.0,
    }


def normalize_formula_grounding_schema(obj: Dict[str, Any], query_bundle: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return formula_grounding_stub(query_bundle, status='not_found', reason='result_not_object')

    out = formula_grounding_stub(query_bundle, status='not_found')

    status = obj.get('status')
    if status not in {'ok', 'not_found'}:
        has_formula_signal = any([
            obj.get('source_formula_candidate_id'),
            obj.get('supporting_span'),
            isinstance(obj.get('formula'), dict) and (
                obj.get('formula', {}).get('equation_sympy') or
                obj.get('formula', {}).get('equation_latex') or
                obj.get('formula', {}).get('variables') or
                obj.get('formula', {}).get('variable_semantics')
            ),
        ])
        status = 'ok' if has_formula_signal else 'not_found'
    out['status'] = status

    topk = query_bundle.get('topk_formula_candidates', []) or []
    cand = None
    raw_cid = obj.get('source_formula_candidate_id', '')
    if raw_cid:
        k = normalize_identifier(raw_cid)
        for c in query_bundle.get('formula_candidate_set_F_L', []):
            if normalize_identifier(c.get('candidate_id', '')) == k:
                cand = c
                break
    if cand is None and topk:
        cand = topk[0]

    formula_name = obj.get('formula_name', '')
    if isinstance(formula_name, dict):
        formula_name = formula_name.get('name', '')
    out['formula_name'] = str(formula_name or query_bundle.get('formula_name_used', ''))

    if status == 'ok' and cand is not None:
        out['source_formula_candidate_id'] = cand.get('candidate_id', '')
        sp = cand.get('source_page', {}) or {}
        out['source_page'] = {
            'title': sp.get('title', ''),
            'lang': sp.get('lang', ''),
            'source_type': sp.get('source_type', ''),
        }
        out['supporting_span'] = str(obj.get('supporting_span') or clip_text(cand.get('candidate_text', ''), 200))
    else:
        out['source_formula_candidate_id'] = ''
        out['source_page'] = {'title': '', 'lang': '', 'source_type': ''}
        out['supporting_span'] = ''

    formula = obj.get('formula', {}) if isinstance(obj.get('formula', {}), dict) else {}
    equation_sympy = str(formula.get('equation_sympy', '') or '')
    equation_latex = str(formula.get('equation_latex', '') or equation_sympy)
    variable_semantics = formula.get('variable_semantics', {}) if isinstance(formula.get('variable_semantics', {}), dict) else {}
    variables = formula.get('variables', []) if isinstance(formula.get('variables', []), list) else []
    if not variables and variable_semantics:
        variables = list(variable_semantics.keys())

    out['formula'] = {
        'equation_latex': equation_latex,
        'equation_sympy': equation_sympy,
        'variables': variables,
        'variable_semantics': variable_semantics,
        'formula_role_in_scene': str(formula.get('formula_role_in_scene', '') or ''),
    }

    try:
        out['confidence'] = float(obj.get('confidence', 0.0) or 0.0)
    except Exception:
        out['confidence'] = 0.0

    return out


def build_formula_schema_repair_user_prompt(step1_obj: Dict[str, Any], query_bundle: Dict[str, Any], raw_obj: Dict[str, Any]) -> str:
    return json.dumps({
        'task': 'repair formula_grounding output to exact schema',
        'step1_output': step1_obj,
        'query_bundle': compact_query_bundle_for_formula_prompt(query_bundle),
        'partial_formula_grounding': raw_obj,
        'required_schema': {
            'status': 'ok|not_found',
            'formula_name': 'string',
            'source_formula_candidate_id': 'string',
            'source_page': {'title': 'string', 'lang': 'en|zh', 'source_type': 'exact|search|wikidata'},
            'supporting_span': 'string',
            'formula': {
                'equation_latex': 'string',
                'equation_sympy': 'string',
                'variables': ['string'],
                'variable_semantics': {'var': 'meaning'},
                'formula_role_in_scene': 'string',
            },
            'confidence': 0.0,
        },
        'instruction': 'Return one JSON object only. If the partial object cannot be supported by the provided candidates, return a complete not_found object with all required fields present.',
    }, ensure_ascii=False, indent=2)


def run_step2_formula_grounding(step1_rows: List[Dict[str, Any]], llm: LocalLLM, languages: Sequence[str], out_dir: Path, save_intermediate: bool = False) -> List[Dict[str, Any]]:
    rows = []
    total = len(step1_rows)
    for row in step1_rows:
        idx = row["index"]
        step1_obj = row["step1"]
        hprint(f"Step2 {idx}/{total}")
        print(f"[Step2] original_formula_name={step1_obj['formula_name']['name']}")
        retrieval = build_retrieval_package(step1_obj, llm, languages)
        step1_effective = retrieval["step1_effective"]
        query_bundle = retrieval["query_bundle"]
        print(f"[Step2] effective_formula_name={step1_effective['formula_name']['name']}")
        print_page_preview("[Step2] final_retrieved_pages", query_bundle.get('retrieved_pages', []), limit=6)
        print_formula_preview("[Step2] final_topk_formula_candidates", query_bundle.get('topk_formula_candidates', []), limit=6)

        grounded_raw = llm.generate_json(FORMULA_SYSTEM_PROMPT, build_formula_user_prompt(step1_effective, query_bundle))
        print(f"[Step2] grounded_formula_raw={grounded_raw}")

        schema_issue = get_formula_schema_issue(grounded_raw)
        if schema_issue:
            print(f"[Step2] schema_issue_detected={schema_issue} -> repair once")
            repaired = llm.generate_json(
                FORMULA_SCHEMA_REPAIR_SYSTEM_PROMPT,
                build_formula_schema_repair_user_prompt(step1_effective, query_bundle, grounded_raw),
            )
            print(f"[Step2] grounded_formula_repaired={repaired}")
            grounded = normalize_formula_grounding_schema(repaired, query_bundle)
        else:
            grounded = normalize_formula_grounding_schema(grounded_raw, query_bundle)

        print(f"[Step2] grounded_formula_normalized={grounded}")
        ok, check = validate_formula_grounding(grounded, query_bundle)
        print(f"[Step2] formula_ok={ok} | formula_support_check={check}")
        rows.append({
            "index": idx,
            "step1": step1_obj,
            "step1_effective": step1_effective,
            "step1_validation": row.get("step1_validation", {}),
            "formula_name_regeneration": retrieval.get("formula_name_regeneration", {}),
            "query_bundle": query_bundle,
            "formula_grounding": grounded,
            "formula_support_check": check,
            "formula_ok": ok,
        })
    if save_intermediate:
        dump_json(rows, out_dir / "step2_formula_grounding.json")
    return rows


def build_param_user_prompt(step1_obj: Dict[str, Any], formula_obj: Dict[str, Any], query_bundle: Dict[str, Any]) -> str:
    compact_bundle = compact_query_bundle_for_param_prompt(query_bundle, formula_obj)
    return json.dumps({
        "task": "produce a calculation-ready parameter table",
        "step1_output": step1_obj,
        "grounded_formula": formula_obj,
        "query_bundle": compact_bundle,
        "instruction": (
            "Assign exactly one numeric value and one explicit unit to each variable in grounded_formula.formula.variables. "
            "You MUST copy every parameter name exactly, character by character, from grounded_formula.formula.variables. "
            "Do not invent extra parameter names. Do not omit any variable. "
            "For dimensionless quantities such as refractive index, coefficient, or ratio, use unit='dimensionless' instead of an empty string."
        ),
    }, ensure_ascii=False, indent=2)


def _is_numeric(x: Any) -> bool:
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, str):
        try:
            float(x)
            return True
        except Exception:
            return False
    return False


def _normalize_unit_text(unit: Any) -> str:
    s = str(unit or "").strip()
    if not s:
        return ""
    s_fold = s.casefold()
    if s_fold in {"dimensionless", "unitless", "no unit", "none", "1", "scalar"}:
        return "dimensionless"
    return s


def _is_dimensionless_variable(var_name: str, formula_obj: Dict[str, Any], row: Optional[Dict[str, Any]] = None) -> bool:
    formula = formula_obj.get("formula", {}) if isinstance(formula_obj, dict) else {}
    var_sem = formula.get("variable_semantics", {}) if isinstance(formula.get("variable_semantics", {}), dict) else {}
    sem = str(var_sem.get(var_name, ""))
    rationale = str((row or {}).get("rationale", ""))
    context = f"{sem} {rationale}".casefold()
    key = normalize_var_name(var_name)

    if any(k in context for k in ["dimensionless", "refractive index", "coefficient", "ratio"]):
        return True
    if key in {"n", "n1", "n2", "n_air", "n_water", "mu", "muk", "mus"}:
        return True
    return False


def normalize_parameter_units(obj: Dict[str, Any], formula_obj: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    if not isinstance(obj, dict):
        return False, {"ok": False, "reason": "parameter result is not an object"}
    table = obj.get("parameter_table", [])
    if not isinstance(table, list):
        return False, {"ok": False, "reason": "parameter_table is not a list"}

    changed = []
    for row in table:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        unit = _normalize_unit_text(row.get("unit", ""))
        if not unit and _is_dimensionless_variable(name, formula_obj, row):
            unit = "dimensionless"
            changed.append(name)
        row["unit"] = unit

    return True, {
        "ok": True,
        "reason": "parameter units normalized",
        "auto_filled_dimensionless": changed,
    }


_GREEK_MAP = {
    "θ": "theta", "Θ": "theta",
    "ρ": "rho",   "Ρ": "rho",
    "γ": "gamma", "Γ": "gamma",
    "δ": "delta", "Δ": "delta",
    "μ": "mu",    "Μ": "mu",
    "α": "alpha", "Α": "alpha",
    "β": "beta",  "Β": "beta",
    "λ": "lambda","Λ": "lambda",
    "φ": "phi",   "Φ": "phi",
    "ω": "omega", "Ω": "omega",
    "ν": "nu",    "Ν": "nu",
    "σ": "sigma", "Σ": "sigma",
    "τ": "tau",   "Τ": "tau",
}


def normalize_var_name(name: str) -> str:
    s = str(name or "").strip()

    for k, v in _GREEK_MAP.items():
        s = s.replace(k, v)

    s = s.replace("\\", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(" ", "")
    s = s.casefold()
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def align_parameter_table_to_formula_vars(obj: Dict[str, Any], formula_obj: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    if not isinstance(obj, dict):
        return False, {"ok": False, "reason": "parameter result is not an object"}

    table = obj.get("parameter_table", [])
    if not isinstance(table, list):
        return False, {"ok": False, "reason": "parameter_table is not a list"}

    required_vars = formula_obj.get("formula", {}).get("variables", []) or []
    if not required_vars:
        return False, {"ok": False, "reason": "formula.variables is empty; cannot align parameter names"}

    canon_map: Dict[str, str] = {}
    for v in required_vars:
        k = normalize_var_name(v)
        if k in canon_map and canon_map[k] != v:
            return False, {"ok": False, "reason": f"ambiguous normalized formula variable key: {k}"}
        canon_map[k] = v

    aligned_rows: Dict[str, Dict[str, Any]] = {}
    extras: List[str] = []

    for row in table:
        if not isinstance(row, dict):
            return False, {"ok": False, "reason": "one parameter row is not an object"}

        raw_name = row.get("name", "")
        key = normalize_var_name(raw_name)

        if key not in canon_map:
            extras.append(str(raw_name))
            continue

        canonical_name = canon_map[key]
        if canonical_name in aligned_rows:
            return False, {"ok": False, "reason": f"duplicate parameter mapped to formula variable: {canonical_name}"}

        new_row = dict(row)
        new_row["name"] = canonical_name
        aligned_rows[canonical_name] = new_row

    missing = [v for v in required_vars if v not in aligned_rows]
    if missing:
        return False, {"ok": False, "reason": f"missing parameters for formula variables: {missing}"}
    if extras:
        return False, {"ok": False, "reason": f"extra parameters not in formula variables: {extras}"}

    obj["parameter_table"] = [aligned_rows[v] for v in required_vars]

    return True, {
        "ok": True,
        "reason": "parameter names aligned to formula variables successfully",
        "aligned_names": required_vars,
    }


def validate_parameter_table(obj: Dict[str, Any], formula_obj: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    need = {"status", "is_calculation_ready", "parameter_table", "assumptions", "notes"}
    if not isinstance(obj, dict):
        return False, {"ok": False, "reason": "result is not an object"}

    miss = [k for k in need if k not in obj]
    if miss:
        return False, {"ok": False, "reason": f"missing keys: {miss}"}

    if obj.get("status") != "ready" or not obj.get("is_calculation_ready"):
        return False, {"ok": False, "reason": "status is not ready or is_calculation_ready is false"}

    table = obj.get("parameter_table", [])
    if not isinstance(table, list) or not table:
        return False, {"ok": False, "reason": "parameter_table is empty"}

    needed_vars = formula_obj.get("formula", {}).get("variables", [])
    names = [row.get("name") for row in table if isinstance(row, dict)]
    if names != needed_vars:
        return False, {
            "ok": False,
            "reason": f"parameter names are not strictly aligned with formula variables order: needed={needed_vars}, got={names}"
        }

    for row in table:
        if not isinstance(row, dict):
            return False, {"ok": False, "reason": "one parameter row is not an object"}

        for k in ["name", "value", "unit", "rationale"]:
            if k not in row:
                return False, {"ok": False, "reason": f"parameter row missing key: {k}"}

        if not _is_numeric(row.get("value")):
            return False, {"ok": False, "reason": f"value is not numeric for {row.get('name')}"}

        unit_text = _normalize_unit_text(row.get("unit", ""))
        if not unit_text:
            if _is_dimensionless_variable(str(row.get("name", "")), formula_obj, row):
                row["unit"] = "dimensionless"
            else:
                return False, {"ok": False, "reason": f"unit is empty for {row.get('name')}"}
        else:
            row["unit"] = unit_text

    return True, {"ok": True, "reason": "calculation-ready parameter table is structurally valid"}


def run_step3_parameter_reasoning(step2_rows: List[Dict[str, Any]], llm: LocalLLM, out_dir: Path, save_intermediate: bool = False) -> List[Dict[str, Any]]:
    rows = []
    total = len(step2_rows)

    for row in step2_rows:
        idx = row["index"]
        step1_obj = row.get("step1_effective") or row["step1"]
        formula_obj = row["formula_grounding"]
        query_bundle = row["query_bundle"]

        hprint(f"Step3 {idx}/{total}")
        print(f"[Step3] formula_name={formula_obj.get('formula_name', '')} | variables={formula_obj.get('formula', {}).get('variables', [])}")

        formula_ok = bool(row.get("formula_ok"))
        support_ok = bool((row.get("formula_support_check") or {}).get("ok"))

        if not formula_ok or not support_ok:
            skip_reason = {
                "formula_ok": formula_ok,
                "formula_support_check": row.get("formula_support_check", {}),
            }
            print(f"[Step3] skipped due to upstream step2 failure -> {skip_reason}")

            skipped_params = {
                "status": "skipped",
                "is_calculation_ready": False,
                "parameter_table": [],
                "assumptions": [],
                "notes": "Skipped because step2 formula grounding/support check did not pass."
            }
            check = {
                "ok": False,
                "reason": "upstream_step2_failed"
            }

            rows.append({
                "index": idx,
                "step1": row.get("step1"),
                "step1_effective": step1_obj,
                "formula_name_regeneration": row.get("formula_name_regeneration", {}),
                "query_bundle": query_bundle,
                "formula_grounding": formula_obj,
                "formula_support_check": row.get("formula_support_check", {}),
                "calculation_ready_parameters": skipped_params,
                "parameter_check": check,
                "parameter_ok": False,
            })
            continue

        params = llm.generate_json(
            PARAM_SYSTEM_PROMPT,
            build_param_user_prompt(step1_obj, formula_obj, query_bundle)
        )
        print(f"[Step3] params_raw={params}")

        align_ok, align_info = align_parameter_table_to_formula_vars(params, formula_obj)
        print(f"[Step3] align_ok={align_ok} | align_info={align_info}")

        if not align_ok:
            rows.append({
                "index": idx,
                "step1": row.get("step1"),
                "step1_effective": step1_obj,
                "formula_name_regeneration": row.get("formula_name_regeneration", {}),
                "query_bundle": query_bundle,
                "formula_grounding": formula_obj,
                "formula_support_check": row.get("formula_support_check", {}),
                "calculation_ready_parameters": params,
                "parameter_check": align_info,
                "parameter_ok": False,
            })
            continue

        unit_ok, unit_info = normalize_parameter_units(params, formula_obj)
        print(f"[Step3] unit_normalize_ok={unit_ok} | unit_info={unit_info}")

        ok, check = validate_parameter_table(params, formula_obj)
        print(f"[Step3] parameter_ok={ok} | parameter_check={check}")

        rows.append({
            "index": idx,
            "step1": row.get("step1"),
            "step1_effective": step1_obj,
            "formula_name_regeneration": row.get("formula_name_regeneration", {}),
            "query_bundle": query_bundle,
            "formula_grounding": formula_obj,
            "formula_support_check": row.get("formula_support_check", {}),
            "calculation_ready_parameters": params,
            "parameter_check": check,
            "parameter_ok": ok,
        })

    if save_intermediate:
        dump_json(rows, out_dir / "step3_calculation_ready_parameters.json")
    return rows


def build_ppd_input_item(step3_row: Dict[str, Any]) -> Dict[str, Any]:
    step1_original = step3_row.get("step1", {}) or {}
    step1_effective = step3_row.get("step1_effective", {}) or step1_original
    formula_obj = step3_row.get("formula_grounding", {}) or {}
    formula_core = formula_obj.get("formula", {}) if isinstance(formula_obj.get("formula", {}), dict) else {}
    params_obj = step3_row.get("calculation_ready_parameters", {}) or {}

    return {
        "index": step3_row.get("index"),
        "input_text": step1_effective.get("input_text", ""),
        "category": step1_effective.get("category", ""),
        "category_name": step1_effective.get("category_name", ""),
        "physical_law": step1_effective.get("physical_law", {}),
        "grounded_formula": {
            "formula_name": formula_obj.get("formula_name", ""),
            "equation_latex": formula_core.get("equation_latex", ""),
            "equation_sympy": formula_core.get("equation_sympy", ""),
            "variables": formula_core.get("variables", []),
            "variable_semantics": formula_core.get("variable_semantics", {}),
            "formula_role_in_scene": formula_core.get("formula_role_in_scene", ""),
        },
        "parameter_table": params_obj.get("parameter_table", []),
        "ready_for_ppd": bool(step3_row.get("parameter_ok")),
    }


def build_ppd_input_payload(step3_rows: List[Dict[str, Any]], include_failed_items: bool = False) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for row in step3_rows:
        if not include_failed_items and not bool(row.get("parameter_ok")):
            continue
        items.append(build_ppd_input_item(row))

    return {
        "module": "Physics Formula Grounding",
        "for_module": "Physics Phenomena Decomposition",
        "num_items": len(items),
        "items": items,
    }


def save_ppd_input_payload(step3_rows: List[Dict[str, Any]], out_dir: Path, filename: str = "pfg_minimal_for_ppd.json", include_failed_items: bool = False) -> Dict[str, Any]:
    payload = build_ppd_input_payload(step3_rows, include_failed_items=include_failed_items)
    dump_json(payload, out_dir / filename)
    return payload


def default_prompt_json_path() -> Path:
    return Path(__file__).resolve().parent / "prompt.json"


def load_prompts(args: argparse.Namespace) -> List[str]:
    prompt_path = Path(args.prompts_json) if args.prompts_json else default_prompt_json_path()
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"prompt.json not found: {prompt_path}. "
            "Please place prompt.json in the same directory as pfg.py, "
            "or pass --prompts_json explicitly."
        )
    return list(iter_prompts_from_json(prompt_path))

def load_step1_rows(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("step1_json must be a list")
    return data


def run_all_from_prompts(
    prompts: List[str],
    llm: LocalLLM,
    languages: Sequence[str],
    out_dir: Path,
    save_intermediate: bool = False,
    final_json_name: str = "pfg_minimal_for_ppd.json",
    include_failed_items: bool = False,
) -> Dict[str, Any]:
    step1_rows = run_step1_generation(prompts, llm, out_dir, languages, save_intermediate=save_intermediate)
    step2_rows = run_step2_formula_grounding(step1_rows, llm, languages, out_dir, save_intermediate=save_intermediate)
    step3_rows = run_step3_parameter_reasoning(step2_rows, llm, out_dir, save_intermediate=save_intermediate)
    payload = save_ppd_input_payload(
        step3_rows,
        out_dir,
        filename=final_json_name,
        include_failed_items=include_failed_items,
    )
    return {
        "pipeline": "step1 + step2_query_bundle_formula_candidate_set_formula_grounding + step3_calculation_ready_parameters",
        "final_json": str(out_dir / final_json_name),
        "num_items": payload.get("num_items", 0),
    }



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PECR full pipeline, paper-aligned single-file version (compact final-json mode)")
    p.add_argument("--mode", choices=["step1", "step2", "step3", "all"], default="all")
    p.add_argument(
    "--prompts_json",
    type=str,
    default=str(Path(__file__).resolve().parent / "prompt.json"),
    help="Path to prompt.json. Default: prompt.json in the same directory as pfg.py"
    )
    p.add_argument("--step1_json", type=str, default="", help="Existing step1_verified.json")
    p.add_argument("--step2_json", type=str, default="", help="Existing step2_formula_grounding.json")
    p.add_argument("--out_dir", type=str, default="./pecr_outputs_pfg_final_only")
    p.add_argument("--final_json_name", type=str, default="pfg_minimal_for_ppd.json", help="Single minimal json filename for downstream PPD")
    p.add_argument("--save_intermediate", action="store_true", help="Also save step1/step2/step3 intermediate json files")
    p.add_argument("--include_failed_items", action="store_true", help="Include failed rows in the final compact json")
    p.add_argument("--languages", type=str, default=",".join(LANGS), help="Comma-separated languages, e.g. en,zh")
    p.add_argument("--model_path", type=str, default=DEFAULT_SNAP_DIR)
    p.add_argument("--device", type=str, default="cuda:0", help="Model device, e.g. cuda:0, cuda:1, cpu, or auto")
    p.add_argument("--quiet", action="store_true", help="Reduce terminal logging")
    p.add_argument("--debug_api", action="store_true", help="Print raw HTTP request/response heads")
    return p


def main() -> None:
    global VERBOSE, DEBUG_API
    args = build_argparser().parse_args()
    VERBOSE = not args.quiet
    DEBUG_API = bool(args.debug_api)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    langs = tuple(x.strip() for x in args.languages.split(",") if x.strip())

    print(f"[Init] mode={args.mode} | out_dir={out_dir} | languages={langs} | verbose={VERBOSE} | debug_api={DEBUG_API}")
    print(f"[Init] Loading local model from: {args.model_path}")
    print(f"[Init] device={args.device}")
    llm = LocalLLM(args.model_path, device=args.device)
    print("[Init] Model ready.")

    if args.mode == "step1":
        prompts = load_prompts(args)
        run_step1_generation(prompts, llm, out_dir, langs, save_intermediate=True)
        return

    if args.mode == "step2":
        rows = load_step1_rows(args.step1_json)
        if not rows:
            raise ValueError("Please provide --step1_json for mode=step2")
        run_step2_formula_grounding(rows, llm, langs, out_dir, save_intermediate=True)
        return

    if args.mode == "step3":
        if not args.step2_json:
            raise ValueError("Please provide --step2_json for mode=step3")
        rows = load_json(args.step2_json)
        if not isinstance(rows, list):
            raise ValueError("step2_json must be a list")
        run_step3_parameter_reasoning(rows, llm, out_dir, save_intermediate=True)
        return

    if args.step1_json:
        rows = load_step1_rows(args.step1_json)
        step2 = run_step2_formula_grounding(rows, llm, langs, out_dir, save_intermediate=args.save_intermediate)
        step3 = run_step3_parameter_reasoning(step2, llm, out_dir, save_intermediate=args.save_intermediate)
        payload = save_ppd_input_payload(
            step3,
            out_dir,
            filename=args.final_json_name,
            include_failed_items=args.include_failed_items,
        )
        print(json.dumps({
            "final_json": str(out_dir / args.final_json_name),
            "num_items": payload.get("num_items", 0),
        }, ensure_ascii=False, indent=2))
        return

    prompts = load_prompts(args)
    summary = run_all_from_prompts(
        prompts,
        llm,
        langs,
        out_dir,
        save_intermediate=args.save_intermediate,
        final_json_name=args.final_json_name,
        include_failed_items=args.include_failed_items,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
