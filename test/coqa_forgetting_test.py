from pathlib import Path
import os, warnings, re, sys, gc, argparse, random, string
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as hf_logging
from peft import PeftModel
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_error as ds_set_verbosity_error

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CSV_PATH = Path("coqa_debug_results.csv")
VARIANT_DIRS = ["8bits_r8", "4bits_r4", "8bits_r8_dora", "8bits_r8_rslora"]

DEFAULT_SEED = 42
DEFAULT_N_DIALOGS = 50
DEFAULT_MAX_TURNS = 10
DEFAULT_HISTORY_TURNS = 2

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
ds_set_verbosity_error()

def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, gold_list: List[str]) -> float:
    pn = normalize_answer(prediction)
    for gt in gold_list:
        if pn == normalize_answer(gt):
            return 1.0
    return 0.0

def f1_score(prediction: str, gold_list: List[str]) -> float:
    ptoks = normalize_answer(prediction).split()
    if not ptoks:
        return 0.0
    best = 0.0
    for gt in gold_list:
        gtoks = normalize_answer(gt).split()
        if not gtoks:
            continue
        common = set(ptoks) & set(gtoks)
        if not common:
            continue
        prec = len(common) / len(ptoks)
        rec = len(common) / len(gtoks)
        f1 = 2 * (prec * rec) / (prec + rec)
        best = max(best, f1)
    return best

def _as_text(x, prefer_keys=("input_text", "text", "span_text", "answer")) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in prefer_keys:
            v = x.get(k)
            if isinstance(v, str):
                return v
    return str(x)

def get_questions_list(ex: Dict[str, Any]) -> List[str]:
    q = ex.get("questions")
    if isinstance(q, list):
        return [_as_text(item) for item in q]
    if isinstance(q, dict):
        qit = q.get("input_text")
        if isinstance(qit, list):
            return [_as_text(item) for item in qit]
        if isinstance(qit, str):
            return [qit]
    return []

def get_answers_list_primary(ex: Dict[str, Any]) -> List[str]:
    a = ex.get("answers")
    if isinstance(a, dict):
        it = a.get("input_text")
        if isinstance(it, list):
            return [_as_text(s) for s in it]
        if isinstance(it, str):
            return [it]
    if isinstance(a, list):
        out = []
        for item in a:
            out.append(_as_text(item))
        return out
    return []

def get_additional_answers_turn(ex: Dict[str, Any], turn_idx: int) -> List[str]:
    """
    Accept shapes like:
        additional_answers = {
          "worker1": {"input_text": [...], "answer_start": [...], "answer_end": [...]},
          "worker2": {"input_text": [...]}, ...
        }
    or, rarely, lists/strings aligned by turn.
    """
    adds = ex.get("additional_answers", {})
    golds: List[str] = []
    if isinstance(adds, dict):
        for _, v in adds.items():
            if isinstance(v, dict):
                it = v.get("input_text")
                if isinstance(it, list) and 0 <= turn_idx < len(it):
                    golds.append(_as_text(it[turn_idx]))
                elif isinstance(it, str) and turn_idx == 0:
                    golds.append(_as_text(it))
            elif isinstance(v, list) and 0 <= turn_idx < len(v):
                golds.append(_as_text(v[turn_idx]))
            elif isinstance(v, str) and turn_idx == 0:
                golds.append(_as_text(v))
    return golds

def coqa_gold_answers(ex: Dict[str, Any], turn_idx: int) -> List[str]:
    golds = []
    prim = get_answers_list_primary(ex)
    if 0 <= turn_idx < len(prim):
        golds.append(prim[turn_idx])
    golds.extend(get_additional_answers_turn(ex, turn_idx))
    seen = set(); out = []
    for g in golds:
        if g not in seen:
            seen.add(g); out.append(g)
    return out if out else [""]

class HFLLM:
    def __init__(self, model_name_or_path: str, adapter_path: str | None = None):
        self._name = model_name_or_path if adapter_path is None else Path(adapter_path).name
        self._adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def generate(self, prompt: str) -> str:
        prompt = f"Answer with just the answer (no explanation or punctuation).\n{prompt}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        in_len = inputs['input_ids'].shape[1]
        gen = outputs[0][in_len:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        text = text.split("\n", 1)[0].strip()
        text = re.sub(r'^(Answer:|A:|The answer is:?)\s*', '', text, flags=re.IGNORECASE)
        text = re.split(r'[.:;()\-\u2013\u2014]', text, maxsplit=1)[0].strip().strip('"\'')

        text = re.sub(r'\s*(?:[.?!])$', '', text)
        return text

    def get_model_name(self) -> str:
        return self._name

def load_coqa_validation(n_dialogs: int, seed: int):
    """
    Loads CoQA validation split; normalize later with safe extractors.
    Using explicit repo id to avoid schema drift.
    """
    ds = load_dataset("stanfordnlp/coqa", split="validation")
    ds = ds.shuffle(seed=seed)
    if n_dialogs and n_dialogs < len(ds):
        ds = ds.select(range(n_dialogs))
    return ds

def build_prompt(story: str,
                 q_list: List[str],
                 a_primary: List[str],
                 turn_idx: int,
                 history_turns: int) -> str:
    """
    Build a prompt with story + limited QA history from primary answers.
    """
    q_text = q_list[turn_idx] if 0 <= turn_idx < len(q_list) else ""
    start_hist = max(0, turn_idx - history_turns)
    history_lines = []
    for i in range(start_hist, turn_idx):
        qi = q_list[i] if i < len(q_list) else ""
        ai = a_primary[i] if i < len(a_primary) else ""
        history_lines.append(f"Q{i+1}: {qi}\nA{i+1}: {ai}")
    history_block = ("\n".join(history_lines) + "\n") if history_lines else ""

    prompt = (
        f"Story:\n{story}\n\n"
        f"History:\n{history_block}"
        f"Question: {q_text}\n"
        f"Answer:"
    )
    return prompt

def find_adapters(finetuned_root: Path) -> List[Path]:
    paths: List[Path] = []
    for variant_dir in VARIANT_DIRS:
        vpath = finetuned_root / variant_dir
        if not vpath.exists():
            continue
        for p in vpath.iterdir():
            if not p.is_dir():
                continue
            n = p.name.lower()
            if "finer" not in n:
                continue
            if "_llama_3_1_8b_" not in n:
                continue
            if (p / "adapter_model.safetensors").exists():
                paths.append(p)
    return sorted(paths, key=lambda p: p.name)

def eval_coqa_debug(model: HFLLM, ds, max_turns: int, history_turns: int) -> Tuple[float, float, int]:
    exact_matches, f1_scores = [], []
    n_items = 0

    print(f"\n=== Evaluating {model.get_model_name()} on {len(ds)} dialogs (up to {max_turns} turns each) ===\n")
    for d_idx, ex in enumerate(tqdm(ds, total=len(ds), leave=False)):
        story = ex.get("story", "")
        q_list = get_questions_list(ex)
        a_primary = get_answers_list_primary(ex)
        n_turns = min(len(q_list), len(a_primary), max_turns)
        if n_turns == 0:
            print(f"[WARN] dialog {d_idx}: zero turns | "
                  f"len(questions)={len(q_list)} len(primary_answers)={len(a_primary)}")
        for t in range(n_turns):
            gold_list = coqa_gold_answers(ex, t)
            prompt = build_prompt(story, q_list, a_primary, t, history_turns)
            pred = model.generate(prompt)

            em = exact_match_score(pred, gold_list)
            f1 = f1_score(pred, gold_list)
            exact_matches.append(em)
            f1_scores.append(f1)
            n_items += 1

            q_text = q_list[t]
            gi = f"{d_idx:02d}-{t:02d}"
            print(f"[{gi}]\nQ: {q_text}\nPred: {pred}\nGold ({len(gold_list)}): " +
                  " | ".join(gold_list[:10]) + ("" if len(gold_list) <= 10 else " | ..."))
            print(f"EM={int(em)} F1={f1:.3f}\n")

            r_em = sum(exact_matches) / len(exact_matches)
            r_f1 = sum(f1_scores) / len(f1_scores)
            print(f"[running] {len(exact_matches)} QA pairs  EM={r_em:.3f}  F1={r_f1:.3f}\n")

    em_score = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"Summary for {model.get_model_name()}: EM={em_score:.3f}, F1={f1_avg:.3f} over {n_items} QA pairs")
    return em_score, f1_avg, n_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogs", type=int, default=DEFAULT_N_DIALOGS, help="Number of validation dialogs to evaluate (default 10)")
    parser.add_argument("--max_turns", type=int, default=DEFAULT_MAX_TURNS, help="Max turns per dialog to evaluate (default 10)")
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY_TURNS, help="How many previous QA pairs to include (default 2)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed (default 42)")
    parser.add_argument("--save", type=Path, default=CSV_PATH, help="CSV path for summary results")
    parser.add_argument("--adapters_root", type=Path, default=Path("../lora_adapters").resolve(),
                        help="Root dir containing adapter variant subdirs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading CoQA validation ({args.dialogs} dialogs, seed={args.seed})…")
    ds = load_coqa_validation(args.dialogs, args.seed)
    print(f"Loaded {len(ds)} dialogs")

    adapter_paths = find_adapters(args.adapters_root)
    if adapter_paths:
        print(f"Found {len(adapter_paths)} finer adapters")
    else:
        print("Found 0 finer adapters")

    results: List[Dict[str, Any]] = []

    print("\n[base] evaluating base model first")
    try:
        base = HFLLM(BASE_MODEL_NAME)
        em, f1, n_eval = eval_coqa_debug(base, ds, args.max_turns, args.history)
        results.append({
            "model_name": "base_model",
            "adapter_path": None,
            "task": "base",
            "variant": "base",
            "n_pairs": n_eval,
            "em_score": em,
            "f1_score": f1
        })
        del base
        torch.cuda.empty_cache(); gc.collect()
        pd.DataFrame(results).to_csv(args.save, index=False)
        print(f"Saved → {args.save}")
    except Exception as e:
        print(f"Error on base model: {e}")
        results.append({
            "model_name": "base_model",
            "adapter_path": None,
            "task": "base",
            "variant": "base",
            "n_pairs": 0, "em_score": 0.0, "f1_score": 0.0
        })
        pd.DataFrame(results).to_csv(args.save, index=False)

    for i, ap in enumerate(tqdm(adapter_paths, desc="Adapters")):
        print(f"[{i+1}/{len(adapter_paths)}] {ap.name}")
        try:
            model = HFLLM(BASE_MODEL_NAME, str(ap))
            em, f1, n_eval = eval_coqa_debug(model, ds, args.max_turns, args.history)
            results.append({
                "model_name": ap.name,
                "adapter_path": str(ap),
                "task": "finer",
                "variant": ap.parent.name,
                "n_pairs": n_eval,
                "em_score": em,
                "f1_score": f1
            })
            del model
            torch.cuda.empty_cache(); gc.collect()
            pd.DataFrame(results).to_csv(args.save, index=False)
            print(f"Saved → {args.save}")
        except Exception as e:
            print(f"Error on {ap}: {e}")
            results.append({
                "model_name": ap.name,
                "adapter_path": str(ap),
                "task": "finer",
                "variant": ap.parent.name,
                "n_pairs": 0, "em_score": 0.0, "f1_score": 0.0
            })
            pd.DataFrame(results).to_csv(args.save, index=False)
            torch.cuda.empty_cache(); gc.collect()

    try:
        df = pd.read_csv(args.save)
        print(df)
    except Exception:
        pass

if __name__ == "__main__":
    main()
