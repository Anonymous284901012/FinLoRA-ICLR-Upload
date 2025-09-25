from pathlib import Path
import os, warnings, re, torch, pandas as pd, random, string, numpy as np, gc
from typing import List, Dict, Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as hf_logging
from peft import PeftModel
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_error as ds_set_verbosity_error

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TEST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = Path("triviaqa_forgetting_results.csv")

# Only FINER adapters
VARIANT_DIRS = ["8bits_r8", "4bits_r4", "8bits_r8_dora", "8bits_r8_rslora"]

# Exactly 500 questions with seed=42
N_SAMPLES = 500
SPLIT_EXPR_PRIMARY = f"validation.shuffle(seed=42)[:{N_SAMPLES}]"
SPLIT_EXPR_FALLBACK = f"validation[:{N_SAMPLES}]"

# Repro
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Suppress warnings/log spam
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

def exact_match_score(prediction: str, ground_truth_list: List[str]) -> float:
    pn = normalize_answer(prediction)
    for gt in ground_truth_list:
        if pn == normalize_answer(gt):
            return 1.0
    return 0.0

def f1_score(prediction: str, ground_truth_list: List[str]) -> float:
    ptoks = normalize_answer(prediction).split()
    if not ptoks:
        return 0.0
    best_f1 = 0.0
    for gt in ground_truth_list:
        gtoks = normalize_answer(gt).split()
        if not gtoks:
            continue
        common = set(ptoks) & set(gtoks)
        if not common:
            continue
        precision = len(common) / len(ptoks)
        recall = len(common) / len(gtoks)
        f1 = 2 * (precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1

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
        # Make answers terse to help EM/F1
        prompt = f"Answer with just the answer (no explanation or punctuation).\n{prompt}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,        # short answer only
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

        # final cleanup
        text = re.sub(r'\s*(?:[.?!])$', '', text)
        return text

    def get_model_name(self) -> str:
        return self._name

def load_triviaqa_dataset():
    print(f"Loading TriviaQA dataset ({N_SAMPLES}, seed=42)…")
    try:
        return load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=SPLIT_EXPR_PRIMARY)
    except Exception:
        return load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=SPLIT_EXPR_FALLBACK)

def evaluate_triviaqa(model: HFLLM, dataset) -> tuple[float, float]:
    exact_matches, f1_scores = [], []
    for example in tqdm(dataset, total=len(dataset), desc=f"Eval {model.get_model_name()}", leave=False):
        question = example.get('question', '')
        if not question:
            continue
        answer_list: List[str] = []
        ans = example.get('answer', {})
        if isinstance(ans, dict):
            if 'normalized_aliases' in ans and isinstance(ans['normalized_aliases'], list):
                answer_list.extend(a for a in ans['normalized_aliases'] if isinstance(a, str))
            if isinstance(ans.get('value'), str):
                answer_list.append(ans['value'])
        if not answer_list:
            continue
        pred = model.generate(f"Question: {question}\nAnswer:")
        em = exact_match_score(pred, answer_list)
        f1 = f1_score(pred, answer_list)
        exact_matches.append(em)
        f1_scores.append(f1)
    em_score = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return em_score, f1_avg

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
            if "_llama_3_1_8b_" not in n:  # check lowercased token
                continue
            if (p / "adapter_model.safetensors").exists():
                paths.append(p)
    return sorted(paths, key=lambda p: p.name)

def save_results_incrementally(results: List[Dict[str, Any]], csv_path: Path):
    if results:
        pd.DataFrame(results).to_csv(csv_path, index=False)

def main():
    print("Starting TriviaQA Forgetting Test")
    print("=" * 50)

    triviaqa_dataset = load_triviaqa_dataset()
    print(f"Loaded {len(triviaqa_dataset)} examples")

    finetuned_root = Path("../lora_adapters").resolve()
    adapter_paths = find_adapters(finetuned_root)
    if not adapter_paths:
        print("No finer adapters found!")
    else:
        print(f"Found {len(adapter_paths)} finer adapters to test")

    results: List[Dict[str, Any]] = []

    # ---- Base FIRST
    print("\nEvaluating base model first…")
    try:
        base_model = HFLLM(BASE_MODEL_NAME)
        em_score, f1_sc = evaluate_triviaqa(base_model, triviaqa_dataset)
        print(f"[base_model] EM={em_score:.9f}, F1={f1_sc:.9f}")  # <-- current EM/F1
        results.append({
            'model_name': 'base_model',
            'adapter_path': 'None',
            'task': 'base',
            'variant': 'base',
            'em_score': em_score,
            'f1_score': f1_sc
        })
        del base_model
        torch.cuda.empty_cache(); gc.collect()
        save_results_incrementally(results, CSV_PATH)
    except Exception as e:
        print(f"Error testing base model: {e}")
        results.append({
            'model_name': 'base_model',
            'adapter_path': 'None',
            'task': 'base',
            'variant': 'base',
            'em_score': 0.0,
            'f1_score': 0.0
        })
        save_results_incrementally(results, CSV_PATH)

    # ---- Adapters next
    for adapter_path in tqdm(adapter_paths, desc="Adapters"):
        adapter_name = adapter_path.name
        variant_name = adapter_path.parent.name
        try:
            model = HFLLM(BASE_MODEL_NAME, str(adapter_path))
            em_score, f1_sc = evaluate_triviaqa(model, triviaqa_dataset)
            print(f"[{adapter_name}] EM={em_score:.9f}, F1={f1_sc:.9f}")  # <-- current EM/F1
            results.append({
                'model_name': adapter_name,
                'adapter_path': str(adapter_path),
                'task': 'finer',
                'variant': variant_name,
                'em_score': em_score,
                'f1_score': f1_sc
            })
            del model
            torch.cuda.empty_cache(); gc.collect()
            save_results_incrementally(results, CSV_PATH)
        except Exception as e:
            print(f"Error testing {adapter_name}: {e}")
            results.append({
                'model_name': adapter_name,
                'adapter_path': str(adapter_path),
                'task': 'finer',
                'variant': variant_name,
                'em_score': 0.0,
                'f1_score': 0.0
            })
            save_results_incrementally(results, CSV_PATH)
            torch.cuda.empty_cache(); gc.collect()

    print(f"\nFinal results saved to: {CSV_PATH}")
    try:
        print(pd.read_csv(CSV_PATH))
    except Exception:
        pass

if __name__ == "__main__":
    main()