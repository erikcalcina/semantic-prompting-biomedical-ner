"""Entity extraction with an LLM: zero-shot, few-shot, or with a fine-tuned LoRA adapter.

Writes a JSON list with one {"text": ..., "labels": [{"text": ..., "label": ...}]} per sentence.

    python scripts/predict.py --model_name meta-llama/Llama-3.1-8B-Instruct \
        --dataset data/test.llm.json --dataset_name maccrobat --semantic \
        --prompt_type few_shot_prompting --temperature 0.2 --output_file out.json
"""
import re
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompts"))
from prompt_builder import Prompts
from label_sets import DATASET_NAMES, LABEL_SETS
from model_utils import DEEPSEEK_ASSISTANT, is_deepseek, load_model, load_tokenizer, merge_system_into_user, supports_system_role

PROMPT_TYPES = ["prompt_only", "few_shot_prompting", "instruction_prompt", "conversational_prompt"]


def prepare_model_and_tokenizer(model_name: str, adapter_name: str = None):
    """Loads the 4-bit base model, optionally with a LoRA adapter, and its tokenizer."""
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, adapter_name)
    print("Model used:", model_name, "| adapter:", adapter_name)
    return model, tokenizer


def build_messages(prompts: Prompts, prompt_type: str, labels, text: str):
    if prompt_type == "prompt_only":
        return prompts.create_prompt_only_prompt(labels, text)
    if prompt_type == "few_shot_prompting":
        return prompts.create_few_shot_prompt(labels, text)
    if prompt_type == "instruction_prompt":
        return [{"role": "user", "content": prompts.create_instruction_message(labels, text)["prompt"]}]
    if prompt_type == "conversational_prompt":
        return prompts.create_conversational_message(labels, text)
    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def build_input_ids(messages, tokenizer, prompt_type: str):
    if not supports_system_role(tokenizer):
        messages = merge_system_into_user(messages)
    if prompt_type == "instruction_prompt" and is_deepseek(tokenizer):
        # DeepSeek adapters were trained on "<｜User｜>prompt<｜Assistant｜>answer"; the default
        # generation prompt would add "<think>\n", so end the prompt at "<｜Assistant｜>" instead.
        messages = [dict(m) for m in messages]
        messages[-1]["content"] += DEEPSEEK_ASSISTANT
        return tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True)


#: End-of-turn markers some chat templates use instead of the tokenizer's eos_token. When a model's
#: generation_config does not list them (TxGemma lists only <eos>), a fine-tuned model writes its
#: answer, emits the marker and keeps generating until max_new_tokens.
TURN_END_TOKENS = ["<end_of_turn>", "<|eot_id|>", "<|im_end|>"]


def stop_token_ids(model, tokenizer):
    eos = model.generation_config.eos_token_id
    ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id, tokenizer.pad_token_id}
    return {i for i in ids if i is not None}


def turn_end_token_ids(tokenizer):
    """Ids of the end-of-turn markers present in this tokenizer's vocabulary."""
    vocab = tokenizer.get_vocab()
    return [vocab[t] for t in TURN_END_TOKENS if t in vocab]


def generate_batch(model, tokenizer, prompt_ids, idx, temperature, top_p, max_new_tokens, stops, eos_ids=None):
    """Generates for the prompts in idx; returns (response, new-token count, hit the cap) per prompt."""
    width = max(len(prompt_ids[k]) for k in idx)
    input_ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(prompt_ids[k])) + prompt_ids[k] for k in idx], device=model.device)
    attention_mask = torch.tensor([[0] * (width - len(prompt_ids[k])) + [1] * len(prompt_ids[k]) for k in idx], device=model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            **({"eos_token_id": eos_ids} if eos_ids else {}),
        )
    results = []
    for output in output_ids:
        new = output[width:].tolist()
        n_new = next((j for j, t in enumerate(new) if t in stops), len(new))
        results.append((tokenizer.decode(output[width:], skip_special_tokens=True), n_new, n_new >= max_new_tokens))
    return results


def generate_responses(model, tokenizer, prompt_ids, temperature: float, top_p: float, max_new_tokens: int,
                       batch_size: int = 1, verbose: bool = False, stop_at_turn_end: bool = False):
    """Generates one response per tokenized prompt.

    batch_size 1 is the original one-prompt-at-a-time loop. Larger batches group prompts of
    similar length and left-pad them, which is much faster on a GPU. A batch that runs out of
    GPU memory is retried in halves.

    Returns (responses, new-token counts, whether each hit max_new_tokens).
    """
    responses, n_tokens, hit_cap = [None] * len(prompt_ids), [0] * len(prompt_ids), [False] * len(prompt_ids)
    stops = stop_token_ids(model, tokenizer)
    eos_ids = None
    if stop_at_turn_end:
        eos_ids = sorted(stops | set(turn_end_token_ids(tokenizer)))
        stops = set(eos_ids)
        print("Stopping also at end-of-turn markers; eos_token_id =", eos_ids)
    order = list(range(len(prompt_ids)))
    if batch_size > 1:
        order.sort(key=lambda k: len(prompt_ids[k]))
    for start in tqdm(range(0, len(order), batch_size), desc="generating", unit="batch"):
        pending = [order[start:start + batch_size]]
        while pending:
            idx = pending.pop()
            try:
                results = generate_batch(model, tokenizer, prompt_ids, idx, temperature, top_p, max_new_tokens, stops, eos_ids)
            except torch.cuda.OutOfMemoryError:
                if len(idx) == 1:
                    raise
                torch.cuda.empty_cache()
                print(f"Out of GPU memory with {len(idx)} prompts; retrying in halves.")
                pending += [idx[len(idx) // 2:], idx[:len(idx) // 2]]
                continue
            for k, (response, n_new, capped) in zip(idx, results):
                responses[k], n_tokens[k], hit_cap[k] = response, n_new, capped
                if verbose:
                    print(response)
    return responses, n_tokens, hit_cap


def parse_response(response: str):
    """Parses the first JSON list of entities out of the generated text."""
    try:
        # Use a regex pattern to find JSON-like structures
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if json_match:
            # Replace single quotes with double quotes for JSON compatibility
            json_str = json_match.group().replace("'", '"')
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    # Fallback for cases where strict JSON parsing fails
    try:
        for line in response.splitlines():
            if "[" in line and "{" in line:
                json_str = line.strip().replace("'", '"')
                return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return []


def clean_invalid_surrogates(obj):
    if isinstance(obj, str):
        return obj.encode('utf-16', 'surrogatepass').decode('utf-16', 'replace')
    if isinstance(obj, list):
        return [clean_invalid_surrogates(item) for item in obj]
    if isinstance(obj, dict):
        return {key: clean_invalid_surrogates(value) for key, value in obj.items()}
    return obj


def main(args):
    with open(args.dataset, "r", encoding="utf8") as file:
        data = json.load(file)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    prompts = Prompts(semantic=args.semantic)
    # Zero-/few-shot ask for one label per generation; the instruction prompt asks for the full schema.
    per_label = args.prompt_type in ("prompt_only", "few_shot_prompting")
    labels = LABEL_SETS[args.dataset_name]["prompt_keys" if per_label else "all"]
    print(f"Setting: {'semantic' if args.semantic else 'baseline'} | prompt: {args.prompt_type} | labels: {labels}")

    model, tokenizer = prepare_model_and_tokenizer(args.model_name, args.adapter_name)

    # One generation per sentence and label (zero-/few-shot) or per sentence (instruction prompt).
    jobs = [(i, label) for i in range(len(data)) for label in labels] if per_label else [(i, labels) for i in range(len(data))]
    prompt_ids = [
        build_input_ids(build_messages(prompts, args.prompt_type, label, data[i]["text"]), tokenizer, args.prompt_type)[0].tolist()
        for i, label in jobs
    ]
    if args.seed is not None:
        set_seed(args.seed)
    responses, n_tokens, hit_cap = generate_responses(model, tokenizer, prompt_ids, args.temperature, args.top_p,
                                                      args.max_new_tokens, args.batch_size, args.verbose,
                                                      args.stop_at_turn_end)

    extracted = [[] for _ in data]
    unparsed = 0
    records = []
    for (i, label), response, n_new, capped in zip(jobs, responses, n_tokens, hit_cap):
        entities = parse_response(response)
        unparsed += entities == []
        if per_label:
            extracted[i].extend(entities)
        else:
            extracted[i] = entities
        records.append({"index": i, "label": label if per_label else None, "new_tokens": n_new, "hit_max_new_tokens": capped,
                        "parsed": entities != [], "response": response})
    print(f"{unparsed} of {len(responses)} responses had no valid JSON (or an empty list); "
          f"{sum(hit_cap)} hit max_new_tokens.")

    if args.responses_file:
        Path(args.responses_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.responses_file, "w", encoding="utf8") as f:
            for record in records:
                f.write(json.dumps(clean_invalid_surrogates(record), ensure_ascii=False) + "\n")

    predictions = [{"text": example["text"], "labels": entities} for example, entities in zip(data, extracted)]
    output_path = args.output_file if args.output_file.endswith('.json') else args.output_file + '.json'
    with open(output_path, "w", encoding="utf8") as json_file:
        json.dump(clean_invalid_surrogates(predictions), json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face base model.")
    parser.add_argument("--adapter_name", type=str, default=None, help="LoRA adapter directory (fine-tuned regime).")
    parser.add_argument("--dataset", type=str, required=True, help="Test dataset JSON.")
    parser.add_argument("--dataset_name", type=str, default="maccrobat", choices=DATASET_NAMES, help="Selects the label set.")
    parser.add_argument("--semantic", action="store_true", help="Use semantic label definitions instead of the plain descriptions.")
    parser.add_argument("--prompt_type", type=str, default="prompt_only", choices=PROMPT_TYPES,
                        help="prompt_only = zero-shot, few_shot_prompting = few-shot, instruction_prompt = fine-tuned.")
    parser.add_argument("--output_file", type=str, required=True, help="Where to write the predictions JSON.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p value for nucleus sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Generation length limit.")
    parser.add_argument("--batch_size", type=int, default=1, help="Prompts generated together; 1 = the original one-by-one loop.")
    parser.add_argument("--verbose", action="store_true", help="Print every generated response.")
    parser.add_argument("--responses_file", type=str, default=None,
                        help="Optional JSONL with every raw response, its new-token count and whether it parsed.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for sampling; unset = unseeded.")
    parser.add_argument("--stop_at_turn_end", action="store_true",
                        help="Also stop at the chat template's end-of-turn marker. Off by default (the paper's "
                             "behaviour): TxGemma and base Llama do not list it as eos, so they generate to the cap.")
    main(parser.parse_args())
