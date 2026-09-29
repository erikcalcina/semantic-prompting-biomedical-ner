"""QLoRA supervised fine-tuning for entity extraction, with baseline or semantic prompts.

    python scripts/train.py --model_name meta-llama/Llama-3.1-8B-Instruct \
        --dataset data/train.llm.json --dataset_name maccrobat --semantic \
        --output_dir runs/adapters/Llama-3.1-8B-Instruct
"""
import sys
import json
import inspect
from pathlib import Path
from typing import List
from functools import partial
from argparse import ArgumentParser

from peft import LoraConfig
from datasets import Dataset
from transformers import set_seed
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompts"))
from prompt_builder import Prompts
from label_sets import DATASET_NAMES, LABEL_SETS
from model_utils import load_model, load_tokenizer, response_template_for

LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'lm_head']


def training_data_individual_labels(dataset: List[dict], labels: List[str], prompts: Prompts, output_type: str = "instruction_training") -> Dataset:
    """Creates one training example per (text, label), each asking for a single label."""
    if output_type not in ["instruction_training", "conversational_training"]:
        raise ValueError(f"Unknown output_type: {output_type}")

    train_format_dataset = []
    for example in dataset:
        relevant_labels = [label for label in example["labels"] if label["label"] in labels]
        grouped_entities = {label: [item for item in relevant_labels if item['label'] == label] for label in labels}
        for label in labels:
            if output_type == "instruction_training":
                message = prompts.create_instruction_training_message_with_completion([label], example["text"], grouped_entities[label])
                train_format_dataset.append(message)
            elif output_type == "conversational_training":
                message = prompts.create_conversational_training_message_with_completion([label], example["text"], grouped_entities[label])
                train_format_dataset.append({"messages": message})
    return Dataset.from_list(train_format_dataset)


def training_data_combined_labels(dataset: List[dict], labels: List[str], prompts: Prompts, output_type: str = "instruction_training") -> Dataset:
    """Creates one training example per text, asking for all labels at once (the paper's setup)."""
    if output_type not in ["instruction_training", "conversational_training"]:
        raise ValueError(f"Unknown output_type: {output_type}")

    train_format_dataset = []
    for example in dataset:
        relevant_labels = [label for label in example["labels"] if label["label"] in labels]
        if output_type == "instruction_training":
            message = prompts.create_instruction_training_message_with_completion(labels, example["text"], relevant_labels)
            train_format_dataset.append(message)
        elif output_type == "conversational_training":
            message = prompts.create_conversational_training_message_with_completion(labels, example["text"], relevant_labels)
            train_format_dataset.append({"messages": message})
    return Dataset.from_list(train_format_dataset)


def formatting(examples, tokenizer, output_type="instruction_training"):
    """Renders a batch of examples with the tokenizer's chat template (prompt + answer)."""
    if output_type == "instruction_training":
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
                tokenize=False, add_generation_prompt=False,
            )
            for prompt, completion in zip(examples["prompt"], examples["completion"])
        ]
    return [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) for messages in examples["messages"]]


def check_response_template(text: str, tokenizer, response_template: str):
    """Fails early if the collator would not find the answer start (it would silently mask everything)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    n = len(template_ids)
    if not any(ids[i:i + n] == template_ids for i in range(len(ids) - n + 1)):
        raise ValueError(f"Response template {response_template!r} not found in a formatted training example; pass --response_template.")


def main(args):
    with open(args.dataset, "r", encoding="utf-8") as file:
        data = json.load(file)

    prompts = Prompts(semantic=args.semantic)
    labels = LABEL_SETS[args.dataset_name]["all"]
    build_dataset = training_data_combined_labels if args.training_type == "combined" else training_data_individual_labels
    dataset = build_dataset(data, labels, prompts, args.output_type)
    print(f"Setting: {'semantic' if args.semantic else 'baseline'} | {len(dataset)} training examples | labels: {labels}")

    if args.seed is not None:
        set_seed(args.seed)  # before loading, so the LoRA initialisation is seeded too
    model, tokenizer = load_model(args.model_name), load_tokenizer(args.model_name)
    print("Fine-tuning:", args.model_name)

    formatting_func = partial(formatting, tokenizer=tokenizer, output_type=args.output_type)
    response_template = args.response_template or response_template_for(tokenizer)
    check_response_template(formatting_func(dataset[:1])[0], tokenizer, response_template)
    print("Response template:", repr(response_template))
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    sft_config = SFTConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        gradient_checkpointing=True,
        per_device_train_batch_size=args.batch_size,
        output_dir=args.output_dir,
        report_to="none",
        save_strategy="no",
        max_seq_length=args.max_seq_length,
        packing=False,
        **({"seed": args.seed} if args.seed is not None else {}),
    )
    peft_config = LoraConfig(
        target_modules=LORA_TARGET_MODULES,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # TRL renamed `tokenizer` to `processing_class`; support both.
    tokenizer_kwarg = "processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else "tokenizer"
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        data_collator=collator,
        args=sft_config,
        peft_config=peft_config,
        **{tokenizer_kwarg: tokenizer},
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Adapter saved to", args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face base model.")
    parser.add_argument("--dataset", type=str, required=True, help="Training dataset JSON.")
    parser.add_argument("--dataset_name", type=str, default="maccrobat", choices=DATASET_NAMES, help="Selects the label set.")
    parser.add_argument("--semantic", action="store_true", help="Use semantic label definitions instead of the plain descriptions.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the LoRA adapter.")
    parser.add_argument("--training_type", type=str, default="combined", choices=["combined", "individual"],
                        help="combined = one example per text with all labels (paper); individual = one per label.")
    parser.add_argument("--output_type", type=str, default="instruction_training", choices=["instruction_training", "conversational_training"])
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--response_template", type=str, default=None, help="Override the automatically detected assistant-turn marker.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps (smoke tests); -1 = train all epochs.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for LoRA initialisation and data order; unset = the Trainer default (42), LoRA init unseeded.")
    main(parser.parse_args())
