"""Model loading and chat-template handling shared by training and extraction.

Replaces the per-model code edits the original scripts needed: the DeepSeek prompt
suffix, Gemma's missing system role, base models without a chat template, and the
SFT response template.
"""
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEMPLATE_DIR = Path(__file__).resolve().parent / "chat_templates"

DEEPSEEK_ASSISTANT = "<｜Assistant｜>"

#: (marker found in the chat template, text that starts the assistant turn)
RESPONSE_TEMPLATES = [
    (DEEPSEEK_ASSISTANT, DEEPSEEK_ASSISTANT),                                  # DeepSeek-R1 distills
    ("<|start_header_id|>", "<|start_header_id|>assistant<|end_header_id|>"),  # Llama 3
    ("<start_of_turn>", "<start_of_turn>model\n"),                             # Gemma / TxGemma
    ("<|im_start|>", "<|im_start|>assistant\n"),                               # Qwen / ChatML
]


def load_tokenizer(model_name: str):
    """Loads the slow tokenizer, as the original runs did. Llama 3 has no slow tokenizer and
    some transformers versions then return False instead of falling back, so load the fast one."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=False)
    if not hasattr(tokenizer, "eos_token"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    return ensure_chat_template(tokenizer)


def load_model(model_name: str, adapter_name: str = None):
    """Loads the base model in 4-bit (bitsandbytes), optionally with a LoRA adapter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config).to(device)
    if adapter_name is not None:
        model = PeftModel.from_pretrained(model, adapter_name).to(device)
    return model


def ensure_chat_template(tokenizer):
    """Base models (e.g. Meta-Llama-3.1-8B) ship without a chat template; the paper's
    runs gave them the Llama 3.1 Instruct template."""
    if not tokenizer.chat_template:
        tokenizer.chat_template = (TEMPLATE_DIR / "llama-3.1.jinja").read_text(encoding="utf-8")
        print("Tokenizer has no chat template; using the Llama 3.1 template.")
    return tokenizer


def is_deepseek(tokenizer) -> bool:
    return DEEPSEEK_ASSISTANT in (tokenizer.chat_template or "")


def supports_system_role(tokenizer) -> bool:
    return "System role not supported" not in (tokenizer.chat_template or "")


def merge_system_into_user(messages):
    """Gemma-style templates reject a system turn: prepend it to the first user turn."""
    if not messages or messages[0]["role"] != "system":
        return messages
    system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    rest[0]["content"] = system + "\n\n" + rest[0]["content"]
    return rest


def response_template_for(tokenizer) -> str:
    """The text that starts the assistant turn, used to mask the prompt out of the SFT loss."""
    template = tokenizer.chat_template or ""
    for marker, response_template in RESPONSE_TEMPLATES:
        if marker in template:
            return response_template
    raise ValueError("Cannot infer the response template from the chat template; pass --response_template.")
