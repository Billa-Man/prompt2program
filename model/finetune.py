import argparse
from pathlib import Path

from unsloth import PatchDPOTrainer

from typing import Any, List, Literal, Optional

import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import TextStreamer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

from settings import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PatchDPOTrainer()

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


def load_model(model_name: str,
               max_seq_length: int,
               load_in_4bit: bool,
               lora_rank: int,
               lora_alpha: int,
               lora_dropout: float,
               target_modules: List[str],
               chat_template: str,
              ) -> tuple:
  
  model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name,
                                                       max_seq_length=max_seq_length,
                                                       load_in_4bit=load_in_4bit,
                                                       )

  model = FastLanguageModel.get_peft_model(model,
                                           r=lora_rank,
                                           lora_alpha=lora_alpha,
                                           lora_dropout=lora_dropout,
                                           target_modules=target_modules,
                                           )

  tokenizer = get_chat_template(tokenizer,
                                chat_template=chat_template,
                              )

  return model, tokenizer


def finetune(model_name: str,
             output_dir: str,
             dataset_huggingface_workspace: str,
             max_seq_length: int = 2048,
             load_in_4bit: bool = False,
             lora_rank: int = 32,
             lora_alpha: int = 32,
             lora_dropout: float = 0.0,
             target_modules: List[str] = ["q_proj", "k_proj", "v_proj", 
                                          "up_proj", "down_proj", "o_proj", "gate_proj"],
             chat_template: str = "chatml",
             learning_rate: float = 3e-4,
             num_train_epochs: int = 3,
             per_device_train_batch_size: int = 2,
             gradient_accumulation_steps: int = 8,
             is_dummy: bool = True,
             ) -> tuple:
  
  model, tokenizer = load_model(model_name, max_seq_length, load_in_4bit, 
                                lora_rank, lora_alpha, lora_dropout, 
                                target_modules, chat_template)
  
  EOS_TOKEN = tokenizer.eos_token
  print(f"[FINETUNE] Setting EOS_TOKEN to {EOS_TOKEN}")

  if is_dummy is True:
    num_train_epochs = 1
    print(f"[FINETUNE] Training in dummy mode. Setting num_train_epochs to '{num_train_epochs}'")
    print(f"[FINETUNE] Training in dummy mode. Reducing dataset size to 500.")

  def format_samples_sft(examples):
    text = []
    for instruction, output in zip(examples["instruction"], examples["output"], strict=False):
      message = alpaca_template.format(instruction, output) + EOS_TOKEN
      text.append(message)

    return {"text": text}

  dataset1 = load_dataset(settings.DATASET_ID, split="train")
  dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
  dataset3 = load_dataset("iamtarun/code_instructions_120k_alpaca", split="train[:10000]")

  dataset = concatenate_datasets([dataset1, dataset2, dataset3])

  if is_dummy:
      dataset = dataset.select(range(500))
  print(f"[FINETUNE] Loaded dataset with {len(dataset)} samples.")

  dataset = dataset.map(format_samples_sft, batched=True, remove_columns=dataset.column_names)
  dataset = dataset.train_test_split(test_size=0.05)

  print("[FINETUNE] Training dataset example:")
  print(dataset["train"][0])

  trainer = SFTTrainer(model=model,
                       tokenizer=tokenizer,
                       train_dataset=dataset["train"],
                       eval_dataset=dataset["test"],
                       dataset_text_field="text",
                       max_seq_length=max_seq_length,
                       dataset_num_proc=2,
                       packing=True,
                       args=TrainingArguments(learning_rate=learning_rate,
                                              num_train_epochs=num_train_epochs,
                                              per_device_train_batch_size=per_device_train_batch_size,
                                              gradient_accumulation_steps=gradient_accumulation_steps,
                                              fp16=not is_bfloat16_supported(),
                                              bf16=is_bfloat16_supported(),
                                              logging_steps=1,
                                              optim="adamw_8bit",
                                              weight_decay=0.01,
                                              lr_scheduler_type="linear",
                                              per_device_eval_batch_size=per_device_train_batch_size,
                                              warmup_steps=10,
                                              output_dir=output_dir,
                                              # report_to="comet_ml",
                                              seed=0,
                                              ),
                        )

  trainer.train()

  return model, tokenizer


def inference(model: Any,
              tokenizer: Any,
              prompt: str = "Write a Python program that reads a number from the user and prints the number squared.",
              max_new_tokens: int = 512,
              ) -> None:
  
  model = FastLanguageModel.for_inference(model)
  message = alpaca_template.format(prompt, "")
  inputs = tokenizer([message], return_tensors="pt").to(device)

  text_streamer = TextStreamer(tokenizer)
  _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True)


def save_model(model: Any, 
               tokenizer: Any, 
               output_dir: str, 
               push_to_hub: bool = False, 
               repo_id: Optional[str] = None):
  
  model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

  if push_to_hub and repo_id:
    print(f"Saving model to '{repo_id}'")
    model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")


# MAIN FUNCTION
if __name__ == "__main__":

  # Parse Arguments
  parser = argparse.ArgumentParser()

  parser.add_argument("--num_train_epochs", type=int, default=2)
  parser.add_argument("--per_device_train_batch_size", type=int, default=2)
  parser.add_argument("--learning_rate", type=float, default=3e-4)
  parser.add_argument("--dataset_huggingface_workspace", type=str, default=settings.HUGGINGFACE_USERNAME)
  parser.add_argument("--model_output_huggingface_workspace", type=str, default=settings.HUGGINGFACE_USERNAME)
  parser.add_argument("--is_dummy", type=bool, default=False, help="Flag to reduce the dataset size for testing")

  parser.add_argument("--output_data_dir", type=str, default="/content/output")
  parser.add_argument("--model_dir", type=str, default="/content/model")
  parser.add_argument("--n_gpus", type=str, default=1)

  args = parser.parse_args()

  # Check parameters
  print(f"Num training epochs: '{args.num_train_epochs}'")
  print(f"Per device train batch size: '{args.per_device_train_batch_size}'")
  print(f"Learning rate: {args.learning_rate}")
  print(f"Datasets will be loaded from Hugging Face workspace: '{args.dataset_huggingface_workspace}'")
  print(f"Models will be saved to Hugging Face workspace: '{args.model_output_huggingface_workspace}'")
  print(f"Training in dummy mode? '{args.is_dummy}'")

  print(f"Output data dir: '{args.output_data_dir}'")
  print(f"Model dir: '{args.model_dir}'")
  print(f"Number of GPUs: '{args.n_gpus}'")

  print("Starting SFT training...")
  base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
  print(f"Training from base model: '{base_model_name}'")

  output_dir_sft = Path(args.model_dir) / "output_sft"
  model, tokenizer = finetune(model_name=base_model_name,
                              output_dir=str(output_dir_sft),
                              dataset_huggingface_workspace=args.dataset_huggingface_workspace,
                              num_train_epochs=args.num_train_epochs,
                              per_device_train_batch_size=args.per_device_train_batch_size,
                              learning_rate=args.learning_rate,
                              is_dummy=False,
                              )
  
  inference(model, tokenizer)

  sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/prompt2program-sft"
  save_model(model, tokenizer, "model_sft", push_to_hub=True, repo_id=sft_output_model_repo_id)