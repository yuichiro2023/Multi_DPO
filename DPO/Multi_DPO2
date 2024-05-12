# プロンプト+応答からのプロンプトを抽出。
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_anthropic_prompt(sample):
    text0 = sample["chosen"]
    text0 = text0.replace("\\n\\nhuman:", "User: ")
    text0 = text0.replace("\\n\\nAssistant:", "\n\nAssistant: ")
    text1 = sample["rejected"]
    text1 = text1.replace("\\n\\nhuman:", "User: ")
    text1 = text1.replace("\\n\\nAssistant:", "\n\nAssistant: ")
    search_term = "\n\nAssistant: "
    search_term_idx0 = text0.rfind(search_term)
    search_term_idx1 = text1.rfind(search_term)

    return {
        "prompt": text0[: search_term_idx0 + len(search_term)],
        "chosen": text0[search_term_idx0 + len(search_term):],
        "rejected": text1[search_term_idx1 + len(search_term):],
    }
    return sample

from datasets import Dataset, load_dataset
from typing import Dict

def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    # デフォルトのBuilderConfigを使用してデータセットをロード
    dataset = load_dataset("Anthropic/hh-rlhf", "default", cache_dir=cache_dir)

    # train_test_splitを使用してデータセットを分割
    if "train" in dataset and "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=False)

    # 指定されたsplitを選択
    dataset = dataset[split]

    # sanity_checkがTrueの場合、データセットのサイズを1000に制限
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    # プロンプトとレスポンスを分けるための関数
    def split_prompt_and_responses(sample) -> Dict[str, str]:
        return extract_anthropic_prompt(sample)

    # map関数を使用して、全てのデータサンプルにsplit_prompt_and_responsesを適用
    return dataset.map(split_prompt_and_responses)

# データセットの準備
train_dataset = get_hh("train", sanity_check=True)
eval_dataset = get_hh("test", sanity_check=True)

# データセットの確認
print(train_dataset)
print(eval_dataset)
print("--prompt--\n", train_dataset[2]["prompt"])
print("--chosen--\n", train_dataset[2]["chosen"])
print("--rejected--\n", train_dataset[2]["rejected"])
print("--prompt--\n", eval_dataset[2]["prompt"])
print("--chosen--\n", eval_dataset[2]["chosen"])
print("--rejected--\n", eval_dataset[2]["rejected"])

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

from accelerate import PartialState
device_string = PartialState().process_index

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    "JINIAC/JINIAC-5B-base-en",
    trust_remote_code=True,
    device_map={'':device_string},
    load_in_8bit=True,
)
model.config.use_cache = False  # キャッシュ (学習時はFalse)
model.config.pretraining_tp = 1  # 事前学習で使用したテンソル並列ランク

# 参照モデルの準備
model_ref = AutoModelForCausalLM.from_pretrained(
    "JINIAC/JINIAC-5B-base-en",
    trust_remote_code=True,
    device_map={'':device_string},
    load_in_8bit=True,
)
model_ref.config.pretraining_tp = 1  # 事前学習で使用したテンソル並列ランク

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(
    "JINIAC/JINIAC-5B-base-en",
    use_fast=False,
)

from transformers import TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig

# LoRAパラメータ
peft_config = LoraConfig(
    r=64,  # LoRAアテンションの次元
    lora_alpha=16,  # LoRAスケーリングのAlphaパラメータ
    lora_dropout=0.1,  # LoRA レイヤーのドロップアウト確率
    bias="none",  # LoRAのバイアス種別 ("none","all", "lora_only")
    task_type="CAUSAL_LM",  # タスク種別
    target_modules=["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "k_proj", "v_proj"]
)

# 学習パラメータ
training_args = TrainingArguments(
    output_dir="./train_logs",  # 出力フォルダ
    fp16=False,  # fp16学習の有効化
    bf16=True,  # bf16学習の有効化
    max_steps=300,  # 学習ステップ数
    per_device_train_batch_size=4,  # 学習用のGPUあたりのバッチサイズ
    gradient_accumulation_steps=1,  # 勾配を蓄積するための更新ステップの数
    optim="paged_adamw_32bit",  # オプティマイザ
    learning_rate=2e-4,  # 初期学習率
    lr_scheduler_type="cosine",  # 学習率スケジュール
    max_grad_norm=0.3,  # 最大法線勾配 (勾配クリッピング)
    warmup_ratio=0.03,  # 線形ウォームアップのステップ比率 (0から学習率まで)
    weight_decay=0.001,  # bias/LayerNormウェイトを除く全レイヤーに適用するウェイト減衰
    save_steps=50,  # 何ステップ毎にチェックポイントを保存するか
    logging_steps=50,  # 何ステップ毎にログを記録するか
    report_to="tensorboard"  # レポート
)

#DPOトレーナーの準備
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    force_use_ref_model=True,
)


# 学習の実行
dpo_trainer.train()
dpo_trainer.model.save_pretrained("./dpo_model")


# プロンプトの準備
prompt = """User: Do black people scare you?

Assistant: """

# 推論の実行
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
output_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)
output = tokenizer.decode(output_ids.tolist()[0])
print(output)

# プロンプトの準備
prompt = """User: What kind of noises did dinosaurs make?

Assistant: """

# 推論の実行
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
output_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)
output = tokenizer.decode(output_ids.tolist()[0])
print(output)

# プロンプトの準備
prompt = """User: What are some cuss words in english?

Assistant: """

# 推論の実行
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
output_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)
output = tokenizer.decode(output_ids.tolist()[0])
print(output)
