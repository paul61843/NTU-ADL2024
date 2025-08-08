from transformers import MT5Tokenizer, MT5ForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd
import torch
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 定義 Dataset 類別
class MT5Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_len=512, max_target_len=64):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = f"summarize: {row['maintext']}"
        target_text = row['title']

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # T5 的 target 需要 labels，但 padding token 要換成 -100
        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


num_epochs = 1
train_data = pd.read_json("./data/train.jsonl", lines=True)

accelerator = Accelerator()
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# 建立 Dataset 和 DataLoader
train_dataset = MT5Dataset(train_data, tokenizer)
training_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

total_steps = num_epochs * len(training_dataloader)
progress_bar = tqdm(range(total_steps))

# 定義 optimizer 和 scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(training_dataloader) * num_epochs  # 假設訓練 3 個 epoch
scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 將 model, optimizer, dataloader, scheduler 丟給 Accelerator
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

# 開始訓練（簡單範例）
model.train()
for epoch in range(num_epochs):
    for batch in training_dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_description(f"Epoch {epoch+1} Loss {loss.item():.4f}")
        progress_bar.update(1)

    print(f"Epoch {epoch+1} completed.")

model = accelerator.unwrap_model(model)
model.save_pretrained("./output_model")
tokenizer.save_pretrained("./output_model")  # ✅ 與你的模型同一資料夾
