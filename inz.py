from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from datasets import load_dataset

from project.model.process import processor, model

# Загружаем модель и токенайзер
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2")

# Загружаем обучающие данные
dataset = load_dataset("path_to_your_dataset")  # Укажите путь к своим данным

# Подготовка данных (например, токенизация)
def preprocess_function(batch):
    inputs = processor.feature_extractor(
        batch["audio"]["array"],
        sampling_rate=16000,
        return_tensors="pt"
    )
    labels = processor.tokenizer(
        batch["text"],
        return_tensors="pt",
        truncation=True
    )

    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels.input_ids[0]
    return batch


tokenized_dataset = dataset.map(preprocess_function)

# Тренировка модели
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./whisper_finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()
