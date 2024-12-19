from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from datasets import load_dataset

# Загружаем модель и токенайзер
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2")

# Загружаем обучающие данные
dataset = load_dataset("path_to_your_dataset")  # Укажите путь к своим данным

# Подготовка данных (например, токенизация)
def preprocess_function(batch):
    inputs = tokenizer(batch["audio_text"], return_tensors="pt", padding=True)
    return inputs

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
