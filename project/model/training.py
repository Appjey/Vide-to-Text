# model/training.py

import asyncio
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from project.model.processor import model, processor


async def load_dataset_from_hf_link(link: str, data_dir: Path):
    from datasets import load_dataset
    dataset = load_dataset(link, split="train")
    dataset.save_to_disk(str(data_dir))
    return dataset


async def train_model(num_epochs: int, lr: float, output_dir: Path):
    """
    Асинхронная функция для обучения модели.
    Генерирует строки прогресса, которые можно отправлять клиентам.
    """
    try:
        for epoch in range(1, num_epochs + 1):
            # Реальная логика обучения модели
            # Здесь заменяем на задержку для примера
            await asyncio.sleep(2)  # Замените на реальное обучение

            # Генерация прогресса
            progress = f"Epoch {epoch}/{num_epochs} completed."
            yield progress

        # Сохранение модели после обучения
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_dir))
            processor.save_pretrained(str(output_dir))

        yield "Training completed and model saved."
    except Exception as e:
        yield f"Error during training: {str(e)}"
