# model/processor.py

from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
import io
import wave
from pydub import AudioSegment
import numpy as np
import torch
from pathlib import Path
import logging

# Определяем устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка процессора (токенизатора) и модели из Hugging Face
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)

model.eval()


def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000, language: str = None) -> str:
    """
    Транскрибирует аудио данные в текст.

    :param audio_bytes: Аудио данные в формате WAV, OGG или WebM
    :param sample_rate: Частота дискретизации (по умолчанию 16000 Гц)
    :param language: Язык аудио (например, 'ru' для русского). Если не указан, модель будет пытаться определить язык автоматически.
    :return: Транскрибированный текст или сообщение об ошибке
    """
    try:
        logging.info("Начало транскрибирования аудио.")

        # Определение формата аудио на основе магических чисел
        if audio_bytes.startswith(b'OggS'):
            audio_format = 'ogg'
            logging.info("Определен формат аудио: OGG")
        elif audio_bytes.startswith(b'\x1A\x45\xDF\xA3'):
            audio_format = 'webm'
            logging.info("Определен формат аудио: WebM")
        elif audio_bytes.startswith(b'\x52\x49\x46\x46') and audio_bytes[8:12] == b'WAVE':
            audio_format = 'wav'
            logging.info("Определен формат аудио: WAV")
        else:
            audio_format = 'unknown'
            logging.error("Неизвестный формат аудио.")
            return "Unsupported audio format."

        # Конвертация аудио в WAV, если необходимо
        if audio_format in ['ogg', 'webm']:
            target_format = 'wav'
            logging.info(f"Конвертация аудио из {audio_format.upper()} в WAV.")
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                audio = audio.set_frame_rate(sample_rate).set_channels(
                    1)  # Установка частоты дискретизации и моно-канальности
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                audio_bytes = wav_io.getvalue()
                logging.info(f"Конвертация из {audio_format.upper()} в WAV завершена.")
            except Exception as e:
                logging.error(f"Ошибка при конвертации {audio_format.upper()} в WAV: {e}")
                return f"Error during conversion: {str(e)}"

        elif audio_format == 'wav':
            logging.info("Формат аудио уже WAV, конвертация не требуется.")

        # Чтение WAV данных
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)
            logging.info(f"Чтение WAV данных завершено: {n_channels} канал(ов), {sampwidth * 8}-бит, {framerate} Гц.")
        except wave.Error as e:
            logging.error(f"Ошибка при чтении WAV данных: {e}")
            return f"Error reading WAV data: {str(e)}"

        # Преобразование в numpy массив
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(
                np.float32) / 32768.0  # Нормализация в диапазон [-1.0, 1.0]
            logging.info(f"Преобразование аудио данных в numpy массив завершено. Длина массива: {len(audio_np)}.")
        except Exception as e:
            logging.error(f"Ошибка при преобразовании аудио данных в numpy массив: {e}")
            return f"Error processing audio data: {str(e)}"

        # Если стерео, берем первый канал
        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels)
            audio_np = audio_np[:, 0]
            logging.info("Обработка стерео аудио: выбран первый канал.")

        # Преобразуем в список
        audio_waveform = audio_np.tolist()

        # Подготовка входных данных для Whisper
        try:
            if language:
                inputs = processor(audio_waveform, sampling_rate=framerate, language=language,
                                   return_tensors="pt").input_features.to(device)
                logging.info(f"Подготовка входных данных для модели завершена с указанным языком: {language}.")
            else:
                inputs = processor(audio_waveform, sampling_rate=framerate, return_tensors="pt").input_features.to(
                    device)
                logging.info(
                    "Подготовка входных данных для модели завершена без указания языка (автоматическое определение).")
        except Exception as e:
            logging.error(f"Ошибка при подготовке входных данных: {e}")
            return f"Error preparing inputs: {str(e)}"

        # Генерация предсказанных идентификаторов
        try:
            with torch.no_grad():
                predicted_ids = model.generate(inputs, max_length=448)
            logging.info("Генерация предсказанных идентификаторов завершена.")
        except Exception as e:
            logging.error(f"Ошибка при генерации предсказанных идентификаторов: {e}")
            return f"Error during generation: {str(e)}"

        # Декодирование предсказанных идентификаторов в текст
        try:
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logging.info(f"Транскрибированный текст: {transcription}")
            return transcription
        except Exception as e:
            logging.error(f"Ошибка при декодировании предсказанных идентификаторов: {e}")
            return f"Error during decoding: {str(e)}"
    except Exception as e:
        logging.error(f"Ошибка при транскрибировании аудио: {e}")
        return f"Error during transcription: {str(e)}"
