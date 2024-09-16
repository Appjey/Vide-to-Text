import torch
import torchaudio
import moviepy.editor as mp
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Пути для видео, аудио и файла с результатами транскрипции
video_path = "./mnt/data/2024-09-11 Иванов SA.mkv"
audio_path = "./mnt/data/interview_audio.wav"
transcription_file_path = "./mnt/data/interview_transcription.txt"

# Максимальная длина фрагмента в секундах (30 секунд)
MAX_AUDIO_LENGTH_SEC = 30


# Функция для извлечения аудио из видео
def extract_audio_from_video(video_path, audio_path):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    print(f"Аудио сохранено в {audio_path}")


# Функция для загрузки и ресемплирования аудио до 16000 Гц и приведения к моноформату
def load_and_resample_audio(audio_path, target_sample_rate=16000):
    # Загружаем аудиофайл
    speech_array, original_sample_rate = torchaudio.load(audio_path)

    # Если аудио стерео, преобразуем в моно
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    # Если частота дискретизации отличается, ресемплируем аудио до 16000 Гц
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        speech_array = resampler(speech_array)

    return speech_array, target_sample_rate


# Функция для разбиения аудио на фрагменты
def split_audio_into_chunks(speech_array, sample_rate, chunk_length_sec=30):
    chunk_length = chunk_length_sec * sample_rate
    num_chunks = (speech_array.size(1) + chunk_length - 1) // chunk_length
    chunks = [speech_array[:, i * chunk_length:(i + 1) * chunk_length] for i in range(num_chunks)]
    return chunks


# Функция для транскрибации фрагмента с использованием Whisper и CUDA
def transcribe_chunk_with_whisper(chunk, model, processor, sample_rate, device):
    # Преобразуем аудио в формат для модели Whisper
    chunk = chunk.squeeze().numpy()  # Преобразуем тензор в numpy
    inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


# Основная функция для обработки аудио и транскрибации по фрагментам
def transcribe_with_whisper_in_chunks(audio_path, chunk_length_sec=30):
    # Проверяем доступность CUDA и выбираем устройство (GPU или CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    # Загружаем модель и процессор
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
    model.config.forced_decoder_ids = None  # Отключаем forced decoder ids

    # Загружаем и ресемплируем аудиофайл до 16000 Гц
    speech_array, sample_rate = load_and_resample_audio(audio_path)

    # Разбиваем аудио на фрагменты
    chunks = split_audio_into_chunks(speech_array, sample_rate, chunk_length_sec)

    # Транскрибируем каждый фрагмент по очереди
    transcription = ""
    for i, chunk in enumerate(chunks):
        print(f"Транскрибируется фрагмент {i + 1}/{len(chunks)} ...")
        chunk_transcription = transcribe_chunk_with_whisper(chunk, model, processor, sample_rate, device)
        transcription += chunk_transcription + " "

    return transcription.strip()


# Основная функция для извлечения аудио из видео и транскрибации
def process_video_to_transcription(video_path, audio_path, transcription_file_path):
    # Шаг 1: Извлечение аудио из видео
    extract_audio_from_video(video_path, audio_path)

    # Шаг 2: Транскрибация аудио по фрагментам
    transcription = transcribe_with_whisper_in_chunks(audio_path)

    # Шаг 3: Сохранение транскрипции в файл
    with open(transcription_file_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    print(f"Транскрипция завершена и сохранена в {transcription_file_path}")


# Запуск основного процесса
process_video_to_transcription(video_path, audio_path, transcription_file_path)
