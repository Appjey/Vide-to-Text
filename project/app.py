import asyncio
import base64

import torch
import torchaudio
from blacksheep import Application, Response, route, get, post, Content, json, ws, WebSocket
from pathlib import Path

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from main import load_and_resample_audio, split_audio_into_chunks, transcribe_chunk_with_whisper
from model.processor import processor, model, transcribe_audio, device
from model.training import train_model, load_dataset_from_hf_link

import logging

logging.basicConfig(level=logging.INFO)

app = Application()
connected_training_websockets = set()
connected_transcribe_websockets = set()
# Состояние микрофона и текущее транскрибируемое сообщение
microphone_enabled = False
current_transcribed_text = ""
current_dataset_dir = Path("data/current_dataset")
trained_models_dir = Path("data/saved_models")

# Маршруты для статических файлов
app.mount("/static/", app.serve_files("./static"))


@ws("/ws/progress")
async def progress_socket(client: WebSocket):
    logging.info("WebSocket клиент подключился на /ws/progress.")
    connected_training_websockets.add(client)
    try:
        await client.accept()
        while True:
            await asyncio.sleep(1)  # Поддерживаем соединение открытым
    except Exception as e:
        logging.error(f"Ошибка WebSocket соединения (/ws/progress): {e}")
    finally:
        connected_training_websockets.remove(client)
        logging.info("WebSocket клиент отключился от /ws/progress.")


# Обработчик WebSocket для транскрибации
@ws("/ws/transcribe")
async def transcribe_socket(client: WebSocket):
    logging.info("WebSocket клиент подключился на /ws/transcribe.")
    connected_transcribe_websockets.add(client)
    try:
        await client.accept()
        while True:
            # Ожидаем получения аудио данных от клиента
            message = await client.receive()

            # Проверяем, есть ли ключ 'text' в сообщении
            if 'text' in message:
                data = message['text']
                logging.info(f"Получено данные по WebSocket (/ws/transcribe): {len(data)} символов")
                logging.info(f"Данные (первые 50 символов): {data[:50]}")

                try:
                    # Декодируем base64 строку в байты
                    audio_bytes = base64.b64decode(data)
                    logging.info(f"Декодировано {len(audio_bytes)} байт аудио данных.")

                    # Сохраняем аудио данные для проверки
                    audio_filename = "received_audio.webm"
                    with open(audio_filename, "wb") as f:
                        f.write(audio_bytes)
                    logging.info(f"Аудио данные сохранены как '{audio_filename}'.")

                    # Обработка аудио данных
                    transcription = transcribe_with_whisper_in_chunks(audio_filename, chunk_length_sec=30)
                    logging.info(f"Отправка транскрибированного текста клиенту: {transcription}")

                    # Отправка транскрибированного текста обратно клиенту
                    await client.send_text(transcription)
                except Exception as e:
                    logging.error(f"Ошибка при обработке аудио данных: {e}")
                    await client.send_text(f"Error during transcription: {str(e)}")
            else:
                logging.warning("Получены данные не в строковом формате.")
    except Exception as e:
        logging.error(f"Ошибка WebSocket соединения (/ws/transcribe): {e}")
    finally:
        connected_transcribe_websockets.remove(client)
        logging.info("WebSocket клиент отключился от /ws/transcribe.")


# Функция для отправки сообщений всем подключенным WebSocket клиентам (обучение)
async def broadcast_training(message: str):
    logging.info(f"Broadcasting training message: {message}")
    disconnected = set()
    for ws in connected_training_websockets.copy():
        if ws is None:
            logging.warning("Найден None в connected_training_websockets, удаление из набора.")
            disconnected.add(ws)
            continue
        try:
            await ws.send_text(message)
            logging.info(f"Сообщение отправлено клиенту: {message}")
        except Exception as e:
            logging.error(f"Ошибка при отправке сообщения клиенту (/ws/progress): {e}")
            disconnected.add(ws)
    connected_training_websockets.difference_update(disconnected)


@get("/")
def index():
    # Отдаём index.html
    index_file = Path("static/index.html")
    if not index_file.exists():
        return Response(content=Content(b"text/plain", b"Index file not found"), status=404)

    # Используем Content для возврата HTML
    content = index_file.read_bytes()
    return Response(content=Content(b"text/html; charset=utf-8", content), status=200)


@post("/api/mic/on")
def mic_on():
    global microphone_enabled
    microphone_enabled = True
    return json({"status": "microphone enabled"})


# Обработчик для выключения микрофона
@post("/api/mic/off")
def mic_off():
    global microphone_enabled
    microphone_enabled = False
    return json({"status": "microphone disabled"})


@post("/api/dataset/load")
async def load_dataset(request):
    try:
        data = await request.json()
        link = data.get("link", "")
        if not link.startswith("bond005/sberdevices_golos_10h_crowd"):
            return json({"error": "Unsupported dataset link format"}, status=400)
        # Загружаем датасет в отдельном потоке
        dataset = await asyncio.to_thread(load_dataset_from_hf_link, link, current_dataset_dir)
        return json({"status": "Dataset loaded", "dataset_size": len(dataset)}) #TODO: fix
    except Exception as e:
        logging.error(f"Ошибка при загрузке датасета: {e}")
        return json({"error": str(e)}, status=500)


# Обработчик для переобучения модели
@post("/api/model/retrain")
async def retrain_model(request):
    try:
        data = await request.json()
        epochs = int(data.get("epochs", 1))
        lr = float(data.get("lr", 1e-4))
        output_dir = trained_models_dir / "latest_model"

        logging.info(f"Получен запрос на переобучение: epochs={epochs}, lr={lr}")

        # Запускаем переобучение в фоновом режиме
        asyncio.create_task(
            background_train_model(epochs, lr, output_dir)
        )

        return json({"status": "Training started"})
    except Exception as e:
        logging.error(f"Ошибка в обработчике retrain_model: {e}")
        return json({"error": str(e)}, status=500)


# Обработчик для получения транскрибированного текста
@get("/api/transcribed")
def get_transcribed_text():
    return json({"transcribed_text": current_transcribed_text})


# Фоновая функция для переобучения модели и отправки прогресса
async def background_train_model(num_epochs, lr, output_dir):
    global model  # Объявляем, что будем изменять глобальную переменную 'model'
    global processor  # Если нужно изменить processor тоже

    logging.info(f"Начало переобучения: epochs={num_epochs}, lr={lr}, output_dir={output_dir}")

    try:
        # Создаём генератор прогресса из функции train_model
        async for progress in train_model(num_epochs, lr, output_dir):
            logging.info(f"Progress: {progress}")
            await broadcast_training(progress)

        # После завершения переобучения обновляем модель
        logging.info("Загрузка новой обученной модели.")
        new_model = WhisperForConditionalGeneration.from_pretrained(str(output_dir)).to(device)
        model = new_model  # Обновляем глобальную переменную
        logging.info("Переобучение завершено и модель обновлена.")
        await broadcast_training("Training completed and model updated.")
    except Exception as e:
        logging.error(f"Ошибка во время переобучения: {e}")
        await broadcast_training(f"Error during training: {str(e)}")

# Функция для транскрипции аудио по вашему примеру
def transcribe_with_whisper_in_chunks(audio_path, chunk_length_sec=30):
    # Проверяем доступность CUDA и выбираем устройство (GPU или CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Используемое устройство: {device}")

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
        logging.info(f"Транскрибируется фрагмент {i + 1}/{len(chunks)} ...")
        chunk_transcription = transcribe_chunk_with_whisper(chunk, model, processor, sample_rate, device)
        transcription += chunk_transcription + " "

    return transcription.strip()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
