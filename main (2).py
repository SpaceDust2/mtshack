from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
from llama_cpp import Llama
from moviepy.editor import VideoFileClip
import os
import cv2
from ultralytics import YOLO
from pytube import YouTube
import requests
from tempfile import NamedTemporaryFile
from bs4 import BeautifulSoup
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect
from confluent_kafka import Producer, Consumer, KafkaException, KafkaError
import json
import asyncio
from models.model import VideoURL, AnalyzeRequest
from utils.kafka import producer, send_message_to_kafka, kafka_consumer_task
import redis

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создание объекта клиента Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Теперь вы можете использовать redis_client для взаимодействия с Redis


# Загружаем модели один раз при запуске приложения
whisper_model = whisper.load_model("small")
model = YOLO('./model/yolov8x-cls.pt')

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13
ROLE_TOKENS = {"user": USER_TOKEN, "bot": BOT_TOKEN, "system": SYSTEM_TOKEN}

model_checkpoint_rus = 'cointegrated/rubert-tiny-toxicity'
tokenizer_rus = AutoTokenizer.from_pretrained(model_checkpoint_rus)
model_rus = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_rus)
if torch.cuda.is_available():
    model_rus.cuda()

classifier_eng = pipeline(
    "text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


def text2toxicity(text, aggregate=True):
    with torch.no_grad():
        inputs = tokenizer_rus(text, return_tensors='pt',
                               truncation=True, padding=True).to(model_rus.device)
        proba = torch.sigmoid(model_rus(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba


def get_video_metadata(video_url):
    print("Функция запущена")
    start_time = time.time()

    response = requests.get(video_url)
    if (response.status_code != 200):
        raise HTTPException(status_code=404, detail="Video not found")

    soup = BeautifulSoup(response.content, 'html.parser')

    title_tag = soup.find('meta', property='og:title')
    description_tag = soup.find('meta', property='og:description')
    age_restriction_tag = soup.find('meta', property='age-restriction')

    title = title_tag['content'] if title_tag else 'Название не найдено'
    description = description_tag['content'] if description_tag else 'Описание не найдено'
    age_restriction = age_restriction_tag['content'] if age_restriction_tag else '0'

    language = detect(description) if description != 'Описание не найдено' else 'unknown'
    print(f"Detected language: {language}")

    if language == 'ru':
        toxicity_score = text2toxicity(description, True)
        is_safe_for_kids = toxicity_score < 0.5
        toxicity_percentage = toxicity_score * 100
        print(f"Токсичность текста: {toxicity_percentage:.2f}%")
    else:
        if description != 'Описание не найдено':
            prediction = classifier_eng(description, return_all_scores=True)
            print(f"Prediction: {prediction}")
            is_safe_for_kids = prediction[0][0]['label'] == 'LABEL_1'
        else:
            is_safe_for_kids = int(age_restriction) < 13

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {
        'title': title,
        'description': description,
        'is_safe_for_kids': is_safe_for_kids,
        'elapsed_time': elapsed_time
    }


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    model_path = "./model/model-q4_K.gguf"
    n_ctx = 10000
    top_k = 30
    top_p = 0.9
    temperature = 0.01
    repeat_penalty = 1.1

    try:
        model = Llama(model_path=model_path, n_ctx=n_ctx, n_parts=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    system_tokens = get_system_tokens(model)
    tokens = system_tokens

    user_message = f"Проанализируй текст видео и определи к какому классу оно относится. Классы видео: 1. Насилие 2. Непристойный контент (материалы сексуального характера) 3. Наркотики и алкоголь 4. Вульгарность и ненормативная лексика 5. Опасное поведение (вредные действия и ненадежный контент) 6. Дискриминация и ненависть 7. Шокирующий контент 8. Унижающий или провокационный контент 9. Пособничество недобросовестной деятельности 10. Запрещенные материалы, связанные с азартными играми. 11. Обычное видео Текст видео - {request.text}"
    message_tokens = get_message_tokens(model=model, role="user", content=user_message)
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens

    generator = model.generate(
        tokens, top_k=top_k, top_p=top_p, temp=temperature, repeat_penalty=repeat_penalty)

    bot_response = ""
    for token in generator:
        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        bot_response += token_str
        if token == model.token_eos():
            break
    redis_key = f"analyze:request:text"
    redis_client.set(redis_key, json.dumps({"dangerous_words": bot_response}))

    # Логирование успешной отправки сообщения в Redis
    print(f"Сообщение успешно отправлено в Redis: {redis_key}")

    return {"dangerous_words": bot_response}


async def download_video(url: str) -> str:
    if 'youtube.com' in url or 'youtu.be' in url:
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension='mp4').first()
        temp_video = NamedTemporaryFile(delete=False, suffix='.mp4')
        stream.download(filename=temp_video.name)
        redis_key = f"downloaded_video:url:{url}"
        redis_client.set(redis_key, temp_video.name)
        return temp_video.name
    else:
        response = requests.get(url, stream=True)
        temp_video = NamedTemporaryFile(delete=False, suffix='.mp4')
        with open(temp_video.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        redis_key = f"downloaded_video:url:{url}"
        redis_client.set(redis_key, temp_video.name)
        return temp_video.name


@app.post("/transcribe")
async def transcribe(video: UploadFile = File(None), url: str = Form(None)):
    if video:
        temp_video_path = f"temp_{video.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await video.read())
    elif url:
        temp_video_path = await download_video(url)
    else:
        raise HTTPException(status_code=400, detail="No video file or URL provided.")

    video_clip = VideoFileClip(temp_video_path)
    temp_audio_path = f"temp_{os.path.basename(temp_video_path).split('.')[0]}.mp3"
    video_clip.audio.write_audiofile(temp_audio_path)
    video_clip.close()

    transcription = whisper_model.transcribe(temp_audio_path)

    # Удаление временных файлов
    for temp_file in [temp_video_path, temp_audio_path]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Отправка транскрипции в Kafka
    message = json.dumps({"transcription": transcription["text"]})
    send_message_to_kafka('video-transcription', message)
    print(f'Message sent to Kafka: {message}')  # Логирование отправленного сообщения

    redis_key = f"transcribed_video:url:{url}"
    redis_client.set(redis_key, transcription["text"])

    return {"transcription": transcription["text"]}


@app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(None), url: str = Form(None)):
    if video:
        temp_video_path = f"temp_{video.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await video.read())
    elif url:
        temp_video_path = await download_video(url)
    else:
        raise HTTPException(
            status_code=400, detail="No video file or URL provided.")

    cap = cv2.VideoCapture(temp_video_path)
    class_words = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            top1_index = result.probs.top1
            top1_confidence = result.probs.top1conf.item()
            label = result.names[top1_index]

            if label not in class_words:
                class_words[label] = top1_confidence
            else:
                class_words[label] = max(class_words[label], top1_confidence)

    cap.release()

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    redis_key = f"analyzed_video:url:{url}"
    redis_client.set(redis_key, json.dumps({"class_words": class_words}))

    return {"class_words": class_words}


@app.post("/metadata")
async def analyze_video_metadata(video: VideoURL):
    try:
        metadata = get_video_metadata(video.url)
        message = json.dumps(metadata)
        send_message_to_kafka('video-metadata', message)
        redis_key = f"video_metadata:url:{video.url}"
        redis_client.set(redis_key, json.dumps(metadata))
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    loop.create_task(kafka_consumer_task())
