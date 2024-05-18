from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает все источники
    allow_credentials=True,
    allow_methods=["*"],  # Разрешает все методы
    allow_headers=["*"],  # Разрешает все заголовки
)

# Загружаем модели один раз при запуске приложения
whisper_model = whisper.load_model("small")
model = YOLO('./model/yolov8x-cls.pt')

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13
ROLE_TOKENS = {"user": USER_TOKEN, "bot": BOT_TOKEN, "system": SYSTEM_TOKEN}

def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens

def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)

class AnalyzeRequest(BaseModel):
    text: str

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
        raise HTTPException(
            status_code=500, detail=f"Model loading failed: {str(e)}")

    system_tokens = get_system_tokens(model)
    tokens = system_tokens

    user_message = f"Проанализируй текст видео и определи к какому классу оно относится.Классы видео: 1. Насилие 2. Непристойный контент (материалы сексуального характера) 3. Наркотики и алкоголь 4. Вульгарность и ненормативная лексика 5. Опасное поведение (вредные действия и ненадежный контент) 6. Дискриминация и ненависть 7. Шокирующий контент 8. Унижающий или провокационный контент 9. Пособничество недобросовестной деятельности 10. Запрещенные материалы, связанные с азартными играми. 11. Обычное видео Текст видео - {request.text}"
    message_tokens = get_message_tokens(
        model=model, role="user", content=user_message)
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
    print(bot_response)
    return {"dangerous_words": bot_response}

async def download_video(url: str) -> str:
    if 'youtube.com' in url or 'youtu.be' in url:
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension='mp4').first()
        temp_video = NamedTemporaryFile(delete=False, suffix='.mp4')
        stream.download(filename=temp_video.name)
        return temp_video.name
    else:
        response = requests.get(url, stream=True)
        temp_video = NamedTemporaryFile(delete=False, suffix='.mp4')
        with open(temp_video.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return temp_video.name

@app.post("/transcribe")
async def transcribe(video: UploadFile = File(None), url: str = None):
    if video:
        temp_video_path = f"temp_{video.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await video.read())
    elif url:
        temp_video_path = await download_video(url)
    else:
        raise HTTPException(status_code=400, detail="No video file or URL provided.")

    # Конвертируем видео в аудио с помощью moviepy
    video_clip = VideoFileClip(temp_video_path)
    temp_audio_path = f"temp_{os.path.basename(temp_video_path).split('.')[0]}.mp3"
    video_clip.audio.write_audiofile(temp_audio_path)
    video_clip.close()  # Закрываем файловый объект

    # Загружаем аудио и транскрибируем
    transcription = whisper_model.transcribe(temp_audio_path)

    # Удаляем временные файлы
    for temp_file in [temp_video_path, temp_audio_path]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return {"transcription": transcription["text"]}

@app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(None), url: str = None):
    if video:
        temp_video_path = f"temp_{video.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await video.read())
    elif url:
        temp_video_path = await download_video(url)
    else:
        raise HTTPException(status_code=400, detail="No video file or URL provided.")

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
                # Выбор максимальной вероятности для каждого класса
                class_words[label] = max(class_words[label], top1_confidence)

    cap.release()

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    return {"class_words": class_words}
