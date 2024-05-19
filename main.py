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
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split

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

# Загрузка модели и токенизатора для русского языка
model_checkpoint_rus = 'cointegrated/rubert-tiny-toxicity'
tokenizer_rus = AutoTokenizer.from_pretrained(model_checkpoint_rus)
model_rus = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint_rus)
if torch.cuda.is_available():
    model_rus.cuda()

classifier_rus = pipeline("text-classification",
                          model="cointegrated/rubert-tiny-toxicity")
# Загрузка модели для английского языка
classifier_eng = pipeline(
    "text-classification", model="s-nlp/roberta_toxicity_classifier")
MAX_TOKEN_LENGTH = 480  # Максимальная длина токенов для модели BERT


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
    start_time = time.time()  # Начало отсчета времени

    response = requests.get(video_url)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Video not found")

    soup = BeautifulSoup(response.content, 'html.parser')

    # Извлечение метаданных
    title_tag = soup.find('meta', property='og:title')
    description_tag = soup.find('meta', property='og:description')
    age_restriction_tag = soup.find('meta', property='age-restriction')

    title = title_tag['content'] if title_tag else 'Название не найдено'
    description = description_tag['content'] if description_tag else 'Описание не найдено'

    # Удаление временных меток из описания
    description = re.sub(r'\d+:\d+', '', description)

    # Предполагаем, что нет ограничений
    age_restriction = age_restriction_tag['content'] if age_restriction_tag else '0'

    # Определение языка текста
    language = detect(
        description) if description != 'Описание не найдено' else 'unknown'
    print(f"Detected language: {language}")

    if language == 'ru':
        # Анализ метаданных с использованием русской модели
        if description != 'Описание не найдено':
            prediction = classifier_rus(description, return_all_scores=True)
            # Вывод предсказания для отладки
            print(f"Prediction: {prediction}")
            # Проверяем категорию 'non-toxic' и ее вероятность
            non_toxic_score = next(
                (item['score'] for item in prediction[0] if item['label'] == 'non-toxic'), 0)
            is_safe_for_kids = non_toxic_score > 0.9
        else:
            is_safe_for_kids = int(age_restriction) < 13
    else:
        # Анализ метаданных с использованием английской модели
        if description != 'Описание не найдено':
            # Анализ метаданных с использованием английской модели
            prediction = classifier_eng(description, return_all_scores=True)
            # Вывод предсказания для отладки
            print(f"Prediction: {prediction}")
            # Проверяем категорию 'toxic' и её вероятность
            toxic_score = next(
                (item['score'] for item in prediction[0] if item['label'] == 'toxic'), 0)
            is_safe_for_kids = toxic_score < 0.1
        else:
            is_safe_for_kids = int(age_restriction) < 13
    end_time = time.time()  # Конец отсчета времени
    elapsed_time = end_time - start_time  # Вычисление затраченного времени
    print(f"Is safe for kids: {is_safe_for_kids}")
    return {
        "title": title,
        "description": description,
        "age_restriction": age_restriction,
        "is_safe_for_kids": is_safe_for_kids,
        'elapsed_time': elapsed_time
    }


class AnalyzeRequest(BaseModel):
    text: str


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
        raise HTTPException(
            status_code=500, detail=f"Model loading failed: {str(e)}")

    system_tokens = get_system_tokens(model)
    tokens = system_tokens

    user_message = f"Проанализируй текст видео и определи к какому классу оно относится.Классы видео: 1. Насилие 2. Непристойный контент (материалы сексуального характера) 3. Наркотики и алкоголь 4. Вульгарность и ненормативная лексика 5. Опасное поведение (вредные действия и ненадежный контент) 6. Дискриминация и ненависть 7. Шокирующий контент 8. Унижающий или провокационный контент 9. Пособничество недобросовестной деятельности 10. Запрещенные материалы, связанные с азартными играми. 11. Детские мультфильмы 12. Взрослые мультфильмы 13. Наука 14. Танцы 15. Кулинария 16. Путешествия 17. Спорт 18. Музыка 19. Образование 20. Документальные 21. Комедия 22. Драма 23. Ужасы 24. Фантастика 25. Боевики 26. Приключения 27. Мистика 28. Фэнтези 29. История 30. Криминал 31. Мелодрама 32. Семейные 33. Аниме 34. Триллеры 35. Реалити-шоу 36. Видео-игры 37. Мода и одежда 38. Политика Текст видео - {request.text}"
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


def split_text(text, max_length):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunks.append(" ".join(words[i:i + max_length]))
    return chunks


@app.post("/transcribe")
async def transcribe(video: UploadFile = File(None), url: str = Form(None)):
    if video:
        temp_video_path = f"temp_{video.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await video.read())
    elif url:
        temp_video_path = await download_video(url)
    else:
        raise HTTPException(
            status_code=400, detail="No video file or URL provided.")

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

    # Определяем язык транскрипции
    text = transcription["text"]
    language = detect(text)
    text_chunks = split_text(text, MAX_TOKEN_LENGTH)
    for chunk in text_chunks:
        if language == 'ru':
            # Анализ текста с использованием русской модели
            prediction = classifier_rus(chunk, return_all_scores=True)
            print(f"Prediction: {prediction}")
            toxic_score = next(
                (item['score'] for item in prediction[0] if item['label'] == 'non-toxic'), 0)
            is_toxic = toxic_score >= 0.8

        else:
            # Анализ текста с использованием английской модели
            prediction = classifier_eng(chunk, return_all_scores=True)
            print(f"Prediction: {prediction}")
            toxic_score = next(
                (item['score'] for item in prediction[0] if item['label'] == 'toxic'), 0)
            is_toxic = toxic_score >= 0.2
            is_toxic = not is_toxic

    return {"transcription": text, "is_toxic": is_toxic}


@app.post("/analyze_video")
# Анализ метаданных с использованием русской модели
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
                # Выбор максимальной вероятности для каждого класса
                class_words[label] = max(class_words[label], top1_confidence)

    cap.release()

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    return {"class_words": class_words}


class VideoURL(BaseModel):
    url: str


@app.post("/metadata")
async def analyze_video_metadata(video: VideoURL):
    try:
        metadata = get_video_metadata(video.url)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Загрузка данных и обучение модели (этот код остается таким же, как и в исходном коде)
df = pd.read_csv('./model/dataset_1.csv', encoding='cp1251', header=0, sep=';')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('russian'))


def preprocess_text(text):
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()
    tokens = word_tokenize(text, language='russian')
    filtered_tokens = [token for token in tokens if token not in stop_words]
    text = ' '.join(filtered_tokens)
    return text


df['text'] = df['title'] + ' ' + df['description']
df['text'] = df['text'].apply(preprocess_text)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['genre'], test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
pipeline.fit(X_train, y_train)


def predict_genre(title, description):
    text = preprocess_text(title + ' ' + description)
    return pipeline.predict([text])[0]


def get_video_metadata2(video_url):
    response = requests.get(video_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title_tag = soup.find('meta', property='og:title')
    description_tag = soup.find('meta', property='og:description')
    title = title_tag['content'] if title_tag else 'Название не найдено'
    description = description_tag['content'] if description_tag else 'Описание не найдено'
    title = str(title)
    description = str(description)
    return title, description


@app.post("/classify")
async def predict_video_genre(video: VideoURL):
    title, description = get_video_metadata2(video.url)
    predicted_genre = predict_genre(title, description)
    return {"title": title, "predicted_genre": predicted_genre}
