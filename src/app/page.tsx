"use client";
import React, { useState } from "react";

const VideoAnalyzer = () => {
    const [videoFile, setVideoFile] = useState<File | null>(null);
    const [videoUrl, setVideoUrl] = useState("");
    const [transcription, setTranscription] = useState("");
    const [dangerousWords, setDangerousWords] = useState("");
    const [classWords, setClassWords] = useState([]);

    const handleVideoFileChange = (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        if (event.target.files && event.target.files.length > 0) {
            setVideoFile(event.target.files[0]);
            setVideoUrl(""); // Clear URL if a file is selected
        }
    };

    const handleVideoUrlChange = (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        setVideoUrl(event.target.value);
        setVideoFile(null); // Clear file if a URL is entered
    };

    const handleTranscribe = async () => {
        const formData = new FormData();
        if (videoFile) {
            formData.append("video", videoFile);
        } else if (videoUrl) {
            formData.append("url", videoUrl);
        } else {
            alert("Please provide a video file or URL");
            return;
        }

        const response = await fetch("http://127.0.0.1:8000/transcribe", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        setTranscription(data.transcription);
    };

    const handleAnalyze = async () => {
        if (transcription) {
            const response = await fetch("http://127.0.0.1:8000/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: transcription }),
            });
            const data = await response.json();
            setDangerousWords(data.dangerous_words);
        }
    };

    const handleAnalyzeVideo = async () => {
        const formData = new FormData();
        if (videoFile) {
            formData.append("video", videoFile);
        } else if (videoUrl) {
            formData.append("url", videoUrl);
        } else {
            alert("Please provide a video file or URL");
            return;
        }

        try {
            const response = await fetch(
                "http://127.0.0.1:8000/analyze_video",
                {
                    method: "POST",
                    body: formData,
                }
            );

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const data = await response.json();

            // Преобразуем объект в массив кортежей (ключ, значение) для удобства
            const classWordsArray = Object.entries(data.class_words);

            // Сортируем массив по значению (в порядке убывания)
            classWordsArray.sort((a, b) => b[1] - a[1]);

            // Получаем только ключи (названия классов) из отсортированного массива
            const sortedClassWords = classWordsArray.map(([key, _]) => key);

            setClassWords(sortedClassWords);
        } catch (error) {
            console.error("Error analyzing video:", error);
            // Здесь можно обработать ошибку, например, показать сообщение пользователю
        }
    };

    const containsDangerousWords = /Обычное|видео/i.test(dangerousWords);

    return (
        <div className="container mx-auto p-4">
            <div className="mb-4">
                <input
                    type="file"
                    accept="video/*"
                    onChange={handleVideoFileChange}
                    className="border border-gray-300 rounded-md p-2"
                />
                <span className="ml-2">или</span>
                <input
                    type="text"
                    placeholder="Введите ссылку на видео с YouTube"
                    value={videoUrl}
                    onChange={handleVideoUrlChange}
                    className="border text-black border-gray-300 rounded-md p-2 ml-2"
                />
            </div>
            <div className="mb-4">
                <button
                    onClick={handleTranscribe}
                    className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mr-2"
                >
                    Транскрибировать
                </button>
                <button
                    onClick={handleAnalyze}
                    className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded mr-2"
                >
                    Проанализировать
                </button>
                <button
                    onClick={handleAnalyzeVideo}
                    className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded"
                >
                    Проанализировать видео
                </button>
            </div>
            <div className="mb-4 flex col">
                <h2 className="text-lg font-bold mb-2">Транскрипция:</h2>
                <p className="bg-gray-100 p-4 text-black text-center flex-wrap rounded-md">
                    {transcription}
                </p>
            </div>
            <div className="mb-4 flex col">
                <h2 className="text-lg font-bold mb-2">Класс видео:</h2>
                <p className="bg-gray-100 p-4 text-black text-center flex-wrap rounded-md">
                    {classWords.join(", ")}
                </p>
            </div>
            <div className="mb-4 flex col">
                <h2 className="text-lg font-bold mb-2">Что в видео:</h2>
                <p className="bg-gray-100 p-4 text-black text-center flex-wrap rounded-md">
                    {dangerousWords}
                </p>
            </div>
            <div className="mb-4 flex col">
                <h2 className="text-lg font-bold mb-2">Что в видео:</h2>
                {containsDangerousWords ? (
                    <p className="bg-green-500 text-white p-4 text-center flex-wrap rounded-md">
                        Видео пригодно для детского просмотра
                    </p>
                ) : (
                    <p className="bg-red-500 text-white p-4 text-center flex-wrap rounded-md">
                        Видео не пригодно для детского просмотра
                    </p>
                )}
            </div>
        </div>
    );
};

export default VideoAnalyzer;
