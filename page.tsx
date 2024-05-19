"use client";
import React, { useState } from "react";

const VideoAnalyzer = () => {
    const [videoFile, setVideoFile] = useState<File | null>(null);
    const [videoUrl, setVideoUrl] = useState("");
    const [transcription, setTranscription] = useState("");
    const [dangerousWords, setDangerousWords] = useState("");
    const [classWords, setClassWords] = useState<string[]>([]);
    const [metadata, setMetadata] = useState<any>({});
    const [toxic, setToxic] = useState<boolean>(false);
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
        setTranscription(data.transcription)
        setToxic(data.is_toxic)
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

    const handleAnalyzeMetadata = async () => {
        if (videoUrl) {
            try {
                const response = await fetch("http://127.0.0.1:8000/metadata", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ url: videoUrl }),
                });

                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }

                const data = await response.json();
                setMetadata(data);
            } catch (error) {
                console.error("Error analyzing metadata:", error);
                // Здесь можно обработать ошибку, например, показать сообщение пользователю
            }
        } else {
            alert("Please provide a URL for metadata analysis");
        }
    };
    const [classifiedGenre, setClassifiedGenre] = useState("");
    const containsDangerousWords = /Обычное|видео/i.test(dangerousWords);
    const handleClassify = async () => {
        if (videoUrl) {
            try {
                const response = await fetch("http://127.0.0.1:8000/classify", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ url: videoUrl }),
                });

                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }

                const data = await response.json();
                setClassifiedGenre(data.predicted_genre);
            } catch (error) {
                console.error("Error classifying video:", error);
                // Здесь можно обработать ошибку, например, показать сообщение пользователю
            }
        } else {
            alert("Please provide a URL for video classification");
        }
    };
    return (
        <div className="container mx-auto p-4">
            <div className="mb-4 flex items-center">
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
                    className="border text-black border-gray-300 rounded-md p-2 ml-2 w-1/2"
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
                    className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded mr-2"
                >
                    Проанализировать видео
                </button>
                <button
                    onClick={handleAnalyzeMetadata}
                    className="bg-yellow-500 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded mr-2"
                >
                    Анализировать метаданные
                </button>
                <button
                    onClick={handleClassify}
                    className="bg-orange-500 hover:bg-orange-700 text-white font-bold py-2 px-4 rounded mr-2"
                >
                    Классифицировать видео
                </button>
            </div>
            <div className="mb-4">
                <h2 className="text-lg font-bold mb-2">Метаданные:</h2>
                <p className="bg-gray-100 p-4 text-black rounded-md">
                    {metadata.title && (
                        <>
                            <strong>Title:</strong> {metadata.title} <br />
                        </>
                    )}
                    {metadata.description && (
                        <>
                            <strong>Description:</strong> {metadata.description}{" "}
                            <br />
                        </>
                    )}
                    {metadata.is_safe_for_kids !== undefined && (
                        <>
                            <strong>Safe for Kids:</strong>{" "}
                            {metadata.is_safe_for_kids ? "Yes" : "No"} <br />
                        </>
                    )}
                    {metadata.elapsed_time && (
                        <>
                            <strong>Analysis Time:</strong>{" "}
                            {metadata.elapsed_time} seconds
                        </>
                    )}
                </p>
            </div>
            <div className="mb-4">
                <h2 className="text-lg font-bold mb-2">Транскрипция:</h2>
                <p className="bg-gray-100 p-4 text-black rounded-md">
                    {transcription}
                    {toxic}
                </p>
            </div>
            <div className="mb-4">
                <h2 className="text-lg font-bold mb-2">Предметы и действия в кадре:</h2>
                <p className="bg-gray-100 p-4 text-black rounded-md">
                    {classWords.join(", ")}
                </p>
            </div>
            <div className="mb-4">
                <h2 className="text-lg font-bold mb-2">Класс видео:</h2>
                <p className="bg-gray-100 p-4 text-black rounded-md">
                    {dangerousWords}
                </p>
            </div>
            <div className="mb-4">
                <h2 className="text-lg font-bold mb-2">
                    Пригодность для детей:
                </h2>
                {toxic ? (
                    <p className="bg-green-500 text-white p-4 text-center rounded-md">
                        Видео пригодно для детского просмотра
                    </p>
                ) : (
                    <p className="bg-red-500 text-white p-4 text-center rounded-md">
                        Видео не пригодно для детского просмотра
                    </p>
                )}
            </div>
        </div>
    );
};

export default VideoAnalyzer;
