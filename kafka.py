from confluent_kafka import Producer, Consumer, KafkaError
import json
import asyncio

producer_config = {
    'bootstrap.servers': 'localhost:9092'
}
producer = Producer(producer_config)


def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]: {msg.value().decode("utf-8")}')


consumer_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(consumer_config)
consumer.subscribe(['video-metadata', 'video-transcription'])


def send_message_to_kafka(topic, message):
    producer.produce(topic, message.encode('utf-8'), callback=delivery_report)
    producer.poll(0)


async def kafka_consumer_task():
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(msg.error())
                break
        message = msg.value().decode('utf-8')
        print(f'Received message: {message}')
        await process_kafka_message(msg.topic(), message)


async def process_kafka_message(topic, message):
    if topic == 'video-metadata':
        await process_video_metadata(message)
    elif topic == 'video-transcription':
        await process_video_transcription(message)


async def process_video_metadata(message):
    metadata = json.loads(message)
    print(f'Processing metadata: {metadata}')
    # Дополнительная логика обработки метаданных


async def process_video_transcription(message):
    transcription = json.loads(message)
    print(f'Processing transcription: {transcription}')
    # Дополнительная логика обработки транскрипции
