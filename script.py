import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from agent import create_agent_executor, manage_memory
from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv
import os


load_dotenv()

speech_client = ElevenLabs(
    api_key=os.getenv("11LABS_API_KEY"),
)

recognizer = sr.Recognizer()
agent_executor = create_agent_executor()


def get_speech_input():
    with sr.Microphone(device_index=2) as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google_cloud(
            audio, credentials_json="clipcraft-account.json"
        )
    except sr.UnknownValueError:
        return "Sorry, I didn't understand that."
    except sr.RequestError:
        return "Could not request results; check your network."


def speak(text):
    audio = speech_client.generate(
        text=text,
        voice="Jessica",
        model="eleven_multilingual_v2",
    )
    play(audio)


chat_history = []

opening = "Hello! I am Teddy, your friendly talking teddy bear. What's up Kid?"
chat_history.append(AIMessage(content=opening))
print(opening)
speak(opening)


while True:
    new_memory = manage_memory(chat_history, k=5)
    if new_memory:
        chat_history = new_memory

    question = get_speech_input()
    print(f"You: {question}")
    result = agent_executor.invoke({"input": question, "chat_history": chat_history})

    print(f"Teddy: {result['output']}")
    speak(result["output"])

    chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=result["output"]),
        ]
    )
