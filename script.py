import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from agent import create_agent_executor, manage_memory
from langchain_core.messages import AIMessage, HumanMessage
from supabase import create_client

from dotenv import load_dotenv
import os

import google.generativeai as genai


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_gemini   = genai.GenerativeModel("gemini-1.5-flash")

supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

load_dotenv()

speech_client = ElevenLabs(
    api_key=os.getenv("11LABS_API_KEY"),
)

recognizer = sr.Recognizer()
agent_executor = create_agent_executor()


def get_speech_input():
    with sr.Microphone(device_index=0) as source:
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

pastMessage = supabase.table('message').select("*").eq('id', 1).execute().data[0]['message']

chat_history = []

opening = "Hello! I am Teddy, your friendly talking teddy bear. What's up Kid?"
chat_history.append(AIMessage(content=opening))
print(opening)
speak(opening)

response = supabase.table('chat_history').delete().neq('id', 0).execute()
print("Chat history cleared in DB")

response = (
        supabase.table("chat_history")
        .insert({"user": 0, "message": opening})
        .execute()
    )

print("Inserted opening message in DB")


while True:
    # Manage chat history
    new_memory = manage_memory(chat_history, k=5)
    if new_memory:
        chat_history = new_memory

    # Get input and print user question
    question = get_speech_input()
    print(f"You: {question}")
    result = agent_executor.invoke({"input": question, "chat_history": chat_history})

    print(f"Teddy: {result['output']}")

    # Insert user and AI messages into chat history
    supabase.table("chat_history").insert({"user": 1, "message": question}).execute()
    supabase.table("chat_history").insert({"user": 0, "message": result['output']}).execute()

    # Generate notification for parent
    notifLLM = llm_gemini.generate_content(f"""
    From the child and AI response between their conversation, 
    is there something the parent should be notified of?
    Child: {question}
    AI: {result['output']}
    
    If no, then just return "No notification."
    """)
    
    # Insert notification if applicable
    if notifLLM.text not in ["No notification", "No notification."]:
        supabase.table("notifications").insert({"notification": notifLLM.text}).execute()
    
    # Fetch chat history from database
    chat_history_db = supabase.table("chat_history").select("*").execute()

    # Generate mood sentiment and engagement
    mood_sentiment = llm_gemini.generate_content(f"""
    Decide the overall mood sentiment of the child.
    
    Don't format, don't tell me why. Just give me the overall mood sntiment.
    Give 2 sentences.   
    {chat_history_db}
    """)

    engagement = llm_gemini.generate_content(f"""
    Decide the overall engagement of the child and the AI.
    
    Don't format, don't tell me why. Just give me the overall engagement.
    Give 2 sentences.
    {chat_history_db}
    """)

    print(notifLLM.text)
    print(engagement.text)

    # Update mood and engagement
    supabase.table("widget_data").update({
        "mood": mood_sentiment.text, 
        "engagement": engagement.text
    }).eq("id", 1).execute()
    
    currentMessage = supabase.table('message').select("*").eq('id', 1).execute().data[0]['message']
    if currentMessage != pastMessage:
        pastMessage = currentMessage
        speak("Message from your parent." + currentMessage)

    # Speak AI response
    speak(result["output"])
    
    


    
    
    
    
    
    
    
