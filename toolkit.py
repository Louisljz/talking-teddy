from langchain.agents import tool
from pydub import AudioSegment
from pydub.playback import play
from twilio.rest import Client
from video_utils import VisionHelper

import os
import uuid


message_client = Client(
    username=os.getenv("TWILIO_ACCOUNT_SID"), password=os.getenv("TWILIO_AUTH_TOKEN")
)

@tool
def analyze_vision(prompt: str, media: str) -> str:
    """Records webcam feed for 3 seconds or takes a snapshot and analyzes the content based on the prompt. Media is either 'video' or 'photo'."""
    print("-------------------------")
    print(f"Instruction: {prompt}")

    if media == "video":
        file_name = f"snapshots/videos/{uuid.uuid4()}.mp4"

        helper = VisionHelper(save_path=file_name)

        print("Recording video...")
        helper.record_video()
        print("Analyzing video...")
        response = helper.send_video_to_gemini(prompt)
        print(f"Activity: {response}")
        print("-------------------------")

    elif media == "photo":
        file_name = f"snapshots/photos/{uuid.uuid4()}.jpg"

        helper = VisionHelper(save_path=file_name)

        print("Snapshot taken...")
        helper.take_snapshot()
        print("Analyzing photo...")
        response = helper.send_image_to_gemini(prompt)
        print(f"Activity: {response}")
        print("-------------------------")

    return response


@tool
def play_music(title: str) -> str:
    """plays 30 sec short music, can be one of: study, chill, playful styles"""
    try:
        song = AudioSegment.from_file(f"music/{title}.mp3")
        play(song)
        return f"{title} song finished playing!"
    except Exception as e:
        return f"Error playing file: {str(e)}"


@tool
def send_alert(message: str) -> str:
    """sends a whatsapp message to the parent's emergency number"""
    message_client.messages.create(
        from_="whatsapp:+14155238886",
        body=message,
        to=f"whatsapp:{os.getenv('EMERGENCY_NUMBER')}",
    )
    return "Alert sent to parent!"


@tool
def check_reminders(placeholder: str) -> str:
    """gets the latest reminders set by parent, input is empty string"""
    # write an API call to get reminders from database
    reminders = [
        "Remember to pick up your toys when you're done playing",
        "Brush your teeth before bed",
        "Don't forget to do your science homework",
    ]
    return reminders


TOOLS = [analyze_vision, play_music, send_alert, check_reminders]
