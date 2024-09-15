from langchain.agents import tool
from pydub import AudioSegment
from pydub.playback import play
from twilio.rest import Client
from video_utils import VisionHelper
from dotenv import load_dotenv
from bucket_utils import BucketUtils

import os
import uuid

load_dotenv()

message_client = Client(
    username=os.getenv("TWILIO_ACCOUNT_SID"), password=os.getenv("TWILIO_AUTH_TOKEN")
)

@tool
def analyze_vision(prompt: str, media: str) -> str:
    """
    Records webcam feed for 3 seconds or takes a snapshot and analyzes the content based on the prompt. 
    Media is either 'video' or 'photo'.
    """
    print("-------------------------")
    print(f"Instruction: {prompt}")

    if media == "video":
        file_name = f"snapshots/videos/{uuid.uuid4()}.mp4"

        helper = VisionHelper(save_path=file_name)
        bucketUtilClient = BucketUtils()

        print("Recording video...")
        helper.record_video()
        print("Analyzing video...")
        bucketUtilClient.upload_blob(file_name, file_name)
        response = helper.send_video_to_gemini(prompt)
        print(f"Activity: {response}")
        print("-------------------------")

    elif media == "photo":
        file_name = f"snapshots/photos/{uuid.uuid4()}.jpg"

        helper = VisionHelper(save_path=file_name)
        bucketUtilClient = BucketUtils()

        print("Snapshot taken...")
        helper.take_snapshot()
        print("Analyzing photo...")
        bucketUtilClient.upload_blob(file_name, file_name)
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


TOOLS = [analyze_vision, play_music, send_alert]
