import os
import time
import cv2
import google.generativeai as genai


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
vision = genai.GenerativeModel("gemini-1.5-flash")


class VisionHelper:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.fps = 24
        self.frame_width = 640
        self.frame_height = 480

        os.makedirs("snapshots/videos", exist_ok=True)
        os.makedirs("snapshots/photos", exist_ok=True)

    def record_video(self, duration: int = 3):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Webcam not found!")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.save_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )

        start_time = time.time()
        while int(time.time() - start_time) < duration:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                cv2.imshow("Recording...", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return self.save_path

    def take_snapshot(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Webcam not found!")

        ret, frame = cap.read()
        if ret:
            cv2.imwrite(self.save_path, frame)
            cv2.imshow("Snapshot", frame)
            cv2.waitKey(1)
        else:
            raise Exception("Failed to take snapshot")

        cap.release()
        cv2.destroyAllWindows()

        return self.save_path

    def mod_prompt(self, prompt: str):
        return prompt + "Answer in one sentence. "

    def send_video_to_gemini(self, prompt):
        vid_file = genai.upload_file(self.save_path)
        while vid_file.state.name == "PROCESSING":
            time.sleep(2)
            vid_file = genai.get_file(vid_file.name)

        if vid_file.state.name == "FAILED":
            raise ValueError(f"Failed to upload video: {vid_file.name}")
        else:
            response = vision.generate_content([vid_file, self.mod_prompt(prompt)])
            genai.delete_file(vid_file.name)
            return response.text

    def send_image_to_gemini(self, prompt):
        img_file = genai.upload_file(self.save_path)
        response = vision.generate_content([img_file, self.mod_prompt(prompt)])
        genai.delete_file(img_file.name)
        return response.text
