import os
import cv2
import subprocess
import threading
import numpy as np
from faster_whisper import WhisperModel
import insightface

ffmpeg_path = "ffmpeg"

source_img_path = 'images/Image.png'
input_video_path = 'videos/input_video1.mp4'
temp_video_no_audio = 'videos/temp_no_audio.avi'
final_output_video = 'videos/output_with_audio.avi'  
extracted_audio = 'videos/extracted_audio.wav'
transcript_file = 'transcript.txt'

print("Initializing face analysis...")
face_app = insightface.app.FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx')

source_img = cv2.imread(source_img_path)
source_face = face_app.get(source_img)[0]

def extract_audio():
    print("Extracting audio...")
    extract_cmd = [
        ffmpeg_path,
        "-y", "-i", input_video_path,
        "-q:a", "0", "-map", "a", extracted_audio
    ]
    result = subprocess.run(extract_cmd)

    if not os.path.exists(extracted_audio):
        raise FileNotFoundError("Audio extraction failed: extracted_audio.wav not found")
    print("Audio extracted successfully.")

def process_video():
    print("Starting face swap...")
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_width = 1280
    new_height = 720

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (new_width, new_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (new_width, new_height))

        if frame_count % 3 == 0:
            faces = face_app.get(frame)
            for face in faces:
                frame = swapper.get(frame, face, source_face, paste_back=True)

        out.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Face-swapped video saved without audio.")

def transcribe_audio():
    print("Transcribing audio with Whisper...")
    model = WhisperModel("tiny", compute_type="int8")

    segments, _ = model.transcribe(extracted_audio, beam_size=5)
    full_transcript = ""

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_transcript += segment.text + " "

    with open(transcript_file, 'w', encoding="utf-8") as f:
        f.write(full_transcript.strip())
    print("Transcript saved to transcript.txt")

def combine_audio_video():
    print("Merging audio with video...")
    combine_cmd = [
        ffmpeg_path,
        "-y", "-i", temp_video_no_audio,
        "-i", extracted_audio,
        "-c:v", "copy",
        "-c:a", "mp3",  
        final_output_video
    ]
    subprocess.run(combine_cmd)
    print("Final video saved:", final_output_video)

if __name__ == "__main__":
    audio_thread = threading.Thread(target=extract_audio)
    video_thread = threading.Thread(target=process_video)

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()

    if os.path.exists(extracted_audio):
        transcribe_audio()
    else:
        raise RuntimeError("Audio file not found after extraction.")

    combine_audio_video()

    for f in [temp_video_no_audio, extracted_audio]:
        if os.path.exists(f):
            os.remove(f)
