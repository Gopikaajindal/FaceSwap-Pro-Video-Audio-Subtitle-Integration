import os
import cv2
import subprocess
import threading
from faster_whisper import WhisperModel
import insightface

# Paths
source_img_path = 'images/Image.png'
input_video_path = 'videos/input_video1.mp4'
temp_video_no_audio = 'videos/temp_no_audio.avi'
extracted_audio = 'videos/extracted_audio.wav'
transcript_srt = 'videos/subs.srt'
final_output_video = 'videos/output_with_audio_and_captions.mp4'

ffmpeg_path = "ffmpeg"

# ------------------------------
# 1. Initialize Face Swapper
# ------------------------------
print("Initializing face analysis and model...")
face_app = insightface.app.FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx')
source_img = cv2.imread(source_img_path)
source_faces = face_app.get(source_img)

if not source_faces:
    raise RuntimeError("No face detected in source image.")
source_face = source_faces[0]

# ------------------------------
# 2. Extract Audio
# ------------------------------
def extract_audio():
    print("Extracting audio...")
    cmd = [
        ffmpeg_path, "-y", "-i", input_video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        extracted_audio
    ]
    subprocess.run(cmd, check=True)
    print("Audio extracted.")

# ------------------------------
# 3. Process Video (Face Swap)
# ------------------------------
def process_video():
    print("Starting face swapping...")
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    new_width, new_height = 1280, 720

    out = cv2.VideoWriter(
        temp_video_no_audio,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (new_width, new_height)
    )

    frame_count = 0
    max_ = 300
    while frame_count<=max_:
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
    print("Face swap complete.")

# ------------------------------
# 4. Transcribe Audio to SRT
# ------------------------------
def transcribe_audio_to_srt():
    print("Transcribing with Whisper...")
    model = WhisperModel("tiny", compute_type="int8")
    segments, _ = model.transcribe(extracted_audio, beam_size=5, word_timestamps=True)

    def format_srt_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    srt_lines = []
    idx = 1

    for segment in segments:
        if not hasattr(segment, "words") or not segment.words:
            continue
        for word in segment.words:
            srt_lines.append(str(idx))
            srt_lines.append(f"{format_srt_time(word.start)} --> {format_srt_time(word.end)}")
            srt_lines.append(word.word.strip())
            srt_lines.append("")
            idx += 1

    if not srt_lines:
        raise RuntimeError("No transcribed words found.")

    with open(transcript_srt, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_lines))
    print("SRT file written.")

# ------------------------------
# 5. Merge Audio + Captions
# ------------------------------
def combine_audio_video_with_captions():
    print("Combining video, audio, and captions...")
    cmd = [
        ffmpeg_path,
        "-y", "-i", temp_video_no_audio,
        "-i", extracted_audio,
        "-vf", f"subtitles={transcript_srt}:force_style='FontSize=24,OutlineColour=&H80000000,BorderStyle=1'",
        "-c:v", "libx264", "-c:a", "aac",
        final_output_video
    ]
    subprocess.run(cmd, check=True)
    print("Final video saved:", final_output_video)

# ------------------------------
# 6. Run Pipeline
# ------------------------------
if __name__ == "__main__":
    # Step 1-2 in parallel
    t1 = threading.Thread(target=extract_audio)
    t2 = threading.Thread(target=process_video)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Step 3
    transcribe_audio_to_srt()

    # Step 4
    combine_audio_video_with_captions()

    # Cleanup temp files
    for f in [temp_video_no_audio, extracted_audio]:
        if os.path.exists(f):
            os.remove(f)
