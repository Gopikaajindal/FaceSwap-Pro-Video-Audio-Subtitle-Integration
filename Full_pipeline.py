import cv2
import insightface
import subprocess
import whisper
import os
import json
import moviepy.editor as mp

source_img_path = 'images/Image.png'
input_video_path = 'videos/input_video1.mp4'
extracted_audio = 'videos/extracted_audio.wav'
temp_video_no_audio = 'videos/temp_no_audio.avi'
final_output_video = 'videos/final_output7777.avi'
word_json_file = 'videos/words.json'

print("Initializing face swapper...")
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx')

source_img = cv2.imread(source_img_path)
source_face = app.get(source_img)[0]

print("Starting face swap...")
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(temp_video_no_audio, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

frame_count = 0
frames = []

while frame_count<=300:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    result_frame = frame.copy()

    for face in faces:
        result_frame = swapper.get(result_frame, face, source_face, paste_back=True)

    frames.append(result_frame)
    out.write(result_frame)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Swapped {frame_count} frames...")

cap.release()
out.release()

print("Face-swapped video saved.")

print("Transcribing audio and extracting word-level timestamps...")
model = whisper.load_model("base")
result = model.transcribe(input_video_path, language="en", word_timestamps=True)

words = []
for segment in result['segments']:
    if 'words' in segment:
        words.extend(segment['words'])

with open(word_json_file, "w", encoding="utf-8") as f:
    json.dump(words, f, indent=2)

print("Rendering subtitles with word-level highlighting...")
cap = cv2.VideoCapture(temp_video_no_audio)
out_highlighted = cv2.VideoWriter('videos/temp_highlighted.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

with open(word_json_file, 'r') as f:
    data = json.load(f)

def wrap_words(words, max_width, font, font_scale, thickness):
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        size = cv2.getTextSize(test_line, font, font_scale, thickness)[0][0]
        if size < max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return lines

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    padding = 10

    for i in range(len(data)):
        word_obj = data[i]
        start = word_obj["start"]
        end = word_obj["end"]
        word = word_obj["word"].strip()

        if start <= t <= end:
            line_words = [data[j]["word"].strip() for j in range(max(0, i-5), min(len(data), i+6))]
            lines = wrap_words(line_words, width - 100, font, font_scale, thickness)

            y_offset = height - (len(lines) * (30 + padding))

            for line in lines:
                line_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                x_start = (width - line_size[0]) // 2

                current_x = x_start
                for w in line.split():
                    color = (0, 255, 255) if w == word else (0, 0, 0)
                    cv2.putText(frame, w + " ", (current_x, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
                    text_width = cv2.getTextSize(w + " ", font, font_scale, thickness)[0][0]
                    current_x += text_width
                y_offset += 30 + padding
            break

    out_highlighted.write(frame)

cap.release()
out_highlighted.release()
print("Subtitled video saved with highlights.")

print("Extracting audio from original video...")
subprocess.run([
    'ffmpeg', '-y', '-i', input_video_path, '-q:a', '0', '-map', 'a', extracted_audio
], check=True)

print("Merging final audio with highlighted subtitle video...")
subprocess.run([
    'ffmpeg', '-y',
    '-i', 'videos/temp_highlighted.avi',
    '-i', extracted_audio,
    '-c:v', 'copy',
    '-c:a', 'aac',
    final_output_video
], check=True)

for f in [temp_video_no_audio, 'videos/temp_highlighted.avi', extracted_audio, word_json_file]:
    if os.path.exists(f):
        os.remove(f)

print(f"✅ All done! Final output at: {final_output_video}")
