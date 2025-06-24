# image swap
import cv2
import insightface
import matplotlib.pyplot as plt

app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640)) 

swapper = insightface.model_zoo.get_model('inswapper_128.onnx')

source_img = cv2.imread(r'images/Image.png')
target_img = cv2.imread(r'images/target.jpg')

source_face = app.get(source_img)[0]
target_faces = app.get(target_img)

result = target_img.copy()
for face in target_faces:
    result = swapper.get(result, face, source_face, paste_back=True)

plt.imshow(result[:, :, ::-1])
plt.axis('off')
plt.title("Face Swapped Output")
plt.show()

cv2.imwrite('output222.jpg', result)
print("Face swapped image saved as output.jpg")
