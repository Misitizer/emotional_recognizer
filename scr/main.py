#from keras.models import load_model
import recognize_face, reshape_photo
import cv2
# from keras import backend as K
# K.common.set_image_dim_ordering('th')

def swish_activation(x):
    return (K.sigmoid(x) * x)

image_path = r'D:\diplom\images\Nastya4.jpg'
image = cv2.imread(image_path)

cropped = recognize_face.cropFace(image)
recognize_face.save_faces(cropped)
reshape_photo.scale_image(input_image_path = r'D:\diplom\faces\1.jpg',
             output_image_path = r'D:\diplom\faces\1.jpg',
             width=48,
             height= 48)

face_path = r'D:\diplom\faces\1.jpg'
face = cv2.imread(face_path, 0)

f = face.reshape(1, 48, 48, 1)

reloaded = load_model('emotional_model.h5', custom_objects={'swish_activation': swish_activation})

emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
predict = reloaded.predict(f)
print(emotion[np.argmax(predict)])


fig, (ax1, ax2) = plt.subplots(1,2)

names = np.arange(7)
ax1.bar(names, height=predict[0]*100)
ax1.set_xticks(names, emotion)
ax2.imshow(face.reshape((48,48)), interpolation = 'none', cmap = 'gray')
plt.show()