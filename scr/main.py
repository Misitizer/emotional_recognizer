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

reloaded = load_model('emotional_model.h5', custom_objects={'swish_activation': swish_activation})

emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
predict = reloaded.predict(f)
print(emotion[np.argmax(predict)])

names = np.arange(7)
plt.bar(names, height=predict[0])
plt.xticks(names, ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
plt.show()