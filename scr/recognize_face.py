import cv2

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cropFace(image):
    roi_color = []
    face_casscade = cv2.CascadeClassifier(r'C:\Users\kdavydov\PycharmProjects\untitled\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_casscade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 100, 0), 2)
        roi_color.append(image[y:y + h , x:x + w])
    return roi_color, len(roi_color)

def save_faces(cropped):
    faces_path = r'C:\Users\kdavydov\PycharmProjects\untitled\faces'
    for i, face in enumerate(cropped):
        cv2.imwrite(faces_path+"\{}.jpg".format(str(i+1)), face)


# path = r'C:\Users\kdavydov\PycharmProjects\untitled\image\1.jpg'
# image = cv2.imread(path)
# cropped, n = cropFace(image)
# save_faces(cropped)
# print(n)
# for i, face in enumerate(cropped):
#     viewImage(face, str(i+1))
# # print(image_path)