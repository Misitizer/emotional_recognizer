#!/usr/bin/env python3

from tensorflow.keras.models import load_model
import recognize_face, reshape_photo
import cv2
from keras import backend as K
K.common.set_image_dim_ordering('th')
from matplotlib import pyplot as plt
import numpy as np
import telebot
import os

def swish_activation(x):
    return (K.sigmoid(x) * x)

bot = telebot.TeleBot('your token')

reloaded = load_model(r'C:\Users\kdavydov\PycharmProjects\untitled\scr\emotional_model.h5',
                      custom_objects={'swish_activation': swish_activation})


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
	bot.reply_to(message, "Отправь свое фото")

@bot.message_handler(content_types=['photo'])
def handle_docs_document(message):

    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    scr = r'C:\Users\kdavydov\PycharmProjects\untitled\image\1.jpg'

    with open(scr, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.reply_to(message, 'Фото добавлено')

    image_path = r'C:\Users\kdavydov\PycharmProjects\untitled\image\1.jpg'
    image = cv2.imread(image_path)
    cropped, faceNum = recognize_face.cropFace(image)
    recognize_face.save_faces(cropped)

    for i in range(faceNum):
        face_path = r'C:\Users\kdavydov\PycharmProjects\untitled\faces\{}.jpg'.format(str(i+1))
        face = open(face_path, 'rb')

        bot.send_message(chat_id= message.chat.id, text='Смотрите кого нашел')
        bot.send_photo(chat_id = message.chat.id, photo= face)

        reshape_photo.scale_image(input_image_path = r'C:\Users\kdavydov\PycharmProjects\untitled\faces\{}.jpg'.format(str(i+1)),
                    output_image_path = r'C:\Users\kdavydov\PycharmProjects\untitled\faces\{}.jpg'.format(str(i+1)),
                    width = 48,
                    height = 48)
        bot.send_message(chat_id=message.chat.id, text = 'Пытаюсь понять эмоцию')
        face_path = r'C:\Users\kdavydov\PycharmProjects\untitled\faces\{}.jpg'.format(str(i+1))
        face = cv2.imread(face_path, 0)
        f = face.reshape(1, 48, 48, 1)


        emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predict = reloaded.predict(f)

        names = np.arange(7)
        plt.bar(names, height=predict[0])
        plt.xticks(names, labels=emotion)
        plt.savefig(r'C:\Users\kdavydov\PycharmProjects\untitled\emotion\{}.jpg'.format(str(i+1)))
        bot.send_message(chat_id=message.chat.id, text = emotion[np.argmax(predict)])
        plot = open(r'C:\Users\kdavydov\PycharmProjects\untitled\emotion\{}.jpg'.format(str(i + 1)), 'rb')
        bot.send_photo(chat_id=message.chat.id, photo=plot)





bot.polling()