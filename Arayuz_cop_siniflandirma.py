from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk

def predict_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0  # Normalization

    # Resmin şeklini (1, 224, 224, 3) olarak genişletme
    img = np.expand_dims(img, axis=0)

    model = load_model('/Users/anilturkmen/Desktop/yuzde_doksan11_bessinif.keras')

    p = model.predict(img)
    labels = {0: 'Metal', 1: 'Cam', 2: 'Kağıt', 3: 'Karton', 4: 'Plastik', 5: 'Çöp'}
    predicted_class = labels[np.argmax(p)]

    return predicted_class

def openImg():
    img_path = filedialog.askopenfilename(title='Select Image')
    result = predict_img(img_path)

    new_img = ImageTk.PhotoImage(Image.open(img_path).resize((224, 224)))
    img_show.config(image=new_img)
    img_show.photo_ref = new_img

    predict_result.config(text="Seçtiğiniz görsel " + result +" Türüne aittir.")

if __name__ == '__main__':
    img_path = '/Users/anilturkmen/Desktop/Gerı_donusum.png'

    root = Tk()
    root.title('Geri Dönüşüm Sınıflandırıcı')
    root.geometry('700x500')

    img = ImageTk.PhotoImage(Image.open(img_path))

    instruction = Label(root, text="Lütfen bir görüntü açın ve sınıflandırma yapın.", font=('Times', 32))
    instruction.pack()

    img_show = Label(root, image=img)
    img_show.pack(expand="yes")

    predict_result = Label(root, text="", font=('Times', 25))
    predict_result.pack()

    open_file = Button(root, text="Açınız", command=openImg)
    open_file.pack()

    root.mainloop()
