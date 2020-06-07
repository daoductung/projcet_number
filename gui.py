import speech_recognition as sr
from tkinter import *
from tkinter import messagebox
from api import *
from PIL import Image, ImageTk

r = sr.Recognizer()


def record_file():
    with sr.Microphone() as source:
        audio = r.listen(source)
        file_record = 'record.wav'
        with open(path_record + '\\' + file_record, 'wb') as f:
            f.write(audio.data)
            f.close()
    so = detect(file_record, load_ann())
    txt_result = 'So ban vua ghi am la ' + str(so)
    messagebox.showinfo('Ket qua', txt_result)


def client_exit():
    exit()


def train_data():
    features, number = read_data_file_wav(path_file_title)
    train(features)
    messagebox.showinfo('Thong bao', 'Thanh cong')


window = Tk()
window.title('Project record by Duong')
window.geometry("350x200")

load = Image.open("icon.jpg")
load.thumbnail((100, 100), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)

img = Label(image=render)
img.image = render
img.place(x=0, y=0)

btn = Button(window, text="Record", bg="orange", fg="red", width=10, command=record_file)

btn.place(x=130, y=50)

btn = Button(window, text="Train", bg="orange", fg="red", width=10, command=train_data)

btn.place(x=130, y=90)

btn = Button(window, text="Quit", bg="orange", fg="red", width=10, command=client_exit)

btn.place(x=130, y=130)

window.mainloop()
