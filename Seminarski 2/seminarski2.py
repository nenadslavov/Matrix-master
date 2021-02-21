import cv2
from cv2 import *
import tkinter as tk
from tkinter import *
import numpy as np
from numpy import array, dtype
from PIL import Image, ImageTk

root = tk.Tk()
matrica = Frame(root, bg='silver')

value11 = Entry(matrica, bg="white", fg="grey", width=7)
value11.grid(column=1, row=1)
value12 = Entry(matrica, bg="white", fg="grey", width=7)
value12.grid(column=2, row=1)
value13 = Entry(matrica, bg="white", fg="grey", width=7)
value13.grid(column=3, row=1)
value14 = Entry(matrica, bg="white", fg="grey", width=7)
value14.grid(column=4, row=1)

value21 = Entry(matrica, bg="white", fg="grey", width=7)
value21.grid(column=1, row=2)
value22 = Entry(matrica, bg="white", fg="grey", width=7)
value22.grid(column=2, row=2)
value23 = Entry(matrica, bg="white", fg="grey", width=7)
value23.grid(column=3, row=2)
value24 = Entry(matrica, bg="white", fg="grey", width=7)
value24.grid(column=4, row=2)

value31 = Entry(matrica, bg="white", fg="grey", width=7)
value31.grid(column=1, row=3)
value32 = Entry(matrica, bg="white", fg="grey", width=7)
value32.grid(column=2, row=3)
value33 = Entry(matrica, bg="white", fg="grey", width=7)
value33.grid(column=3, row=3)
value34 = Entry(matrica, bg="white", fg="grey", width=7)
value34.grid(column=4, row=3)

value41 = Entry(matrica, bg="white", fg="grey", width=7)
value41.grid(column=1, row=4)
value42 = Entry(matrica, bg="white", fg="grey", width=7)
value42.grid(column=2, row=4)
value43 = Entry(matrica, bg="white", fg="grey", width=7)
value43.grid(column=3, row=4)
value44 = Entry(matrica, bg="white", fg="grey", width=7)
value44.grid(column=4, row=4)

matrica.grid(columnspan=4, rowspan=4)


def matrice():
    if (value41.get() != ""):
        inverzna.delete(0, "end")
        determinanta.delete(0, "end")
        adjungova.delete(0, "end")
        b = array([
            [value11.get(), value12.get(), value13.get(), value14.get()],
            [value21.get(), value22.get(), value23.get(), value24.get()],
            [value31.get(), value32.get(), value33.get(), value34.get()],
            [value41.get(), value42.get(), value43.get(), value44.get()]
        ], dtype='int')
        detB = np.linalg.det(b)
        invB = np.linalg.inv(b)
        adjB = [[i for i in range(len(b))] for j in range(len(b))]
        for i in range(len(b)):
            for j in range(len(b)):
                adjB[i][j] = (-1)*(i+j)*detB
        inverzna.insert(0, invB)
        determinanta.insert(0, detB)
        adjungova.insert(0, adjB)
    else:
        inverzna.delete(0, "end")
        determinanta.delete(0, "end")
        adjungova.delete(0, "end")
        a = array([
            [value11.get(), value12.get(), value13.get()],
            [value21.get(), value22.get(), value23.get()],
            [value31.get(), value32.get(), value33.get()]
        ], dtype='int')
        print(a)
        detA = np.linalg.det(a)
        invA = np.linalg.inv(a)
        #invAp = np.linalg.inv(np.transpose(a[np.newaxis])*a)
        adjA = [[i for i in range(len(a))] for j in range(len(a))]
        for i in range(len(a)):
            for j in range(len(a)):
                adjA[i][j] = (-1)*(i+j)*detA
        # inverzna.insert(0, s1)
        # inverzna.insert("end", s2)
        # inverzna.insert("end", s3)
        inverzna.insert(0, invA)
        determinanta.insert(0, detA)
        adjungova.insert(0, adjA)


btn = Button(root, text="potvrdi", command=matrice)
btn.grid(column=1, row=5)

# ispis
lblInv = Label(root, text="Inverzna")
lblInv.grid(column=1, row=8)
inverzna = Listbox(root, width=260, height=5)
inverzna.grid(column=1, row=9, padx=15)
lblDet = Label(root, text="determinanta")
lblDet.grid(column=1, row=10)
determinanta = Listbox(root, width=260, height=5)
determinanta.grid(column=1, row=11, padx=15)
lblAdj = Label(root, text="adjungova")
lblAdj.grid(column=1, row=12)
adjungova = Listbox(root, width=260, height=5)
adjungova.grid(column=1, row=13, padx=15)


def gammaTransform():
    cascFacePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascFacePath)
    image = cv2.imread("img2.jpg", 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=25,
        minSize=(45, 45),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Izlaz", image)
    cv2.waitKey(0)

# image2 = (int(spin.get()) - image1)


spin = Spinbox(root, from_=0, to=255, command=gammaTransform)
spin.grid(column=1, row=15)

root.mainloop()
