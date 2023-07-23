# Importing Necessary Libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np

# Loading the Model
from keras.models import load_model
eye_model = load_model('Eye_Detection.h5')
bald_model = load_model('Bald_Detection.h5')
child_model = load_model('Child_Detection.h5')

# Initializing the GUI
root = tk.Tk()
root.geometry('800x600')
root.minsize(600,600)
root.maxsize(1920,1080)
root.configure(background="skyblue")
root.title("Pankaj's Eye Colour Detection")

#TODO: root,file_path, colours

# Initializing the Labels 
f1 = Frame(root, bg="grey", borderwidth=6, relief=SUNKEN)
f1.pack(side=TOP,fill=X)
f2 = Frame(root, bg="silver", borderwidth=6, relief=SUNKEN)
f2.pack(side=LEFT,fill=Y)
f3 = Frame(root, bg="silver", borderwidth=6, relief=SUNKEN)
f3.pack(side=RIGHT,fill=Y)
f4 = Frame(root, bg="silver", borderwidth=6, relief=SUNKEN)
f4.pack(side=BOTTOM,fill=X)
f5 =Frame(root, bg="silver", borderwidth=6, relief=SUNKEN)
f5.pack(side=TOP,pady=20,padx=20)
label0 = Label(f4,text="  " ,background="silver", font=('lucida', 30, "bold")).pack()
label01 = Label(f2,text="   " ,background="gray64", font=('lucida', 15, "bold")).pack()
label02= Label(f3,text="   " ,background="gray64", font=('lucida', 15, "bold")).pack()

label1 = Label(f5, background="#CDCDCD", font=('lucida', 15, "bold"))
label2 = Label(f5, background="#CDCDCD", font=('lucida', 15, 'bold'))
label3 = Label(f5, background="#CDCDCD", font=('lucida', 15, 'bold'))
sign_image = Label(root)
label1.configure(text='Eye   :  ',foreground="red")
label2.configure(text='Bald   :  ',foreground="blue")
label3.configure(text='Child :  ',foreground="green")


# Defining Detect function which detects the eye colour,bald and child of the person in an image using the model
def Detect(file_path):
    global Label_packed
    image = Image.open(file_path)
    image = image.resize((64,64))
    image = np.array(image)
    image = np.array([image]) / 255
    p_eye = eye_model.predict(image)
    p_bald = bald_model.predict(image)
    p_child = child_model.predict(image)
    eye_pred = int(np.argmax(p_eye)) 
    bald_pred = int(np.argmax(p_bald)) 
    child_pred = int(np.argmax(p_child)) 
    eye_classes=['Black','Blue', 'Brown','Green']
    bald_classes=["No", "Yes"]
    child_classes=["No", "Yes"]
    print("Predicted Eye colour is   : " + eye_classes[eye_pred] )
    print("Predicted Bald or not is  : " + bald_classes[bald_pred])
    print("Predicted Child or not is : " + child_classes[child_pred])
    label1.configure(foreground="red", text=f"Eye :  {eye_classes[eye_pred]}")
    label2.configure(foreground="blue", text=f"Bald  :  {bald_classes[bald_pred]}")
    label3.configure(foreground="green", text=f"Child :  {child_classes[child_pred]}")

# Defining Show_detect button function
def show_Detect_button(file_path):
    try:
        Detect_b = Button(f4, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
        Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        Detect_b.place(relx=0.55, rely=0.1)
        
    except:
        pass

# Defining Upload Image Function
def upload_image():
    try:
        file_path = filedialog.askopenfilename(filetypes =[('Image Files', '*.jpg')])
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((root.winfo_width() / 2.25), (root.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        label1.configure(text='Eye   :  ',foreground="red")
        label2.configure(text='Bald   :  ',foreground="blue")
        label3.configure(text='Child :  ',foreground="green")

        sign_image.configure(image=im)
        sign_image.image = im
        
        show_Detect_button(file_path)

    except:
        pass

upload = Button(f4, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=5)
upload.place(relx=0.3, rely=0.1)
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)
label3.pack(side="bottom", expand=True)
heading = Label(f1, text="Eye Bald & Child Detector", pady=10, font=('arial', 20, "bold"),)
heading.configure(background="silver", foreground="black")
heading.pack(fill=X)


root.mainloop()
