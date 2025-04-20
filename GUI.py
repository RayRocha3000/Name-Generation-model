import customtkinter as ctk
from PIL import Image
import subprocess


win = ctk.CTk()
win.title("Nemesis.AI")
win.geometry("500x700")

win.configure(bg='#000000')
img = ctk.CTkImage(Image.open("Nem.jpg"),size=(300,150))
imglabel=ctk.CTkLabel(win, image=img, text='')
imglabel.place(x=100,y=10)

title=ctk.CTkLabel(win, text="NEMESIS.AI",font=("Copperplate Gothic",40,'bold'),text_color="#3eb0b7")
title.place(x=135,y=170)

box=ctk.CTkTextbox(win,width=400,height=350,corner_radius=6,state='disabled')
box.place(x=50,y=240)

def names():
    box.configure(state='normal')
    box.delete("0.0","end")
    subprocess.run(["python",'NameGM.py'])
    file=open('Nemesis_Output.txt','r')
    string=""
    lines=file.readlines()
    for line in lines:
        string+=line+'\n'
    box.insert("0.0",string)
    box.configure(state='disabled')
    file.close()

button=ctk.CTkButton(win, text="Generate",fg_color="#c52e82",corner_radius=6,font=('Copperplate Gothic',20,'bold'),command=names)
button.place(x=180,y=630)

win.mainloop()