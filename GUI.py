import tkinter as tk
import final_test as test
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
import final_test as test
import numpy as np
import time
import os


def main():
  root = tk.Tk()
  root.title("PREDICT")
  root.geometry("500x470+550+150" )
  root.resizable(0, 0)

  
  def input_image():
    #initialdir 對話框開啟的目錄, title對話框的標題, filetypes找尋的副檔名
    img_path = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files","*.*"),("jpeg files","*.jpg"), ("png files","*.png"), ("gif files","*.gif")))
    img= cv2.cvtColor(cv2.resize(cv2.imread(img_path), (500, 400)), cv2.COLOR_BGR2RGB)
    #Image.fromarray 將陣列轉為圖像
    img = ImageTk.PhotoImage(Image.fromarray(img))
    #image_canvas.create_image(0,0,anchor = NW, image = img)
    image_canvas.create_image(250,200, image = img)
    image_canvas.img = img
    
    #time.sleep(1)

    
    width_size=224
    height_size=224

    model=test.Model(width_size,height_size)
    model.load_model("./model_2/")

    PicDir= img_path
    NArray_2D  = test.Image_to_Array(PicDir,width_size,height_size)  
    NArray_2D = test.to_tensor([NArray_2D])
    prediction=model.predict(NArray_2D)

    print(prediction)
    ans = np.argmax(prediction)
    if ans == 0 :
      lb3 = tk.Label(root,text = "正常拉",font=("標楷體",12))
      lb3.place(x = 1, y = 25)
      print("good")
    elif ans == 1 :
      lb4 = tk.Label(root,text = "不正常",font=("標楷體",12))
      lb4.place(x = 1, y = 25)
      
      print("GG")
    pass
  
    
    
    

  pass
  lb1 = tk.Label(root,text = "INPUT IMAGE:",font=("標楷體",12))
  lb1.grid(columnspan=1,row=0,sticky=tk.W)

  en1 = tk.Entry(root)
  en1.grid(column=1,row=0,sticky=tk.W)
  
  btn1= tk.Button(text ="Input",font=("標楷體",12),command =input_image) #,command =re
  btn1.place(x = 250, y = 0)
  
  #btn2 = tk.Button(text ="PREDICT",font=("標楷體",12),command =predict_image) #,command =re
  #btn2.place(x = 305, y = 0)
  
  image_canvas = tk.Canvas(root, bg = 'white',height = 400, width = 480)
  image_canvas.place(x = 2, y = 45)
  
  #lb2 = tk.Label(root,text = "INPUT IMAGE:",font=("標楷體",12))
  #lb2.place(x = 1, y = 30)

  root.mainloop()  

pass

if __name__ == "__main__":  
  main()
pass