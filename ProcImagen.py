from tkinter import * 
from tkinter import filedialog 
from tkinter import ttk 
import tkinter as tk 
import cv2
from numpy import * 
import numpy as np 
import matplotlib as matplotlib 
from matplotlib import pyplot as plt 
from PIL import Image, ImageTk 
import scipy.misc as scimis 
import imutils 
from mpl_toolkits.mplot3d import axes3d 
from os import remove


path_global = True; 
image_1_global = True; 
hsv_global = True; 
mask_global = True; 
B_global = True; 
B1_global = True; 
B2_global = True; 
magnitudFFT_global = True; 
magnitudFFT2_global = True; 
imagen_ecualizada_global = True; 
imagen_byn_global = True; 
blur_img_tk_global =True; 
median_img_tk_global=True; 
res1_global = True; 
res2_global = True;
blur_global = True; 
median_global = True; 
image_1_global_countour = True; 
contours_cv2_tk_global = True; 
cnts_global = True; 
alto_global = True; 
ancho_global = True; 
alto_adaptado_global= True; 
ancho_adaptado_global = True; 
topx, topy, botx, boty = 0, 0, 0, 0 
rect_id_global = None 
image_original_global = None 
canvas2_global = None 
pix_global = True; 
HayRecorte = 0; 

ventana1 = tk.Tk() 
ventana1.geometry("1200x700") 
ventana1.title("Herramienta de reconocimiento de imágenes")  
ventana1.resizable(0,0)  
 
ventana1.config(bg="#0B121D")  
ventana1.config(cursor="hand2")  
ventana1.config(relief="sunken")  
ventana1.config(bd=8)  
 
labeltitulo= Label(ventana1, text="Herramienta de reconocimiento de imágenes", padx=15,pady=15) 
labeltitulo.config(fg="white", bg="#0B121D",font=("Arial",18)) 
labeltitulo.pack(anchor=N) 
cuaderno1 = ttk.Notebook(ventana1,width=800, height=600) 
cuaderno1.place(x=200,y=50)
pagina1 = ttk.Frame(cuaderno1) 
cuaderno1.add(pagina1, text="Inicio", padding=10) 
 
label1 = ttk.Label(pagina1, text="Bienvenido \nInserte una imagen para tratar con ella.\n") 
label1.grid(column=0, row=0) 
boton1_abrir = ttk.Button(pagina1, text="Abrir imagen",command=choose)  
boton1_abrir.grid(column=0, row=1) 
 
canvas = tk.Canvas(pagina1) 
canvas.config(cursor="arrow") 
 
canvas.bind('<Button-1>', get_mouse_posn) 
canvas.bind('<B1-Motion>', update_sel_rect) 

label19 = Label(pagina1, text=None) 
label19.grid(column=0,row=4) 
 
boton2_limpiar = ttk.Button(pagina1, text="Limpiar",command=limpiar)  
boton2_limpiar.grid(column=0,row=3) 
 
pagina2 = ttk.Frame(cuaderno1) 
cuaderno1.add(pagina2, text="Tratamiento de la imagen",padding=10,state="disabled") 
comboLabel  =  ttk.Combobox(pagina2,  values=[  "FFT",  "Escala  de  grises",  "Filtrado  gaussiano", 
"Histograma", "Filtrado de Mediana", "Contorno imagen"]) 
comboLabel.current(1) 
comboLabel.bind("<<ComboboxSelected>>", imprime_label) 
comboLabel.place(x="20",y="20") 

label2 = ttk.Label(pagina2, text=None) 
label2.place(x="20", y="0") 
label20 = Label(pagina2, image=None) 
label20.place(x="320",y="20") 
label21 = ttk.Label(pagina2,text=None) 
label21.place(x="20", y="100") 
boton3_guardar = ttk.Button(pagina2, text="Guardar",command=save_file,state="disabled" )  
boton3_guardar.place(x="20", y="45") 
boton5_histograma = ttk.Button(pagina2, text="Imprimir Histograma", command=histograma) 

boton5_histograma.place(x="20", y="70") 
label_histograma_titulo = Label(pagina2, text=None) 
label_histograma_titulo.place(x="20", y="175") 
label_histograma = Label(pagina2, image=None) 
label_histograma.place(x="20", y="200")

pagina3 = ttk.Frame(cuaderno1) 
cuaderno1.add(pagina3, text="Contour / 3D",padding=10,state="disabled" ) 
label14 = ttk.Label(pagina3, text=None) 
label14.place(x="175", y="0") 
label15= ttk.Label(pagina3,image=None) 
label15.place(x="0", y="60") 
label16 = ttk.Label(pagina3, text=None) 
label16.place(x="15", y="550") 
label17_titulo = ttk.Label(pagina3, text=None) 
label17_titulo.place(x="550", y="0") 
label17 = ttk.Label(pagina3, image=None) 
label17.place(x="370", y="20") 
boton4_contour = ttk.Button(pagina3, text="Visualizacion 3D", command = contour_y_3d) 
boton4_contour.place(x="650", y="400")

ventana1.mainloop() 

def get_mouse_posn(event): 
    global topy, topx 
    topx, topy = event.x, event.y 
    
def update_sel_rect(event): 
    global rect_id_global 
    global topy, topx, botx, boty 
    botx, boty = event.x, event.y 
    canvas.coords(rect_id_global, topx, topy, botx, boty)
    
    
def contour_y_3d(): 
    global magnitudFFT_global 
    plt.clf() 
    plt.close() 
    x, y = magnitudFFT_global.shape 
    X = np.arange(0, x) 
    Y = np.arange(0, y) 
    xx, yy = np.meshgrid(Y, X)   
    fig2 = plt.figure() 
    ax = fig2.gca(projection='3d', facecolor="#F0F0F0")   
    ax.plot_surface(xx, yy, magnitudFFT_global, facecolor="#F0F0F0")   
    ax.contourf(xx, yy, magnitudFFT_global, zdir='x', offset=-5) 
    plt.show()
    
    
def histograma(): 
    global HayRecorte 
    global pix_global 
 
    if HayRecorte == 0: 
        image_1_global = cv2.imread(path_global, 1)  
    if HayRecorte == 1: 
        image_1_global = pix_global  
 
    fig = plt.figure() 
    histr = cv2.calcHist([image_1_global], [0], None, [256], [0, 256]) 
 
    plt.plot(histr) 
    fig.savefig('plot.png', facecolor="#F0F0F0") 
    image_1_global = cv2.imread('plot.png', 1)   
    image_1_global.shape   
    image = cv2.cvtColor(image_1_global, cv2.COLOR_BGR2RGB)   
    hsv = cv2.cvtColor(image_1_global, cv2.COLOR_BGR2HSV) 
    lw_range = np.array([0, 0, 0])   
    up_range = np.array([255, 255, 255])   
    mask= cv2.inRange(hsv, lw_range, up_range)   
    res = cv2.bitwise_and(image, image, mask=mask)   
     
    imagen_histograma = Image.fromarray(res).resize((280, 280), Image.ANTIALIAS) 
    img_histograma_label = ImageTk.PhotoImage(imagen_histograma)   
    label_histograma.configure(image=img_histograma_label) 
    label_histograma.image = img_histograma_label 
    remove("plot.png") 
    label_histograma_titulo.configure(text="Nº total píxeles - Rango píxeles") 
    plt.close(fig) 
    
    
def NoRecorta(): 
    global image_original_global 
    global canvas2_global 
    global B1_global 
    global HayRecorte 
    global ancho_global ,alto_global 
    global pix_global 
 
    HayRecorte = 0 
    img = ImageTk.PhotoImage(image_original_global) 
    canvas2_global.destroy() 
    canvas.img = img 
    canvas.create_image(0, 0, image=img, anchor=tk.NW) 
    canvas.place(x="320",y="20") 
    B1_global.configure(state="disabled") 
    data_original = image_original_global.size 
    alto_global = data_original[0] 
    ancho_global = data_original[1] 
    pix_global = np.array(image_original_global)
    
    
def Recorta(): 
    global image_1_global 
    global topy, topx, botx, boty 
    global cropped_img_label 
    global canvas2_global 
    global B1_global 
    global pix_global 
    global HayRecorte 
    global alto_global, ancho_global 
    global image_original_global
    
    if topx != 0 or topy != 0 or botx != 0 or boty != 0:  
        canvas.delete("all") 
        HayRecorte = 1 
        canvas2_global = tk.Canvas(pagina1) 
        area = (topx, topy, botx, boty) 
        cropped_img = image_original_global.crop(area) 
        data1 = cropped_img.size 
        alto_global = data1[0] 
        ancho_global = data1[1] 
        if alto_global < 550 and ancho_global < 550: 
            while ancho_global < 450 and alto_global < 450:  
                alto_global = alto_global * 1.02 
                ancho_global = ancho_global * 1.02 
        cropped_img1 = cropped_img.resize((int(alto_global),int(ancho_global)),Image.ANTIALIAS) 
        cropped_img_label = ImageTk.PhotoImage(cropped_img1) 
        canvas2_global.create_image(0, 0, image=cropped_img_label, anchor=tk.NW) 
        canvas2_global.place(x="320",y="20") 
        canvas2_global.config(width=int(alto_global), height=int(ancho_global)) 
        pix_global = np.array(cropped_img) 
        image_1_global = pix_global 
        B1_global.configure(state="normal") 
        
        
def limpiar(): 
    global B_global, B1_global, B2_global 
    global canvas2_global 
    global HayRecorte 
    try: 
        cuaderno1.tab(1, state="disabled") 
        cuaderno1.tab(2,state="disabled") 
        canvas.delete("all") 
        
        if HayRecorte == 1: 
            canvas2_global.destroy() 
         
        label19.configure(text=" ") 
        label_histograma_titulo.configure(text=" ") 
        label21.configure(text=" ") 
        label_histograma.configure(image=None) 
        label_histograma.image = None 
        label20.configure(image=None, relief = None) 
        label20.image=None 
        boton3_guardar.configure(state="disabled") 
         
        B_global.destroy() 
        B1_global.destroy() 
        B2_global.destroy() 
    except AttributeError: 
        print("No hay nada que borrar")
        
        
def ReconocimientoImagen(): 
    global path_global 
    global cropped_img_global 
    global imagen_global 
    global image_1_global 
    global hsv_global 
    global mask_global 
    global magnitudFFT2_global 
    global imagen_ecualizada_global 
    global imagen_byn_global 
    global blur_img_tk_global 
    global median_img_tk_global 
    global magnitudFFT_global 
    global res1_global 
    global blur_global 
    global median_global 
    global res2_global 
    global image_1_global_countour 
    global contours_cv2_tk_global 
    global cnts_global 
    global alto_adaptado_global , ancho_adaptado_global 
    global pix_global 
    global HayRecorte 
    
    
    cuaderno1.tab(1, state="normal") 
    cuaderno1.tab(2, state="normal") 
     
    imagen_recortada= Image.fromarray(pix_global) 
    imagen_recortada.save("Prueba.png") 
    image_2 = cv2.imread("Prueba.png", 0) 
 
    if HayRecorte == 1: 
        data_nuevo = image_2.shape 
        alto_nuevo = data_nuevo[0]         
        ancho_nuevo = data_nuevo[1] 
        while ancho_nuevo < 450 and alto_nuevo < 550:
            alto_nuevo = alto_nuevo * 1.02 
            ancho_nuevo = ancho_nuevo * 1.02 
    if HayRecorte == 0:  
        image_2 = cv2.imread(path_global, 0) 
        image_1_global = cv2.imread(path_global, 1)
        alto_nuevo = alto_adaptado_global  
        ancho_nuevo = ancho_adaptado_global 
    f = np.fft.fft2(image_2, [256,256])  
    fshift = np.fft.fftshift(f) 
    magnitudFFT_global = 20 * np.log(np.abs(fshift)) 
 
    magnitudFFT1_global  =  Image.fromarray(magnitudFFT_global).resize((350,  350), Image.ANTIALIAS) 
    magnitudFFT2_global = ImageTk.PhotoImage(magnitudFFT1_global) 
    gray_img = cv2.cvtColor(image_1_global, cv2.COLOR_BGR2GRAY) 
 
    hsv_global = cv2.cvtColor(image_1_global, cv2.COLOR_BGR2HSV) 
    lw_range = np.array([0, 0, 0])   
    up_range = np.array([255, 255, 255])   
    mask_global=cv2.inRange(hsv_global, lw_range, up_range) 
    
    res1_global = cv2.bitwise_and(gray_img, gray_img, mask=mask_global) 
 
    gray_img1  =  Image.fromarray(res1_global).resize((int(ancho_nuevo),  int(alto_nuevo)), Image.ANTIALIAS) 
    imagen_byn_global = ImageTk.PhotoImage(gray_img1)   
 
    blur_global = cv2.GaussianBlur(image_1_global, (5, 5), 0) 
    
    blur_img  = Image.fromarray(blur_global).resize((int(ancho_nuevo),int(alto_nuevo)),Image.ANTIALIAS) 
    blur_img_tk_global= ImageTk.PhotoImage(blur_img) 
     
    median_global = cv2.medianBlur(image_1_global, 5) 
    median_img  = Image.fromarray(median_global).resize((int(ancho_nuevo),int(alto_nuevo)),Image.ANTIALIAS) 
    median_img_tk_global = ImageTk.PhotoImage(median_img) 
    
    img_to_yuv = cv2.cvtColor(image_1_global, cv2.COLOR_BGR2YUV) 
    img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0]) 
    hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR) 
    res2_global  =  cv2.bitwise_and(hist_equalization_result,  hist_equalization_result, mask=mask_global) 
    hist_equalization_result2  =  Image.fromarray(res2_global).resize((int(ancho_nuevo), int(alto_nuevo)), Image.ANTIALIAS) 
    imagen_ecualizada_global = ImageTk.PhotoImage(hist_equalization_result2)  
    
    image_1_global_countour = image_1_global 
    gray = cv2.cvtColor(image_1_global_countour, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1] 
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE) 
    cnts_global = imutils.grab_contours(cnts) 
    for c in cnts_global: 
        cv2.drawContours(image_1_global_countour, [c], -1, (0, 255, 0), 2) 
    contours_cv2  =  Image.fromarray(image_1_global_countour).resize((int(ancho_nuevo), int(alto_nuevo)), Image.ANTIALIAS) 
    contours_cv2_tk_global = ImageTk.PhotoImage(contours_cv2) 
     
    label14.configure(text="Contour") 
    label17_titulo.configure(text="3D") 
    
    lw_range1 = np.array([0, 0, 0])   
    up_range1 = np.array([255, 255, 255])   
    x, y = magnitudFFT_global.shape 
    X = np.arange(0, x) 
    Y = np.arange(0, y) 
    xx, yy = np.meshgrid(Y, X)   
    fig = plt.figure() 
    contorno1 = plt.contour(xx, yy, magnitudFFT_global, alpha=0.9, levels=7) 
    plt.colorbar(contorno1) 
    
    fig.savefig('plot.png', facecolor="#F0F0F0") 
    imagen_contour = cv2.imread('plot.png', 1)   
    hsv_global1 = cv2.cvtColor(imagen_contour, cv2.COLOR_BGR2HSV) 
    imagen_cv2_contour = cv2.cvtColor(imagen_contour, cv2.COLOR_BGR2RGB)   
    mask_global1 = cv2.inRange(hsv_global1, lw_range1, up_range1)   
    res_global1 = cv2.bitwise_and(imagen_cv2_contour, imagen_cv2_contour, mask=mask_global1)   
    image_1_1 = Image.fromarray(res_global1).resize((400, 400), Image.ANTIALIAS) 
    img_1 = ImageTk.PhotoImage(image_1_1)  
    label15.configure(image=img_1) 
    label15.image = img_1 
    fig.clf() 
    plt.clf()
    
    fig = plt.figure() 
    ax = fig.gca(projection='3d', facecolor="#F0F0F0")   
    ax.plot_surface(xx, yy, magnitudFFT_global, facecolor="#F0F0F0")   
    ax.contourf(xx, yy, magnitudFFT_global, zdir='x', offset=-5) 
    fig.savefig('plot.png', facecolor="#F0F0F0") 
    surface1 = cv2.imread('plot.png', 1)   
    hsv_global2 = cv2.cvtColor(surface1, cv2.COLOR_BGR2HSV) 
    surface1_cv2 = cv2.cvtColor(surface1, cv2.COLOR_BGR2RGB 
    mask_global2 = cv2.inRange(hsv_global2, lw_range1, up_range1)   
    res_global2 = cv2.bitwise_and(surface1_cv2, surface1_cv2, mask=mask_global2)  
    image_surface = Image.fromarray(res_global2).resize((350, 350), Image.ANTIALIAS) 
    img_surface = ImageTk.PhotoImage(image_surface)   
    label17.configure(image=img_surface) 
    label17.image = img_surface
    
    remove('Prueba.png') 
    remove('plot.png') 
    fig.clf() 
    plt.clf() 
    plt.close(fig) 
    
                                
def choose()
    global B_global, B1_global, B2_global 
    global path_global 
    global cropped_img_global 
    global imagen_global 
    global image_1_global 
    global hsv_global 
    global mask_global 
    global funcion_global 
    global res_global 
    global ancho_global, alto_global 
    global alto_adaptado_global, ancho_adaptado_global
    global image_original_global 
    global pix_global 
    global rect_id_global 
    global HayRecorte 
    global canvas2_global 
    try: 
       if HayRecorte == 1: 
            canvas2_global.destroy() 
            path_global = filedialog.askopenfilename()  
            image_1_global = cv2.imread(path_global, 1) 
            data = image_1_global.shape  
            image_original_global = cv2.cvtColor(image_1_global, cv2.COLOR_BGR2RGB)  
            hsv_global = cv2.cvtColor(image_1_global, cv2.COLOR_BGR2HSV) 
            lw_range = np.array([0, 0, 0])  
            up_range = np.array([255, 255, 255])  
            mask_global = cv2.inRange(hsv_global, lw_range, up_range) 
            res_global= cv2.bitwise_and(image_original_global, image_original_global, mask=mask_global) 
            alto_global = data[0] 
            ancho_global = data[1] 
            while ancho_global > 450 or alto_global > 550: 
            	alto_global = alto_global * 0.9 
                ancho_global = ancho_global * 0.9
                                
            alto_adaptado_global = alto_global 
            ancho_adaptado_global = ancho_global 
            image_original_global = Image.fromarray(res_global).resize((int(ancho_global),int(alto_global)),    Image.ANTIALIAS) 
            pix_global = np.array(image_original_global) 
            imagen_global=image_original_global; 
            img = ImageTk.PhotoImage(image_original_global)  
            canvas.img = img   
            canvas.create_image(0, 0, image=img, anchor=tk.NW) 
            canvas.place(x="320",y="20") 
            canvas.config(width=int(ancho_global), height=int(alto_global)) 
            rect_id_global = canvas.create_rectangle(topx, topy, topx, topy,dash=(2, 2), fill='', outline='white') 
	        label19.configure(text="Informacion Imagen Original\nAlto: {} píxeles\nAncho: {} píxeles\nCanales: {} píxeles".format(data[0], data[1], data[2])) 
	        B_global = Button(pagina1, text="Recortar Imagen", command=Recorta) 
	        B_global.place(x=0,y=200) 
	        B1_global = Button(pagina1, text="Sin recortar", command= NoRecorta,state="disabled") 
	        B1_global.place(x=100,y=200) 
            B2_global = Button(pagina1, text="Comenzar", bg = "#F0F0F0" , command=ReconocimientoImagen) 
            B2_global.place(x=220, y=200) 
            rect_id_global = canvas.create_rectangle(topx, topy, topx, topy, dash=(2, 2), fill='', outline='white') 
    except AttributeError: 
        print("Error tipo NoneType")
                               
def save_file(): 
    try: 
        global magnitudFFT_global 
        global res1_global 
        global blur_global 
        global median_global 
        global res2_global 
        global image_1_global_countour 
 
        gray_img1 = Image.fromarray(res1_global) 
        file = filedialog.asksaveasfilename(filetypes=[("PNG",".png")],defaultextension=".png") 
        if comboLabel.get() == "FFT": 
        	matplotlib.image.imsave(str(file), magnitudFFT_global) 
 
        elif comboLabel.get() == "Escala de grises": 
            gray_img1.save(str(file)) 
 
        elif comboLabel.get() == "Filtrado gausiano": 
            matplotlib.image.imsave(str(file), blur_global) 
 
        elif comboLabel.get() == "Histograma": 
            matplotlib.image.imsave(str(file), res2_global) 
 
        elif comboLabel.get() == "Filtrado de Mediana": 
            matplotlib.image.imsave(str(file), median_global) 
                               
        elif comboLabel.get() == "Contorno imagen": 
            matplotlib.image.imsave(str(file), image_1_global_countour) 
    except ValueError: 
        print("No hay formato escogido") 
                               
                               
def imprime_label(event): 
    global magnitudFFT2_global  
    global imagen_ecualizada_global  
    global imagen_byn_global  
    global median_img_tk_global  
    global contours_cv2_tk_global  
    global cnts_global  
 
    label21.configure(text=" ") 
    boton3_guardar.configure(command=save_file, 
                             state="normal")  # state=disabled para deshabilitar 
    if comboLabel.get() == "FFT": 
        label20.configure(image=magnitudFFT2_global) 
        label20.image = magnitudFFT2_global 
        label2.configure(text="Imagen FFT") 
                               
    elif comboLabel.get() == "Escala de grises": 
        label2.configure(text="Imagen original en escala de grises") 
        label20.configure(image=imagen_byn_global) 
        label20.image = imagen_byn_global 
 
    elif comboLabel.get() == "Contorno imagen": 
        label20.configure(image=contours_cv2_tk_global) 
        label20.image = contours_cv2_tk_global 
        label2.configure(text="Contorno imagen original") 
        label21.configure(text="Número de Contour: " + str(len(cnts_global))) 
 
    elif comboLabel.get() == "Histograma":
                               
     label20.configure(image=imagen_ecualizada_global) 
        label20.image = imagen_ecualizada_global 
        label2.configure(text="Imagen histograma ecualizada") 
 
    elif comboLabel.get() == "Filtrado de Mediana": 
        label20.configure(image=median_img_tk_global) 
        label20.image = median_img_tk_global 
        label2.configure(text="Imagen filtrada por mediana") 
 
    elif comboLabel.get() == "Filtrado gaussiano": 
        label20.configure(image=blur_img_tk_global) 
        label20.image = blur_img_tk_global 
        label2.configure(text="Imagen filtrada gaussiana")
