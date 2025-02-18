import tkinter as tk
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageDraw

model = load_model("digit_recognition_model.h5")

window = tk.Tk()
window.title("Handwritten Digit Recognition")
window.geometry("600x700")  # Increased window size
window.configure(bg="#f5f5f5")

canvas_size = 400  # Increased canvas size

canvas = tk.Canvas(window, width=canvas_size, height=canvas_size, bg="black", bd=2, relief="ridge")
canvas.pack(pady=10)

image = Image.new("L", (canvas_size, canvas_size), "black")
draw = ImageDraw.Draw(image)

last_x, last_y = None, None
eraser_mode = False

def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw_digit(event):
    global last_x, last_y
    if last_x and last_y:
        color = "black" if eraser_mode else "white"
        canvas.create_line(last_x, last_y, event.x, event.y, fill=color, width=15)  # Increased brush size
        draw.line((last_x, last_y, event.x, event.y), fill=color, width=15)
    last_x, last_y = event.x, event.y

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, canvas_size, canvas_size), fill="black")
    result_label.config(text="Draw a digit and click 'Recognize'", fg="black")

def toggle_eraser():
    global eraser_mode
    eraser_mode = not eraser_mode
    eraser_button.config(bg="gray" if eraser_mode else "lightgreen")

def predict_digit():
    img = image.resize((28, 28))  
    img = np.array(img) / 255.0  
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100  

    result_label.config(text=f"Predicted: {digit} ({confidence:.2f}%)", fg="blue")

def preprocess_image(img):
    img = img.resize((28, 28))  # Resize to match MNIST format
    img = np.array(img) / 255.0  # Normalize
    
    # Center the digit
    if np.sum(img) == 0:
        return np.zeros((1, 28, 28, 1))  # Avoid division by zero
    
    img = img - np.mean(img)  # Mean normalization
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    return img

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_digit)

button_style = {"font": ("Arial", 14, "bold"), "width": 14, "height": 2, "bd": 3, "relief": "raised"}

predict_button = tk.Button(window, text="Recognize", command=predict_digit, bg="lightblue", **button_style)
predict_button.pack(pady=5)

clear_button = tk.Button(window, text="Clear", command=clear_canvas, bg="lightcoral", **button_style)
clear_button.pack(pady=5)

eraser_button = tk.Button(window, text="Eraser", command=toggle_eraser, bg="lightgreen", **button_style)
eraser_button.pack(pady=5)

result_label = tk.Label(window, text="Draw a digit and click 'Recognize'", font=("Arial", 16, "bold"), bg="#f5f5f5")
result_label.pack(pady=10)

window.mainloop()
