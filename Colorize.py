import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import filedialog, font, messagebox
from PIL import Image, ImageTk
import threading

# Define directories and model paths
DIR = r"C:\Users\sowja\Downloads\Colorizing-black-and-white-images-using-Python-master\Colorizing\images"
PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "pts_in_hull.npy")
MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")

# Load the deep learning model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Configure model specific layers
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Global variables to store colored and enhanced images
colorized = None
enhanced = None
video_frames = []


def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        colorize_image(file_path)


def select_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        colorize_video(file_path)


def colorize_image(image_path):
    global colorized
    image = cv2.imread(image_path)
    colorized = process_frame(image)
    display_image(colorized)


def colorize_video(video_path):
    global video_frames
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    def process_video():
        global colorized
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            color_frame = process_frame(frame)
            video_frames.append(color_frame)
            display_image(color_frame)
        cap.release()

    threading.Thread(target=process_video).start()


def process_frame(frame):
    scaled = frame.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
    L = cv2.split(lab)[0]
    colorized_frame = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_LAB2BGR)
    colorized_frame = np.clip(colorized_frame, 0, 1)
    colorized_frame = (255 * colorized_frame).astype("uint8")
    return colorized_frame


def display_image(image):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.resize(
        (screen_width // 2, screen_height // 2), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    display_label.config(image=photo)
    display_label.image = photo


def save_media():
    global colorized, video_frames
    if colorized is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[
            ("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("BMP files", "*.bmp"), ("MP4 files", "*.mp4")])
        if file_path:
            if file_path.endswith(".mp4") and video_frames:
                height, width, layers = video_frames[0].shape
                size = (width, height)
                out = cv2.VideoWriter(
                    file_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
                for frame in video_frames:
                    out.write(frame)
                out.release()
                messagebox.showinfo("Info", f"Video saved as {file_path}")
            else:
                cv2.imwrite(file_path, colorized)
                messagebox.showinfo("Info", f"Image saved as {file_path}")
    else:
        messagebox.showwarning("Warning", "No media to save.")


def enhance_image():
    global colorized, enhanced
    if colorized is not None:
        enhanced = cv2.GaussianBlur(colorized, (0, 0), 3)
        enhanced = cv2.addWeighted(colorized, 1.5, enhanced, -0.5, 0)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=0)
        display_image(enhanced)

# Main function to construct the GUI


def main():
    global root, customFont, display_label
    root = tk.Tk()
    root.title("Image and Video Colorization GUI")
    customFont = font.Font(family="Helvetica", size=12, weight="bold")

    heading_label = tk.Label(root, text="Image and Video Colorization", font=font.Font(
        family="Helvetica", size=16, weight="bold"), bg='#e6e6fa', fg='#4b0082')
    heading_label.pack(pady=10)

    button_frame = tk.Frame(root, bg='#e6e6fa')
    button_frame.pack(pady=20)

    btn_image = tk.Button(button_frame, text="Load Image", command=select_image, font=customFont,
                          bg='#b19cd9', fg='white', padx=10, pady=5, relief=tk.FLAT)
    btn_image.grid(row=0, column=0, padx=5)

    btn_video = tk.Button(button_frame, text="Load Video", command=select_video, font=customFont,
                          bg='#b19cd9', fg='white', padx=10, pady=5, relief=tk.FLAT)
    btn_video.grid(row=0, column=1, padx=5)

    btn_enhance = tk.Button(button_frame, text="Enhance Image", command=enhance_image, font=customFont,
                            bg='#b19cd9', fg='white', padx=10, pady=5, relief=tk.FLAT)
    btn_enhance.grid(row=0, column=2, padx=5)

    btn_save = tk.Button(button_frame, text="Save Media", command=save_media, font=customFont,
                         bg='#b19cd9', fg='white', padx=10, pady=5, relief=tk.FLAT)
    btn_save.grid(row=0, column=3, padx=5)

    display_label = tk.Label(root, bg='#e6e6fa')
    display_label.pack(pady=20)

    root.configure(bg='#e6e6fa')
    root.mainloop()


if __name__ == "__main__":
    main()
