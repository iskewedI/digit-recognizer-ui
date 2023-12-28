import gradio as gr
import requests
from skimage.transform import resize
import numpy as np
import cv2
import os

recognize_api_url = "http://127.0.0.1:5000/recognize"
sample_images = [cv2.imread(os.path.join("images", file)) for file in os.listdir("images")]

def handle_sample_select(evt: gr.SelectData):
    image = cv2.imread(evt.value["image"]["path"], cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("test_Sample.png", image)

    # Create JSON body payload for the POST request
    payload = {
        "image": image.tolist()
    }

    # Send POST request to the Flask Recognize API
    response = requests.post(recognize_api_url, json=payload)

    # Return the predicted result to be rendered in the UI
    return f"Predicted => {response.json()['result']}"

def predict(img):
    image = img["composite"]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize image to 28x28
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imwrite("test_Drawn.png", resized)

    # Create JSON body payload for the POST request
    payload = {
        "image": resized.tolist()
    }

    # Send POST request to the Flask Recognize API
    response = requests.post(recognize_api_url, json=payload)

    # Return the predicted result to be rendered in the UI
    return f"Predicted => {response.json()['result']}"

with gr.Blocks() as demo:
    gallery = gr.Gallery(sample_images, label="Samples", rows=4, columns=6, object_fit="contain")

    result = gr.Textbox(label="Result: ")

    image_editor = gr.ImageEditor()

    image_editor.change(predict, inputs=image_editor, outputs=result)

    gallery.select(handle_sample_select, None, outputs=result)

demo.launch()