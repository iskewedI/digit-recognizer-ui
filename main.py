import gradio as gr
import requests
import cv2
import os

# URL of the Flask Recognize API
recognize_api_url = "http://127.0.0.1:5000/recognize"

# Read sample images into memory
sample_images = [cv2.imread(os.path.join("images", file)) for file in os.listdir("images")]

def handle_sample_select(evt: gr.SelectData):
    # Get the selected image from the gallery in grayscale
    image = cv2.imread(evt.value["image"]["path"], cv2.IMREAD_GRAYSCALE)

    # Write the image to disk for testing purposes
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

with gr.Blocks(title="Digit Classifier") as demo:
    # Gallery component to display the sample images
    gallery = gr.Gallery(sample_images, label="Samples", rows=4, columns=6, object_fit="contain")

    # Textbox component to display the predicted result
    result = gr.Textbox(label="Result: ")

    # Image editor component to allow the user to draw on the canvas
    image_editor = gr.ImageEditor()

    # Handlers
    # When the user draws and press the apply button on the canvas, the predict function is called
    image_editor.change(predict, inputs=image_editor, outputs=result)

    # When the user selects a sample image, the handle_sample_select function is called
    gallery.select(handle_sample_select, None, outputs=result)

demo.launch()