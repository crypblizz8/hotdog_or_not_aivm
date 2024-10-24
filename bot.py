from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
from slack_sdk.signature import SignatureVerifier
import io

import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import aivm_client as aic
import matplotlib.pyplot as plt

app = Flask(__name__)

# Initialize the Slack client with your bot token
# Add your Slack TOKEN
slack_client = WebClient(token="XXX")

# Initialize the signature verifier
# Add your Slack VERIFIER TOKEN
signature_verifier = SignatureVerifier("XXX")

dataset_path = './hotdog_data'

MODEL_NAME = "LeNet5HotDogOrNot4" 

## Transform for Lenet5
transform = transforms.Compose([
    transforms.Resize((28, 28)),  
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (1.0,))
])

def is_valid_slack_request(request):
    return signature_verifier.is_valid(
        body=request.get_data(),
        timestamp=request.headers.get("X-Slack-Request-Timestamp"),
        signature=request.headers.get("X-Slack-Signature")
    )

def return_just_transformed_data(file_url):
    response = requests.get(file_url, headers={"Authorization": f"Bearer {"XXX"}"})
    response.raise_for_status()  # Raise an exception for bad status codes
    content_type = response.headers.get('Content-Type', '')
    if not content_type.startswith('image/'):
        raise ValueError(f"Unexpected content type: {content_type}")
    
    image = Image.open(BytesIO(response.content))
    image = image.convert('RGB')
    img_tensor = transform(image)

    return img_tensor

## TODO: ADD SLACK TOKEN
def return_transformed_image(file_url):
    try:
        response = requests.get(file_url, headers={"Authorization": f"Bearer {"XXX"}"})
        response.raise_for_status()  # Raise an exception for bad status codes
        
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            raise ValueError(f"Unexpected content type: {content_type}")
        
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        img_tensor = transform(image)
        to_pil = transforms.ToPILImage()
        result_image = to_pil(img_tensor)

        return result_image
    
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except UnidentifiedImageError as e:
        print(f"Error identifying image: {e}")
        print(f"First 100 bytes of content: {response.content[:100]}")
        return None
    except ValueError as e:
        print(f"Error with content type: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing image: {e}")
        return None
    
def upload_bot_response(channel_id, user_id, transformed_image, thread_ts, prediction_result):
    img_byte_arr = io.BytesIO()
    transformed_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = slack_client.files_upload_v2(
        channels=channel_id,
        file=img_byte_arr,
        initial_comment=f"<@{user_id}> Here's your transformed image + it is {prediction_result}",
        thread_ts=thread_ts
    )

def initialize_model():
    global MODEL_INITIALIZED
    try:
        aic.upload_lenet5_model("./best_hotdog_model.pth", MODEL_NAME)
        print(f"Model {MODEL_NAME} uploaded successfully")
        MODEL_INITIALIZED = True
    except Exception as e:
        if "Model LeNet5HotDogOrNot4 already exists" in str(e):
            print(f"Model {MODEL_NAME} already exists, using existing model")
            MODEL_INITIALIZED = True
        else:
            print(f"Error initializing model: {str(e)}")
            MODEL_INITIALIZED = False

initialize_model()

@app.route("/slack/events", methods=["POST"])
def slack_events():
    if not is_valid_slack_request(request):
        return jsonify({"error": "Invalid request"}), 403

    event_data = request.json
    
    if "challenge" in event_data:
        return jsonify({"challenge": event_data["challenge"]})

    if "event" in event_data:
        event = event_data["event"]
        
        if event.get("type") == "app_mention" and "files" in event:
            channel_id = event["channel"]
            user_id = event["user"]
            thread_ts = event.get("thread_ts", event["ts"])

            file = event["files"][0] 
            file_url = file["url_private"]
    
            transformed_image = return_transformed_image(file_url)

            if transformed_image and MODEL_INITIALIZED:
                try:
                    transformed_image_for_prediction = return_just_transformed_data(file_url)
                    encrypted_input = aic.LeNet5Cryptensor(transformed_image_for_prediction.reshape(1, 1, 28, 28))
                    prediction = aic.get_prediction(encrypted_input, MODEL_NAME)
                    prediction_result = "HOTDOG" if torch.argmax(prediction).item() == 0 else "NOT HOTDOG"
                    upload_bot_response(channel_id, user_id, transformed_image, thread_ts, prediction_result)
                except Exception as e:
                    error_message = f"<@{user_id}> Sorry, I couldn't process your image. Error: {str(e)}"
                    slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=error_message
                    )
            elif not transformed_image:
                error_message = f"<@{user_id}> Sorry, I couldn't process your image. Please try again."
                slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=error_message
                )
            elif not MODEL_INITIALIZED:
                error_message = f"<@{user_id}> Sorry, the prediction model is not initialized. Please try again later."
                slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=error_message
                )
            
    return "", 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
