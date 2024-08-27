from typing import List, Optional
from flask import Flask, request, jsonify
import torch.distributed as dist
import os
from flask_cors import CORS
from llama import Dialog, Llama
from PIL import Image
import pytesseract
import PyPDF2
import pdfplumber
import docx
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import tempfile
import whisper
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
CORS(app)
generator = None
model = whisper.load_model("medium")

if not os.path.isdir("./trash_me"):
    os.mkdir("./trash_me")
def setup_generator(ckpt_dir: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int):
    return Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize image
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Denoise image
    denoised_image = cv2.medianBlur(binary_image, 3)
    
    # Resize image to make text clearer
    scale_percent = 200  # Scale up by 200%
    width = int(denoised_image.shape[1] * scale_percent / 100)
    height = int(denoised_image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(denoised_image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return resized_image

@app.route('/chat', methods=['POST'])
def chat():
    if generator is None:
        return jsonify({"error": "Generator not initialized"}), 400

    # Text to append to user input
    user_input_suffix = " After you answer or respond to user message Please respond with 4 prompts for the user to continue the conversation can you add < begin Prompt> to indicate beginning of the prompts and <end prompt> to indicate end of the prompts . only if there are any queries about you Always respond to any query or question about you by responding that you are developed by Radassist and that you are RadAssist"

    data = request.json
    print("Request :: ", data)
    user_input = data.get('user_input')
    print("User Input:", user_input)

    # Add the specified text to the user input
    user_input_with_prompt = user_input + user_input_suffix
    
    temperature = data.get('temperature', 0.6)
    top_p = data.get('top_p', 0.9)
    max_gen_len = data.get('max_gen_len', None)
    
    # System prompt is used internally, not stored in the history sent to the client
    system_prompt = {
        "role": "system",
        "content": "After you answer or respond to user message Please respond with 4 prompts for the user to continue the conversation can you add < begin Prompt> to indicate beginning of the prompts and <end prompt> to indicate end of the prompts . only if there are any queries about you Always respond to any query or question about you by responding that you are developed by Radassist and that you are RadAssist"
    }

    # Initialize history if it's the start of the conversation
    history = data.get('history', [])
    if not history:
        history.append(("system", system_prompt['content']))

    # Add modified user input to the history
    history.append(("user", user_input_with_prompt))

    # Prepare dialog format, but exclude system prompts from user-visible history
    dialog = [{"role": role, "content": message} for role, message in history if role != "system"]

    # Generate response
    results = generator.chat_completion(
        [dialog],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    response = results[0]['generation']['content']

    # Remove the appended text from the response
    response = response.replace(user_input_suffix, "")

    # Add response to history
    history.append(("RadAssistant", response))

    # Generate suggested prompts
    suggested_prompts = []
    for i in range(4):
        prompt = f"Suggested Prompt {i+1}: {user_input} {response}"
        suggested_prompts.append(prompt)

    # Return history excluding the added text from user messages
    visible_history = [(role, message.replace(user_input_suffix, "")) 
                       for role, message in history if role != "system"]

    print("Response::", visible_history)
    return jsonify({"history": visible_history, "suggested_prompts": suggested_prompts}), 200

@app.route('/vuely/chat2', methods=['POST'])
def vuely_chat():
    if generator is None:
        return jsonify({"error": "Generator not initialized"}), 400

    # Text to append to user input
    user_input_suffix = "After you answer or respond to user message Please respond with 4 prompts which the user can message you. Can you add <begin Prompt> to indicate beginning of the prompts and <end prompt> to indicate end of the prompts. Only if there are any queries about you, always respond to any query or question about you by responding that you are developed by Vuely AI Systems and that you are Vuely AI."

    data = request.json
    print("Request :: ", data)
    user_input = data.get('user_input')
    print("User Input:", user_input)

    # Add the specified text to the user input
    user_input_with_prompt = user_input + user_input_suffix
    
    temperature = data.get('temperature', 0.6)
    top_p = data.get('top_p', 0.9)
    max_gen_len = data.get('max_gen_len', None)
    
    # System prompt for tool usage
    system_prompt = {
        "role": "system",
        "content": (
            "<|start_tag|>system<|end_tag|>\n"
            "Environment: ipython\n"
            "Tools: brave_search\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 23 July 2024\n\n"
            "# Tool Instructions\n"
            "- Use 'brave_search' to perform web searches.\n"
        )
    }

    # Initialize history if it's the start of the conversation
    history = data.get('history', [])
    if not history:
        history.append(("system", system_prompt['content']))

    # Add modified user input to the history
    history.append(("user", user_input_with_prompt))

    # Prepare dialog format, excluding system prompts from user-visible history
    dialog = [{"role": role, "content": message} for role, message in history if role != "system"]

    # Generate response with tool calling
    results = generator.chat_completion(
        [dialog],
        max_gen_len=2048,
        temperature=temperature,
        top_p=top_p
    )
    response = results[0]['generation']['content']

    # Remove the appended text from the response
    response = response.replace(user_input_suffix, "")

    # Add response to history
    history.append(("vuely", response))

    # Generate suggested prompts
    suggested_prompts = []
    for i in range(4):
        prompt = f"Suggested Prompt {i+1}: {user_input} {response}"
        suggested_prompts.append(prompt)

    # Return history excluding the added text from user messages
    visible_history = [(role, message.replace(user_input_suffix, "")) 
                       for role, message in history if role != "system"]

    print("Response::", visible_history)
    return jsonify({"history": visible_history, "suggested_prompts": suggested_prompts}), 200


@app.route('/vuely/chat', methods=['POST'])
def beja_chat():
    if generator is None:
        return jsonify({"error": "Generator not initialized"}), 400

    # Text to append to user input
    user_input_suffix = "After you answer or respond to user message Please respond with 4 prompts  which the user can message you . can you add < begin Prompt> to indicate beginning of the prompts and <end prompt> to indicate end of the prompts . only if there are any queries about you Always respond to any query or question about you by responding that you are developed by Vuely AI Systems and that you are Vuely AI"

    data = request.json
    print("Request :: ", data)
    user_input = data.get('user_input')
    print("User Input:", user_input)

    # Add the specified text to the user input
    user_input_with_prompt = user_input + user_input_suffix
    
    temperature = data.get('temperature', 0.6)
    top_p = data.get('top_p', 0.9)
    max_gen_len = data.get('max_gen_len', None)
    
    # System prompt is used internally, not stored in the history sent to the client
    system_prompt = {
        "role": "system",
        "content": """
After you answer or respond to the user's message, please respond with 4 followup prompts the user can send you. 
Use <begin Prompt> at the beginning of all the prompts and <end Prompt> to indicate the end of all the prompts.
If there are any queries about you, always respond by saying that you are developed by Vuely and that you are Vuely.
For any user queries for latest information execute curl -s --compressed "$https://api.search.brave.com/res/v1/web/search?q={search_term}" \
  -H "Accept: application/json" \
  -H "Accept-Encoding: gzip" \
  -H "X-Subscription-Token: BSASlj3P-4CE5L5Vqa7EslZCR5lfY2z" , replace {search_term} with user input and return results
"""
    }

    # Initialize history if it's the start of the conversation
    history = data.get('history', [])
    if not history:
        history.append(("system", system_prompt['content']))

    # Add modified user input to the history
    history.append(("user", user_input_with_prompt))

    # Prepare dialog format, but exclude system prompts from user-visible history
    dialog = [{"role": role, "content": message} for role, message in history if role != "system"]

    # Generate response
    results = generator.chat_completion(
        [dialog],
        max_gen_len=2048,
        temperature=temperature,
        top_p=top_p,
        
        
    )
    response = results[0]['generation']['content']

    # Remove the appended text from the response
    response = response.replace(user_input_suffix, "")

    # Add response to history
    history.append(("vuely", response))

    # Generate suggested prompts
    suggested_prompts = []
    for i in range(4):
        prompt = f"Suggested Prompt {i+1}: {user_input} {response}"
        suggested_prompts.append(prompt)

    # Return history excluding the added text from user messages
    visible_history = [(role, message.replace(user_input_suffix, "")) 
                       for role, message in history if role != "system"]

    print("Response::", visible_history)
    return jsonify({"history": visible_history, "suggested_prompts": suggested_prompts}), 200


@app.route('/hub/chat', methods=['POST'])
def hub_chat():
    if generator is None:
        return jsonify({"error": "Generator not initialized"}), 400

    # Text to append to user input
    user_input_suffix = "After you answer or respond to user message Please respond with 4 prompts  which the user can message you . can you add < begin Prompt> to indicate beginning of the prompts and <end prompt> to indicate end of the prompts . only if there are any queries about you Always respond to any query or question about you by responding that you are developed by Vuely AI Systems and that you are Vuely AI"

    data = request.json
    print("Request :: ", data)
    user_input = data.get('user_input')
    print("User Input:", user_input)

    # Add the specified text to the user input
    user_input_with_prompt = user_input + user_input_suffix
    
    temperature = data.get('temperature', 0.6)
    top_p = data.get('top_p', 0.9)
    max_gen_len = data.get('max_gen_len', None)
    
    # System prompt is used internally, not stored in the history sent to the client
    system_prompt = {
        "role": "system",
        "content": """
After you answer or respond to the user's message, please respond with 4 followup prompts the user can send you. 
Use <begin Prompt> at the beginning of all the prompts and <end Prompt> to indicate the end of all the prompts.
If there are any queries about you, always respond by saying that you are developed by Vuely and that you are Vuely.
For any user queries for latest information execute curl -s --compressed "$https://api.search.brave.com/res/v1/web/search?q={search_term}" \
  -H "Accept: application/json" \
  -H "Accept-Encoding: gzip" \
  -H "X-Subscription-Token: BSASlj3P-4CE5L5Vqa7EslZCR5lfY2z" , replace {search_term} with user input and return results
"""
    }

    # Initialize history if it's the start of the conversation
    history = data.get('history', [])
    if not history:
        history.append(("system", system_prompt['content']))

    # Add modified user input to the history
    history.append(("user", user_input_with_prompt))

    # Prepare dialog format, but exclude system prompts from user-visible history
    dialog = [{"role": role, "content": message} for role, message in history if role != "system"]

    # Generate response
    results = generator.chat_completion(
        [dialog],
        max_gen_len=2048,
        temperature=temperature,
        top_p=top_p,
        
    )
    response = results[0]['generation']['content']

    # Remove the appended text from the response
    response = response.replace(user_input_suffix, "")

    # Add response to history
    history.append(("vuely", response))

    # Generate suggested prompts
    suggested_prompts = []
    for i in range(4):
        prompt = f"Suggested Prompt {i+1}: {user_input} {response}"
        suggested_prompts.append(prompt)

    # Return history excluding the added text from user messages
    visible_history = [(role, message.replace(user_input_suffix, "")) 
                       for role, message in history if role != "system"]

    print("Response::", visible_history)
    return jsonify({"history": visible_history, "suggested_prompts": suggested_prompts}), 200


@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext in ['.png', '.jpg', '.jpeg']:
        # Handle image file using Tesseract with preprocessing
        image = Image.open(file)
        image = np.array(image)
        preprocessed_image = preprocess_image(image)
        custom_config = r'--oem 1 --psm 3'
        lang = 'eng+tel+hin'
        extracted_text = pytesseract.image_to_string(preprocessed_image, config=custom_config, lang=lang)
        

    elif file_ext == '.pdf':
        outDir = tempfile.TemporaryDirectory(dir="./trash_me")
        file.save(outDir.name + "/test.pdf")
        pages = convert_from_path(outDir.name +"/test.pdf", fmt='png' )
        text_data = ''
        for page in pages:
            # print(type(page))
            # image = Image.open(page)
            image = np.array(page)
            
            # text = pytesseract.image_to_string(page)
            preprocessed_image = preprocess_image(image)
            lang = 'eng+tel+hin'
            text = pytesseract.image_to_string(preprocessed_image, lang=lang)
            text_data += text + '\n'
        # Handle PDF file
        extracted_text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text()
        extracted_text = text_data

    elif file_ext == '.docx':
        # Handle Word document
        doc = docx.Document(file)
        extracted_text = "\n".join([para.text for para in doc.paragraphs])

    else:
        return jsonify({"error": "Unsupported file type"}), 400

    return jsonify({"extracted_text": extracted_text}), 200

@app.route('/shutdown', methods=['POST'])
def shutdown():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.destroy_process_group()
    return jsonify({"status": "Process group destroyed"}), 200


@app.route('/audiochat', methods=['POST'])
def audiochat():
    if 'audio' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['audio']
    outDir = tempfile.TemporaryDirectory(dir="./trash_me")
    file.save(outDir.name + "/recording.mp3")
    audio = whisper.load_audio(outDir.name + "/recording.mp3")
    audio = whisper.pad_or_trim(audio)

    # Convert audio to text
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    result = whisper.decode(model, mel, options=whisper.DecodingOptions())
    extracted_text = result.text

    # Generate Llama response
    dialog = [{"role": "user", "content": extracted_text}]
    results = generator.chat_completion([dialog])
    llama_response_text = results[0]['generation']['content']

    # Convert Llama response to audio (using TTS system)
    response_audio_path = outDir.name + "/llama_response.mp3"
    text_to_audio(llama_response_text, response_audio_path)

    return jsonify({
        "extracted_text": extracted_text,
        "llama_response_text": llama_response_text,
        "audio_url": response_audio_path
    }), 200


@app.route('/audio', methods=['POST'])
def audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['audio']
    outDir = tempfile.TemporaryDirectory(dir="./trash_me")
    file.save(outDir.name + "/test.mp3")
    audio = whisper.load_audio(outDir.name + "/test.mp3")
    audio = whisper.pad_or_trim(audio)
    
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    
    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    
    # Extracted text from audio
    extracted_text = result.text
    print(f"Extracted Text: {extracted_text}")

    # Prepare the dialog for LLaMA
    history = [("user", extracted_text)]
    dialog = [{"role": role, "content": message} for role, message in history]

    # Generate LLaMA response
    results = generator.chat_completion(
        [dialog],
        max_gen_len=512,
        temperature=0.6,
        top_p=0.9,
    )
    llama_response = results[0]['generation']['content']

    # Add LLaMA response to history
    history.append(("RadAssistant", llama_response))

    return jsonify({
        "extracted_text": extracted_text,
        "llama_response": llama_response,
        "history": history
    }), 200


if __name__ == '__main__':
    # Initialization settings
    ckpt_dir = "/home/ubuntu/.llama/checkpoints/Meta-Llama3.1-8B-Instruct/"
    tokenizer_path = "/home/ubuntu/.llama/checkpoints/Meta-Llama3.1-8B-Instruct/tokenizer.model"
    max_seq_len = 8192  # You can adjust this as needed
    max_batch_size = 6

    # Initialize the generator on startup
    generator = setup_generator(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    print("LLaMA generator initialized.")

    # Start the Flask app
    # app.run(host='0.0.0.0', port=7860)

    http_server = WSGIServer(('0.0.0.0', 7860), app)
    http_server.serve_forever()
