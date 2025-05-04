# Virtual Robot - EmotiCare AI Assistant

## Overview
Virtual Robot is an empathetic and supportive AI assistant specialized in emotional wellness. It provides compassionate, understanding, and helpful responses tailored to the user's emotional state and intent. The assistant uses a combination of OpenAI's GPT-3.5-turbo API and a local fallback model (DistilGPT2) to generate responses and wellness content such as guided meditations, motivational stories, inspirational quotes, and coping strategies.

## Features
- Emotion detection using a fine-tuned DistilRoBERTa model.
- Intent classification using logistic regression on TF-IDF vectorized text.
- Conversational AI powered by OpenAI GPT-3.5-turbo with local fallback.
- Wellness content generation (meditation, story, quote, coping strategy).
- Facial emotion detection using OpenCV's Haar cascades.
- Session logging for conversation history.
- Graceful handling of OpenAI API quota limits and invalid API keys.

## Requirements
- Python 3.8 or higher
- CUDA-enabled GPU (optional, for faster local model inference)
- The following Python packages (see `requirements.txt`):
  - Flask
  - openai
  - transformers
  - torch
  - python-dotenv
  - nltk
  - scikit-learn
  - opencv-python
  - numpy

## Installation

1. Clone the repository or download the project files.

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r virtual_robot/requirements.txt
   ```

4. Download NLTK stopwords (if not already downloaded):

   ```python
   import nltk
   nltk.download('stopwords')
   ```

5. Set up your OpenAI API key:

   - Create a `.env` file in the `virtual_robot` directory.
   - Add the following line with your API key:

     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage

1. Navigate to the `virtual_robot` directory:

   ```bash
   cd virtual_robot
   ```

2. Run the Flask application:

   ```bash
   python app.py
   ```

3. Open your browser and go to:

   ```
   http://localhost:5000/
   ```

4. Use the web interface to interact with the EmotiCare AI assistant.

## API Endpoints

- `GET /`  
  Serves the main index.html page.

- `POST /detect_emotion`  
  Detects emotion and intent from provided text.  
  Request JSON: `{ "text": "your input text" }`  
  Response JSON: `{ "emotion": "detected_emotion", "intent": "detected_intent" }`

- `POST /converse`  
  Converses with the AI assistant.  
  Request JSON:  
  ```json
  {
    "text": "user input",
    "history": [ { "role": "user/assistant", "content": "previous messages" }, ... ]
  }
  ```  
  Response JSON: `{ "response": "AI reply" }`

- `POST /generate_wellness_content`  
  Generates wellness content based on type and context.  
  Request JSON: `{ "type": "meditation/story/quote/coping_strategy", "context": "optional context" }`  
  Response JSON: `{ "content": "generated content" }`

- `POST /facial_emotion_detection`  
  Detects facial emotion from an uploaded image.  
  Form-data: `image` file  
  Response JSON: `{ "emotion": "detected_emotion" }`

## Unique Requirements and Notes

- The application uses OpenAI's GPT-3.5-turbo model for generating conversational responses and wellness content. Ensure your API key has sufficient quota.
- If the OpenAI API quota is exceeded or the API key is invalid, the app falls back to a local DistilGPT2 model for response generation.
- CUDA-enabled GPU is optional but recommended for faster local model inference.
- The facial emotion detection uses OpenCV's Haar cascades and requires the `haarcascade_frontalface_default.xml` file, which is included with OpenCV.
- Session logs are saved in `session_logs.json` in the project directory.
- The app uses NLTK stopwords for preprocessing text inputs.

## Troubleshooting

- If you encounter errors related to missing NLTK data, run:

  ```python
  import nltk
  nltk.download('stopwords')
  ```

- Ensure your OpenAI API key is correctly set in the `.env` file.
- If you hit API quota limits, the app will automatically use the local model fallback.
- For any issues with the local model, verify that PyTorch and Transformers are installed correctly and that your environment supports CUDA if using GPU acceleration.

## License

This project is provided as-is under the MIT License.

## Contact

For questions or support, please contact the project maintainer.
