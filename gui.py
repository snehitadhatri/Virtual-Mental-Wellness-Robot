import tkinter as tk
from tkinter import ttk
import requests
import threading
import pyaudio
import wave
import io
import base64
import pygame

class EmotiCareGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EmotiCare - AI Emotional Wellness Companion")
        self.geometry("600x750")
        self.configure(bg="white")

        # Initialize pygame mixer for audio playback
        pygame.mixer.init()

        # Robot face label (placeholder for image or animation)
        self.face_label = ttk.Label(self, text=":)", font=("Arial", 100), background="white")
        self.face_label.pack(pady=20)

        # Emotion label
        self.emotion_var = tk.StringVar(value="Neutral")
        self.emotion_label = ttk.Label(self, textvariable=self.emotion_var, font=("Arial", 24), background="white")
        self.emotion_label.pack(pady=10)

        # Conversation display
        self.conversation_text = tk.Text(self, height=15, width=60, state='disabled', wrap='word')
        self.conversation_text.pack(pady=10)

        # Typing indicator label
        self.typing_var = tk.StringVar(value="")
        self.typing_label = ttk.Label(self, textvariable=self.typing_var, font=("Arial", 14), background="white", foreground="gray")
        self.typing_label.pack(pady=5)

        # Quick replies frame
        self.quick_replies_frame = ttk.Frame(self)
        self.quick_replies_frame.pack(pady=5)

        # User input entry
        self.user_input = tk.StringVar()
        self.input_entry = ttk.Entry(self, textvariable=self.user_input, width=50)
        self.input_entry.pack(side='left', padx=(20, 10), pady=10)
        self.input_entry.bind("<Return>", lambda event: self.send_message())

        # Send button
        self.send_button = ttk.Button(self, text="Send", command=self.send_message)
        self.send_button.pack(side='left', padx=(0, 20), pady=10)

        # Voice input button
        self.voice_button = ttk.Button(self, text="🎤 Speak", command=self.record_voice)
        self.voice_button.pack(pady=10)

        # Feedback buttons frame
        self.feedback_frame = ttk.Frame(self)
        self.feedback_frame.pack(pady=5)

        # Initialize conversation history
        self.conversation_history = []

        # Audio recording parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 5
        self.audio_filename = "user_input.wav"

        # Load quick replies from backend
        threading.Thread(target=self.load_quick_replies).start()

    def update_emotion(self, emotion):
        # Update the emotion label and face expression with animation
        self.emotion_var.set(emotion.capitalize())
        face_map = {
            "happy": ":D",
            "sad": ":(",
            "angry": ">:(",
            "neutral": ":|",
            "fear": "D:",
            "surprise": ":O",
            "disgust": ">:P",
            "calm": ":)",
            "anxious": ":/",
        }
        self.face_label.config(text=face_map.get(emotion.lower(), ":|"))
        # Animate background color based on emotion
        color_map = {
            "happy": "#d4f4dd",
            "sad": "#a3c4f3",
            "angry": "#f3a3a3",
            "neutral": "white",
            "fear": "#f3e6a3",
            "surprise": "#f3d1a3",
            "disgust": "#a3f3b1",
            "calm": "#d4f4dd",
            "anxious": "#f3f0a3",
        }
        bg_color = color_map.get(emotion.lower(), "white")
        self.configure(bg=bg_color)
        self.face_label.configure(background=bg_color)
        self.emotion_label.configure(background=bg_color)
        self.conversation_text.configure(background=bg_color)
        self.input_entry.configure(background=bg_color)
        self.status_label.configure(background=bg_color) if hasattr(self, 'status_label') else None

    def append_conversation(self, speaker, text):
        self.conversation_text.config(state='normal')
        self.conversation_text.insert(tk.END, f"{speaker}: {text}\n")
        self.conversation_text.config(state='disabled')
        self.conversation_text.see(tk.END)

    def set_typing(self, is_typing):
        if is_typing:
            self.typing_var.set("EmotiCare is typing...")
            self.send_button.config(state='disabled')
            self.voice_button.config(state='disabled')
        else:
            self.typing_var.set("")
            self.send_button.config(state='normal')
            self.voice_button.config(state='normal')

    def load_quick_replies(self):
        try:
            resp = requests.get("http://127.0.0.1:5000/quick_replies")
            options = resp.json().get("quick_replies", [])
            for option in options:
                btn = ttk.Button(self.quick_replies_frame, text=option["label"], command=lambda t=option["type"]: self.quick_reply_selected(t))
                btn.pack(side='left', padx=5)
        except Exception as e:
            self.append_conversation("Error", f"Failed to load quick replies: {e}")

    def quick_reply_selected(self, reply_type):
        # Use quick reply to generate wellness content and append to conversation
        self.append_conversation("You", reply_type.capitalize())
        self.conversation_history.append({"role": "user", "content": reply_type.capitalize()})
        threading.Thread(target=self.get_wellness_content, args=(reply_type,)).start()

    def get_wellness_content(self, content_type):
        self.set_typing(True)
        try:
            resp = requests.post("http://127.0.0.1:5000/generate_wellness_content", json={"type": content_type})
            content = resp.json().get("content", "")
            self.append_conversation("EmotiCare", content)
            self.conversation_history.append({"role": "assistant", "content": content})
        except Exception as e:
            self.append_conversation("Error", f"Failed to get wellness content: {e}")
        self.set_typing(False)

    def send_message(self):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        self.append_conversation("You", user_text)
        self.user_input.set("")
        self.conversation_history.append({"role": "user", "content": user_text})

        # Run API call in a separate thread to avoid blocking UI
        threading.Thread(target=self.get_bot_response, args=(user_text, self.conversation_history.copy())).start()

    def get_bot_response(self, user_text, history):
        self.set_typing(True)
        try:
            # Detect emotion
            emotion_resp = requests.post("http://127.0.0.1:5000/detect_emotion", json={"text": user_text})
            emotion = emotion_resp.json().get("emotion", "neutral")
            self.update_emotion(emotion)

            # Get bot response
            converse_resp = requests.post("http://127.0.0.1:5000/converse", json={"text": user_text, "history": history})
            bot_reply = converse_resp.json().get("response", "Sorry, I didn't understand that.")
            self.append_conversation("EmotiCare", bot_reply)
            self.conversation_history.append({"role": "assistant", "content": bot_reply})

            # Generate TTS audio - Disabled due to backend TTS removal
            # tts_resp = requests.post("http://127.0.0.1:5000/generate_wellness_content", json={"type": "quote", "context": bot_reply, "generate_audio": True})
            # audio_base64 = tts_resp.json().get("audio_base64", None)
            # if audio_base64:
            #     self.play_audio(audio_base64)
        except Exception as e:
            self.append_conversation("Error", str(e))
        self.set_typing(False)

    def record_voice(self):
        # Record audio from microphone and send to backend for speech-to-text
        threading.Thread(target=self._record_and_send).start()

    def _record_and_send(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        frames = []
        self.append_conversation("System", "Recording voice for 5 seconds...")
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save audio to file
        wf = wave.open(self.audio_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Send audio file to backend for speech-to-text
        with open(self.audio_filename, 'rb') as f:
            files = {'audio': f}
            try:
                resp = requests.post("http://127.0.0.1:5000/speech_to_text", files=files)
                text = resp.json().get("text", "")
                self.append_conversation("You (voice)", text)
                self.user_input.set(text)
                self.send_message()
            except Exception as e:
                self.append_conversation("Error", f"Voice recognition failed: {e}")

    def play_audio(self, audio_base64):
        # Play base64 encoded audio using pygame
        audio_data = base64.b64decode(audio_base64)
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)
        pygame.mixer.music.load("temp_audio.wav")
        pygame.mixer.music.play()

    def send_feedback(self, feedback_text):
        if not self.conversation_history:
            return
        last_user_input = None
        last_bot_response = None
        # Find last user and bot messages
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant" and last_bot_response is None:
                last_bot_response = msg["content"]
            elif msg["role"] == "user" and last_user_input is None:
                last_user_input = msg["content"]
            if last_user_input and last_bot_response:
                break
        if not last_user_input or not last_bot_response:
            return
        try:
            resp = requests.post("http://127.0.0.1:5000/feedback", json={
                "feedback": feedback_text,
                "user_input": last_user_input,
                "bot_response": last_bot_response
            })
            if resp.status_code == 200:
                self.append_conversation("System", f"Feedback '{feedback_text}' sent successfully.")
            else:
                self.append_conversation("System", f"Failed to send feedback: {resp.text}")
        except Exception as e:
            self.append_conversation("System", f"Error sending feedback: {e}")

    def add_feedback_buttons(self):
        for widget in self.feedback_frame.winfo_children():
            widget.destroy()
        thumbs_up = ttk.Button(self.feedback_frame, text="👍 Helpful", command=lambda: self.send_feedback("helpful"))
        thumbs_down = ttk.Button(self.feedback_frame, text="👎 Not Helpful", command=lambda: self.send_feedback("not helpful"))
        thumbs_up.pack(side='left', padx=5)
        thumbs_down.pack(side='left', padx=5)

    def append_conversation(self, speaker, text):
        self.conversation_text.config(state='normal')
        self.conversation_text.insert(tk.END, f"{speaker}: {text}\n")
        self.conversation_text.config(state='disabled')
        self.conversation_text.see(tk.END)
        # Add feedback buttons after bot response
        if speaker == "EmotiCare":
            self.add_feedback_buttons()

if __name__ == "__main__":
    app = EmotiCareGUI()
    app.mainloop()
