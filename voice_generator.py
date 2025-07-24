import whisper
import speech_recognition as sr
import openai
import os
import subprocess

# Folder containing ffmpeg.exe (directory, NOT the exe itself)
ffmpeg_dir = r"C:\ffmpeg\ffmpeg-2025-07-21-git-8cdb47e47a-full_build\bin"

# Add ffmpeg directory to PATH environment variable
os.environ["PATH"] += os.pathsep + ffmpeg_dir

subprocess.run([ffmpeg_dir, "-version"])

openai.api_key = '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'  # Replace with your actual API key


model = whisper.load_model("base")

def record_audio(filename="voice_input.wav", duration=5):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now...")
        audio = r.listen(source, phrase_time_limit=duration)
        print("🔄 Saving audio...")
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())
    return filename


def transcribe_audio(filename):
    print("🧠 Transcribing with Whisper...")
    result = model.transcribe(filename)
    print(f"📝 You said: {result['text']}")
    return result['text']

def get_movie_recommendations(prompt_text):
    print("🎬 Asking GPT for recommendations...")
    system_prompt = "You're a movie expert. Recommend a few movies based on the user's interest."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ],
        max_tokens=200
    )
    
    recommendations = response['choices'][0]['message']['content']
    return recommendations


if __name__ == "__main__":
    audio_file = record_audio()
    text_prompt = transcribe_audio(audio_file)
    movie_recs = get_movie_recommendations(text_prompt)

    print("\n🎥 Movie Recommendations:")
    print(movie_recs)
