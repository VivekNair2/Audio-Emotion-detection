from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa
from pydub import AudioSegment, silence
import os

# Load the model and feature extractor
model_name = "Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

def predict_emotion(audio_path):
    audio, sample_rate = librosa.load(audio_path, sr=16000)
    from gradio_client import Client, file

    client = Client("latterworks/speech-emotion-recognition")
    result = client.predict(
	param_0=file(audio_path),
	api_name="/predict"
)
    max_confidence = max(result['confidences'], key=lambda x: x['confidence'])
    max_label = max_confidence['label']
    max_score = max_confidence['confidence']
    return max_label

def convert_to_wav(input_path):
    """Convert audio to WAV if necessary."""
    if not input_path.lower().endswith('.wav'):
        output_path = input_path.rsplit('.', 1)[0] + '.wav'
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    return input_path

def split_audio_on_silence(audio_path, min_silence_len=500, silence_thresh=-40, keep_silence=500):
    """Split audio file into chunks based on silence."""
    audio = AudioSegment.from_wav(audio_path)
    chunks = silence.split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence)
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    return chunk_paths

def main(audio_path):
    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    wav_path = convert_to_wav(audio_path)
    chunk_paths = split_audio_on_silence(wav_path)
    emotions = []
    for chunk_path in chunk_paths:
        emotion = predict_emotion(chunk_path)
        emotions.append(emotion)
        print(f"Chunk {chunk_path}: {emotion}")
    
    # Save results to a text file
    result_path = os.path.join("uploads", "emotions.txt")
    with open(result_path, "w") as result_file:
        for i, emotion in enumerate(emotions):
            result_file.write(f"Chunk {i}: {emotion}\n")
    
    print(f"Emotions saved to {result_path}")

if __name__ == "__main__":
    audio_path = "REc98f14af2b375db6b9b53e31c5d6ae87.mp3"  # Replace with your audio file path
    main(audio_path)
