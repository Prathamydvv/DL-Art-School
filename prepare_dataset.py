
!pip install -U datasets
from datasets import load_dataset
dataset_full = load_dataset("SPRINGLab/IndicTTS-Hindi", split="train")
from datasets import load_dataset
import os, shutil, random
import soundfile as sf
import torchaudio
import torch

# 🔄 Optional: Limit for quick runs
dataset = dataset_full.select(range(400))  # Remove this line to use full data

# 📁 Paths
input_path = "spring_hindi_100"
output_path = "hindi_dataset"
os.makedirs(f"{input_path}/wavs", exist_ok=True)
os.makedirs(f"{output_path}/wavs", exist_ok=True)

# 🔉 Target sample rate as per your config
target_sr = 22050

# 📥 Save .wav files and build metadata
transcribed_audio_samples = []

for i, sample in enumerate(dataset):
    filename = f"utt_{i:04d}"
    text = sample["text"].strip()
    audio_array = sample["audio"]["array"]
    original_sr = sample["audio"]["sampling_rate"]

    # Resample if needed
    if original_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            torch.tensor(audio_array), orig_freq=original_sr, new_freq=target_sr
        ).numpy()
    else:
        audio_tensor = audio_array

    wav_path = f"{input_path}/wavs/{filename}.wav"
    sf.write(wav_path, audio_tensor, target_sr)

    transcribed_audio_samples.append((wav_path, f"wavs/{filename}.wav|{text}"))

# 🔀 Shuffle and split into train/val
random.shuffle(transcribed_audio_samples)
split_idx = int(0.85 * len(transcribed_audio_samples))
train_data = transcribed_audio_samples[:split_idx]
val_data = transcribed_audio_samples[split_idx:]

# 📝 Copy .wav files and write train/val metadata files
for stage, dataset in [('train', train_data), ('val', val_data)]:
    with open(f"{output_path}/{stage}.txt", "w", encoding="utf-8") as f:
        for src_path, line in dataset:
            dest_path = os.path.join(output_path, line.split('|')[0])
            shutil.copyfile(src_path, dest_path)
            f.write(line + '\n')

print("✅ Dataset preparation complete. Ready for DL-Art-School training.")
