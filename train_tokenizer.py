import re
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# ✅ Load dataset from HuggingFace
dataset = load_dataset("SPRINGLab/IndicTTS-Hindi", split="train")

# ✅ Extract and clean Hindi text
def clean_hindi_text(text):
    text = text.lower()
    text = re.sub(r"[^ऀ-ॿ\s.,!?]", "", text)  # Keep only Devanagari + basic punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

text_corpus = [clean_hindi_text(sample["text"]) for sample in dataset if sample["text"].strip()]

# ✅ Ensure corpus is valid
assert isinstance(text_corpus, list) and all(isinstance(t, str) for t in text_corpus)

# ✅ Tokenizer configuration
trainer = BpeTrainer(
    vocab_size=255,
    special_tokens=["[STOP]", "[UNK]", "[SPACE]"]
)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# ✅ Training loop (batched for performance)
def batch_iterator(batch_size=1000):
    for i in range(0, len(text_corpus), batch_size):
        yield text_corpus[i:i + batch_size]

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# ✅ Save tokenizer
tokenizer.save("custom_hindi_tokenizer.json")
print("✅ Tokenizer saved to /content/custom_hindi_tokenizer.json")