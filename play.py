import kokoro
from onnxruntime import InferenceSession
import torch
import numpy as np
import sounddevice as sd

# Function to split tokens into manageable chunks
def split_tokens_into_chunks(tokens, max_length=510):
    """
    Split tokens into chunks with a specified max length.
    """
    chunks = []
    for i in range(0, len(tokens), max_length - 2):  # Leave room for padding
        chunk = tokens[i:i + max_length - 2]
        chunks.append([0, *chunk, 0])  # Add padding tokens at start and end
    return chunks

# Function to process chunks and generate audio
def process_chunks(tokens_chunks, ref_s, sess):
    """
    Process each token chunk and generate audio.
    """
    audio_chunks = []
    for tokens in tokens_chunks:
        audio = sess.run(None, dict(
            tokens=[tokens],  # ONNX expects a batch dimension
            style=ref_s,
            speed=np.ones(1, dtype=np.float32)
        ))[0]
        audio_chunks.append(audio)
    return audio_chunks

# Function to combine audio chunks
def assemble_audio(audio_chunks, silence_duration=0.4, samplerate=24000):
    """
    Combine audio chunks with silence in between.
    """
    silence = np.zeros(int(samplerate * silence_duration), dtype=np.float32)
    final_audio = np.concatenate([np.concatenate([chunk, silence]) for chunk in audio_chunks[:-1]] + [audio_chunks[-1]])
    return final_audio

# Main code
if __name__ == "__main__":
    # Tokens produced by phonemize() and tokenize() in kokoro.py
    text = "This is an invoice, number INV-2025-001, issued by ABC Services Pty Ltd, represented by Johnny Depp, located at 123 Business Street, Johannesburg, 2000, South Africa. The invoice is billed to XYZ Enterprises, situated at 789 Corporate Avenue, Pretoria, 0001, South Africa, and the payment is due by January 13, 2025. The invoice includes two items: first, a brochure design with a quantity of two, each priced at 100.00, with a tax of 24.00, totaling 200.00. Second, a hosting subscription with a quantity of one, priced at 1200.00, with a tax of 144.00, totaling 1200.00. The subtotal for all items is 1400.00, with a total tax of 168.00, making the grand total amount due 1568.00."

    phonemizer = kokoro.phonemize(text, lang='a')
    tokens = kokoro.tokenize(phonemizer)
    print(f"Tokens: {tokens}")

    # Load style vector
    style_vectors = torch.load('voices/af.pt')
    if len(tokens) < len(style_vectors):
        ref_s = style_vectors[len(tokens)].numpy()
    else:
        # Use the maximum available style vector
        ref_s = style_vectors[-1].numpy()
        print(f"Warning: Using the default maximum style vector due to token length.")

    # Chunk tokens into manageable sizes
    tokens_chunks = split_tokens_into_chunks(tokens, max_length=510)

    # Load ONNX model
    sess = InferenceSession('kokoro-v0_19.onnx')

    # Process each chunk and generate audio
    audio_chunks = process_chunks(tokens_chunks, ref_s, sess)

    # Combine audio chunks into a single audio array
    final_audio = assemble_audio(audio_chunks, silence_duration=0.4)

    # Play the final audio
    sd.play(final_audio, samplerate=24000)
    sd.wait()  # Wait until playback is finished
