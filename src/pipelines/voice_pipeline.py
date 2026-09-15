from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import io
import librosa
import streamlit as st


@st.cache_resource
def load_voice_encoder():
    return VoiceEncoder()


def get_voice_embedding(audio_bytes):
    try:
        if audio_bytes is None:
            return None

        encoder = load_voice_encoder()

        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True
        )

        if audio is None or len(audio) == 0:
            return None

        if len(audio) < 16000 * 0.5:
            return None

        wav = preprocess_wav(audio)

        if wav is None or len(wav) == 0:
            return None

        embedding = encoder.embed_utterance(wav)

        if embedding is None:
            return None

        embedding = np.asarray(embedding, dtype=np.float32)

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        embedding = embedding / norm

        return embedding.tolist()

    except Exception as e:
        st.error(f"Voice recognition error: {e}")
        return None


def cosine_similarity(embedding1, embedding2):
    try:
        embedding1 = np.asarray(embedding1, dtype=np.float32)
        embedding2 = np.asarray(embedding2, dtype=np.float32)

        if embedding1.shape != embedding2.shape:
            return -1.0

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return -1.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        return float(similarity)

    except Exception:
        return -1.0


def identify_speaker(new_embedding, candidates_dict, threshold=0.70):
    if new_embedding is None:
        return None, 0.0

    if not candidates_dict:
        return None, 0.0

    best_sid = None
    best_score = -1.0

    new_embedding_array = np.asarray(new_embedding, dtype=np.float32)

    for sid, stored_embedding in candidates_dict.items():
        if stored_embedding is None:
            continue

        try:
            stored_embedding_array = np.asarray(stored_embedding, dtype=np.float32)

            if new_embedding_array.shape != stored_embedding_array.shape:
                continue

            score = cosine_similarity(new_embedding_array, stored_embedding_array)

            if score > best_score:
                best_score = score
                best_sid = sid

        except Exception:
            continue

    if best_sid is not None and best_score >= threshold:
        return best_sid, best_score

    return None, best_score


def process_bulk_audio(audio_bytes, candidates_dict, threshold=0.70):
    try:
        if audio_bytes is None:
            return {}

        if not candidates_dict:
            return {}

        encoder = load_voice_encoder()

        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True
        )

        if audio is None or len(audio) == 0:
            return {}

        segments = librosa.effects.split(audio, top_db=30)

        identified_results = {}

        for start, end in segments:
            segment_length = end - start

            if segment_length < int(sr * 0.5):
                continue

            segment_audio = audio[start:end]

            wav = preprocess_wav(segment_audio)

            if wav is None or len(wav) == 0:
                continue

            embedding = encoder.embed_utterance(wav)

            if embedding is None:
                continue

            embedding = np.asarray(embedding, dtype=np.float32)

            norm = np.linalg.norm(embedding)

            if norm == 0:
                continue

            embedding = embedding / norm

            sid, score = identify_speaker(embedding, candidates_dict, threshold)

            if sid is not None:
                if sid not in identified_results or score > identified_results[sid]:
                    identified_results[sid] = score

        return identified_results

    except Exception as e:
        st.error(f"Bulk voice processing error: {e}")
        return {}