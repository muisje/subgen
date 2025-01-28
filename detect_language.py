import faster_whisper
import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor


# https://github.com/openai/whisper/blob/25639fc/whisper/audio.py#L25-L62


def buffer_to_np_array(buffer):
    return np.frombuffer(buffer, np.int16).flatten().astype(np.float32) / 32768.0

def detect_most_probable_language(model: faster_whisper.WhisperModel, audio, seconds = 30, probability_threshold = 0.8) -> str:
    print("detect_most_probable_language")
    feature_extractor = FeatureExtractor()
    if model.feature_extractor is not None:
        feature_extractor = model.feature_extractor
    sampling_rate = feature_extractor.sampling_rate

    if not isinstance(audio, np.ndarray):
        audio = decode_audio(audio, sampling_rate=sampling_rate)

    features = feature_extractor(audio)

    total_segment_duration = 0
    all_language_probs = []
    segment_count = 0
    
    while total_segment_duration < seconds:
        segment = features[:, : feature_extractor.nb_max_frames]
        segment_duration = segment.shape[1] * feature_extractor.hop_length / sampling_rate
        if total_segment_duration + segment_duration > seconds:
            break
        total_segment_duration += segment_duration
        encoder_output = model.encode(segment)
        # results is a list of tuple[str, float] with language names and
        # probabilities.
        results = model.model.detect_language(encoder_output)[0]
        # Parse language names to strip out markers
        all_language_probs.extend([(token[2:-2], prob) for (token, prob) in results])
        features = features[:, feature_extractor.nb_max_frames:]
        segment_count += 1

    high_probabilities = [(lang, prob) for lang, prob in all_language_probs if prob > probability_threshold]
    print("High language probabilities (above {:.2f}):".format(probability_threshold))
    for lang, prob in sorted(high_probabilities, key=lambda x: x[1], reverse=True):
        print("  {}: {:.2f}".format(lang, prob))

    # Count the number of times each language exceeds the threshold
    language_counts = {}
    for lang, prob in all_language_probs:
        if prob > probability_threshold:
            if lang in language_counts:
                language_counts[lang] += 1
            else:
                language_counts[lang] = 1
    
    if not language_counts:
        return None
    
    # Get top language token and count
    most_probable_language = max(language_counts, key=language_counts.get)

    return most_probable_language
