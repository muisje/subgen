import ffmpeg
import logging
import os
from typing import Dict, Any, List
from language_code import LanguageCode
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_audio_languages_mapped(file_path: str) -> Dict[int, LanguageCode]:
    """
    Retrieves the language metadata from audio streams in a media file and returns them in a dictionary.

    Assumptions:
      - Audio streams are reindexed sequentially based on their order in the file.
      - For example, if a file contains audio streams with ffmpeg indices 1 and 2,
        they are reindexed as 0 and 1 respectively.
      - If a language tag is not available, a LanguageCode.NONE is returned.

    :param file_path: Path to the media file.
    :return: A dictionary mapping sequential audio stream numbers to language codes.
             Example: {0: LanguageCode.PORTUGUESE, 1: LanguageCode.DUTCH}
    """
    if not os.path.exists(file_path):
        logger.warning("File %s does not exist.", file_path)
        return {}

    try:
        probe = ffmpeg.probe(file_path)
    except ffmpeg.Error as e:
        logger.warning("Error probing file %s: %s", file_path, e.stderr.decode('utf-8', 'ignore'))
        return {}

    # Filter for audio streams and reindex them sequentially.
    audio_streams: List[Dict[str, Any]] = [
        stream for stream in probe.get('streams', [])
        if stream.get('codec_type') == 'audio'
    ]
    if not audio_streams:
        logger.warning("No audio streams found in file %s.", file_path)
        return {}

    languages: Dict[int, LanguageCode] = {}
    for idx, stream in enumerate(audio_streams):
        tags = stream.get('tags', {})
        languages[idx] = LanguageCode.from_iso_639_2(tags.get('language', 'und'))
    return languages


def verify_audio_languages_mapped(file_path: str, expected_languages: Dict[int, LanguageCode]) -> bool:
    """
    Verifies that the audio streams in the file have the expected language tags.

    Assumptions:
      - Audio streams are reindexed sequentially (0, 1, 2, ...) based on their order in the file.
      - The expected_languages dictionary should use this sequential indexing.
      - For instance, if the expected_languages is {0: LanguageCode.PORTUGUESE, 1: LanguageCode.DUTCH}, it assumes that
        the first audio stream is LanguageCode.PORTUGUESE and the second is LanguageCode.DUTCH.

    :param file_path: Path to the media file.
    :param expected_languages: Dictionary mapping sequential audio stream numbers to expected language codes.
                               Example: {0: LanguageCode.PORTUGUESE, 1: LanguageCode.DUTCH}
    :return: True if all specified streams match the expected language codes; False otherwise.
    """
    actual_languages = get_audio_languages_mapped(file_path)
    for index, expected_language in expected_languages.items():
        actual_language = actual_languages.get(index, LanguageCode.NONE)
        if not actual_language == expected_language:
            logger.warning(f"Mismatch at index {index}: expected {expected_language}, got {actual_language}")
            return False
    return True


def set_audio_languages_mapped(input_path: str,
                        output_path: str,
                        languages: Dict[int, LanguageCode]) -> None:
    """
    Sets the language metadata on audio streams of the input media file and writes the result to a new file.

    Assumptions:
      - Audio streams's indexes passed sequentially (0, 1, 2, ...), but the actual index in the ouput/input will be different (1, 2, 3, ...)
      - For example, if languages is {0: LanguageCode.PORTUGUESE, 1: LanguageCode.DUTCH}, the first audio stream will be set to LanguageCode.PORTUGUESE
        and the second to LanguageCode.DUTCH, regardless of the original stream indices in the input file.

    The function uses ffmpeg options to set the language metadata in the following format:
      -metadata:s:a:<index> language=<value>

    :param input_path: Path to the input media file.
    :param output_path: Path to the output media file.
    :param languages: Dictionary mapping sequential audio stream numbers to language codes.
                      Example: {0: LanguageCode.PORTUGUESE, 1: LanguageCode.DUTCH}
    :return: None
    """
    if not os.path.exists(input_path):
        logger.warning("Input file %s does not exist.", input_path)
        return

    # Build ffmpeg options: copy codecs and map all streams.
    output_options: Dict[str, Any] = {'c': 'copy', 'map': '0'}

    # Add metadata options for each audio stream based on the sequential index.
    for stream_index, language in languages.items():
        option_key = f"metadata:s:a:{stream_index}"
        output_options[option_key] = f"language={language.to_iso_639_2_t()}"

    try:
        ffmpeg.input(input_path) \
            .output(output_path, **output_options) \
            .overwrite_output() \
            .run(quiet=True)
            
    except ffmpeg.Error as e:
        logger.warning(f"Error processing file {input_path}: {e}")
        raise

def over_write_audio_language_metadata(input_file: str, languages: Dict[int, LanguageCode]) -> bool:
    """
    Overwrites the language metadata of the audio streams in the given input file.

    This function performs the following steps:
      1. Creates a temporary output file name based on the input file.
      2. Calls `set_audio_languages_mapped` to update the audio language metadata in the temporary file.
      3. Verifies that the language metadata in the temporary file matches the expected languages using
         `verify_audio_languages_mapped`.
      4. If verification succeeds, replaces the original input file with the temporary file.
      5. Cleans up the temporary file.
    
    Assumptions:
      - Audio streams are reindexed sequentially (0, 1, 2, ...) based on their order in the file.
      - The `languages` dictionary uses this sequential indexing to specify the language for each audio stream.
      - If verification fails, the original file remains unchanged.

    :param input_file: Path to the media file whose audio language metadata is to be updated.
    :param languages: Dictionary mapping sequential audio stream numbers to the desired language codes.
                      Example: {0: LanguageCode.PORTUGUESE, 1: LanguageCode.DUTCH}
    :return: True if the metadata was successfully updated and verified; False otherwise.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, os.path.basename(input_file))
        
        set_audio_languages_mapped(input_file, output_file, languages)
        
        success = verify_audio_languages_mapped(output_file, languages)
        
        if success:
            os.replace(output_file, input_file)
    
    return success 
    
