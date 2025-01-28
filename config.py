from typing import List, Union, Optional
from language_code import LanguageCode
from name_subtitle import SubtitleTagType


def load_subtitle_tag_config(
    config_string,
    whisper_model: str,
    subtitle_language_naming_type: str = "ISO_639_1", 
    language: Optional[LanguageCode] = None
) -> List[Union[str, SubtitleTagType]]:
    """
    Load subtitle tags from environment variable 'SUBTITLE_TAGS'.
    
    Args:
        subtitle_language_naming_type (str): Default naming type for language tag
        language (Optional[LanguageCode]): Optional language to use for bare LANGUAGE tags
    
    Returns:
        List[Union[str, SubtitleTagType]]: Parsed list of subtitle tags
    """
    if not config_string:
        return []
    
    config_items = [item.strip() for item in config_string.split('|')]
    parsed_items = []
    
    for item in config_items:
        parsed_item = item
        if str(item) == str("LANGUAGE"):
            parsed_item = SubtitleTagType.LANGUAGE(
                language=language, 
                subtitle_language_naming_type=subtitle_language_naming_type
            )
        elif item == "WHISPER_MODEL":
            parsed_item = SubtitleTagType.WHISPER_MODEL(whisper_model=whisper_model)
        else:
            parts = item.split(':')
            setting = parts[0]
            rename = parts[1] if len(parts) > 1 else None
            if setting == "WORD_LEVEL_HIGHLIGHT" or setting == "SHOULD_STREAM_SUBTITLE":
                # print(f"Adding setting {setting} with rename={rename}")
                parsed_item = SubtitleTagType.SETTING(
                    setting_name=setting,
                    rename=rename
                )
        parsed_items.append(parsed_item)
    return parsed_items