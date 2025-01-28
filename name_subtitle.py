from language_code import LanguageCode
from enum import Enum
import os
from typing import Union, List

def define_subtitle_language_naming(language: LanguageCode, type):
    """
    Determines the naming format for a subtitle language based on the given type.

    Args:
        language (LanguageCode): The language code object containing methods to get different formats of the language name.
        type (str): The type of naming format desired, such as 'ISO_639_1', 'ISO_639_2_T', 'ISO_639_2_B', 'NAME', or 'NATIVE'.

    Returns:
        str: The language name in the specified format. If an invalid type is provided, it defaults to the language's name.
    """
    switch_dict = {
        "ISO_639_1": language.to_iso_639_1,
        "ISO_639_2_T": language.to_iso_639_2_t,
        "ISO_639_2_B": language.to_iso_639_2_b,
        "NAME": language.to_name,
        "NATIVE": lambda : language.to_name(in_english=False)
    }
    return switch_dict.get(type, language.to_name)()

class FileWriteBehavior(Enum):
    OVERWRITE = "overwrite"
    SKIP = "skip"
    UNIQUE = "unique"
    
class SubtitleTagType:
    class BaseTag:
        def __eq__(self, other):
            if isinstance(other, type):
                return type(self) is other
            if isinstance(other, type(self)):  # Check if both are instances of the same class
                return self._stored_arguments == other._stored_arguments
            if other is None:
                return False
            return False
        
        def __str__(self):
            string = self.resolve()
            return string if string != "" else "No string"
        
        def __repr__(self):
            return f"{self.__class__.__name__}({repr(self._stored_arguments)})"
        

        def resolve(self) -> str:
            if isinstance(self, SubtitleTagType.LANGUAGE) and self._stored_arguments:
                return define_subtitle_language_naming(self._stored_arguments['language'], self._stored_arguments['subtitle_language_naming_type'])
            elif isinstance(self, SubtitleTagType.WHISPER_MODEL):
                return self._stored_arguments['whisper_model']
            elif isinstance(self, SubtitleTagType.SETTING):
                if 'value' in self._stored_arguments and self._stored_arguments['value']:
                    if 'rename' in self._stored_arguments and self._stored_arguments['rename']:
                        return self._stored_arguments['rename']
                    return f"{self._stored_arguments['setting_name']}"
                else:
                    return ""
            else:
                raise ValueError(f"Unknown tag type: {self.__class__.__name__} with args={self._stored_arguments}")

    class SETTING(BaseTag):
        def __init__(self, **kwargs):
            required = {'setting_name'}
            missing = required - set(kwargs.keys())
            if missing:
                raise ValueError(f"Missing required arguments for SETTING: {missing}")
            if 'value' in kwargs and not isinstance(kwargs['value'], bool):
                raise TypeError("'value' must be of type bool for SETTING.")
            self._stored_arguments = kwargs

    class LANGUAGE(BaseTag):
        def __init__(self, **kwargs):
            required = {'language', 'subtitle_language_naming_type'}
            missing = required - set(kwargs.keys())
            if missing:
                raise ValueError(f"Missing required arguments for LANGUAGE: {missing}")
            self._stored_arguments = kwargs

    class WHISPER_MODEL(BaseTag):
        def __init__(self, **kwargs):
            required = {'whisper_model'}
            missing = required - set(kwargs.keys())
            if missing:
                raise ValueError(f"Missing required arguments for WHISPER_MODEL: {missing}")
            self._stored_arguments = kwargs

def generate_tag(tag) -> str:
    """
    Generate the tag value. If the tag is a function (Enum), it will call the function, 
    otherwise it returns the string as is.
    """
    if isinstance(tag, SubtitleTagType):
        return tag.resolve()
    else:
        # If it's a string, return it directly
        return tag

def name_subtitle(file_path: str, file_write_behavior: FileWriteBehavior = FileWriteBehavior.UNIQUE, tags :List[Union[str, SubtitleTagType]] = ["subgen",  "default"] , subtitle_tag_delimiter: str = ".") -> str:
    """
    Name the the subtitle file to be written, based on the source file and the language of the subtitle.
    
    Args:
        file_path: The path to the source file.
        language: The language of the subtitle.
    
    Returns:
        The name of the subtitle file to be written.
    """
    
    tag_string = ""
    for tag in tags:
        if tag is not None:
            try:
                generated_tag = generate_tag(tag)
                if generated_tag and generated_tag != "":
                    tag_string += subtitle_tag_delimiter + str(generated_tag)
            except Exception as e:
                    print(f"Error while generating tag {tag.__class__.__name__}: {e}")

    file_name = f"{os.path.splitext(file_path)[0]}{tag_string}.srt"
    
    # Check if the file already exists and naming the subtitle file based on the chosen behavior
    if os.path.exists(file_name):
        if file_write_behavior == FileWriteBehavior.OVERWRITE:
            return file_name
        elif file_write_behavior == FileWriteBehavior.SKIP:
                print(f"File {file_name} already exists. Skipping.")
                return None
        elif file_write_behavior == FileWriteBehavior.UNIQUE:
            base, ext = os.path.splitext(file_name)
            counter = 1
            
            # Increment the counter if the file already exists with counter
            possible_base, tag = os.path.splitext(base)
            if tag.isdigit():
                base = possible_base
                counter = int(tag) + 1
                
            while os.path.exists(file_name):
                file_name = f"{base}.{counter}{ext}"
                counter += 1
        else:
            raise ValueError(f"Unknown behavior: {file_write_behavior}")

    
    return file_name