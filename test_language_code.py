import unittest
from language_code import LanguageCode


# from https://github.com/openai/whisper/blob/517a43ecd132a2089d85f4ebc044728a71d49f6e/whisper/tokenizer.py#L10
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

class TestLanguageCode(unittest.TestCase):
    def test_is_valid_language(self):
        self.assertTrue(LanguageCode.is_valid_language("en"))
        self.assertFalse(LanguageCode.is_valid_language("invalid"))
        
    def test_contains(self):
        self.assertTrue("en" in LanguageCode.ENGLISH, "'en' should be in ENGLISH")
        self.assertFalse("es" in LanguageCode.ENGLISH, "'es' should not be in ENGLISH")
    
    def test_supported_whisper_languages(self):
        for key, value in LANGUAGES.items():
            self.assertEqual(key, LanguageCode.from_iso_639_1(key).to_iso_639_1())
            self.assertEqual(key, LanguageCode.from_string(value).to_iso_639_1())
            

    def test_add_synonyms(self):
        language = LanguageCode.ENGLISH.add_synonym("engels").add_synonym("anglaise").add_synonym("ingles")
        self.assertTrue(all(item in language.get_synonyms() for item in ['engels', 'anglaise', 'ingles']))


    def test_add_synonym(self):
        language = LanguageCode.ENGLISH
        language.add_synonym("united kingdom")
        self.assertIn("united kingdom", language.get_synonyms())

    def test_add_duplicate_synonym(self):
        language = LanguageCode.ENGLISH
        with self.assertRaises(ValueError) as context:
            language.add_synonym("english")
        self.assertEqual(str(context.exception), "Synonym cannot be the same as English or native name")

    def test_remove_synonym(self):
        language = LanguageCode.ENGLISH
        language.remove_synonym("ingles")
        self.assertNotIn("ingles", language.get_synonyms())

    def test_invalid_synonym_type(self):
        language = LanguageCode.ENGLISH
        with self.assertRaises(ValueError) as context:
            language.add_synonym(123)  # Non-string input
        self.assertEqual(str(context.exception), "Synonym must be a string")

    def test_empty_synonym(self):
        language = LanguageCode.ENGLISH
        with self.assertRaises(ValueError) as context:
            language.add_synonym("")  # Empty string
        self.assertEqual(str(context.exception), "Synonym cannot be empty")
    
    def test_equal_language_codes(self):
        lang1 = LanguageCode.from_string("en")
        lang2 = LanguageCode.from_string("EN")
        self.assertEqual(lang1, lang2)

    def test_different_language_codes(self):
        lang1 = LanguageCode.from_string("en")
        lang2 = LanguageCode.from_string("es")
        self.assertNotEqual(lang1, lang2)

    def test_comparison_with_none(self):
        lang1 = LanguageCode.from_string("en")
        lang_none = LanguageCode.from_string(None)
        self.assertNotEqual(lang1, None)
        self.assertEqual(lang_none, None)

    def test_comparison_with_string(self):
        lang1 = LanguageCode.from_string("en")
        lang2 = LanguageCode.from_string("es")
        self.assertEqual(lang1, "en")
        self.assertNotEqual(lang2, "en")

    def test_comparison_with_unsupported_type(self):
        lang1 = LanguageCode.from_string("en")
        self.assertNotEqual(lang1, 42)
        
    def test_bool(self):
        self.assertTrue(bool(LanguageCode.ENGLISH), "ENGLISH should be truthy")
        self.assertTrue(bool(LanguageCode.SPANISH), "SPANISH should be truthy")
        self.assertFalse(bool(LanguageCode.NONE), "NONE should be falsy")
        
    def test_if_value(self):
        # Test with a LanguageCode that has a valid `iso_639_1` value
        self.assertTrue(LanguageCode.ENGLISH, "LanguageCode.ENGLISH should evaluate to True in a boolean context")

    def test_if_not_value(self):
        # Test with a LanguageCode that represents NONE (with no `iso_639_1` value)
        self.assertFalse(LanguageCode.NONE, "LanguageCode.NONE should evaluate to False in a boolean context")

    def test_eq_with_none(self):
        self.assertFalse(LanguageCode.ENGLISH == None, "ENGLISH should not equal None")
        self.assertTrue(LanguageCode.NONE == None, "NONE should equal None")
        
    def test_english_is_not_none(self):
        # 'is not' checks identity, so this should pass
        self.assertIsNot(LanguageCode.ENGLISH, None, "LanguageCode.ENGLISH should not be None")

    def test_english_eq_none(self):
        # '==' uses __eq__, which should return False when comparing ENGLISH to None
        self.assertFalse(LanguageCode.ENGLISH == None, "LanguageCode.ENGLISH should not equal None when using '=='")

    def test_is_none(self):
        # This should fail because `LanguageCode.NONE is None` is not true
        self.assertIsNot(LanguageCode.NONE, None, "LanguageCode.NONE should not be None when using 'is'")

    def test_eq_none(self):
        # This should pass because `LanguageCode.NONE == None` is defined in __eq__
        self.assertTrue(LanguageCode.NONE == None, "LanguageCode.NONE should equal None when using '=='")
    def test_eq_with_string(self):
        self.assertTrue(LanguageCode.ENGLISH == "en", "ENGLISH should equal 'en'")
        self.assertFalse(LanguageCode.SPANISH == "en", "SPANISH should not equal 'en'")

    def test_eq_with_another_enum(self):
        self.assertTrue(LanguageCode.ENGLISH == LanguageCode.ENGLISH, "Same enums should be equal")
        self.assertFalse(LanguageCode.ENGLISH == LanguageCode.SPANISH, "Different enums should not be equal")

    def test_contains(self):
        self.assertTrue("en" in LanguageCode.ENGLISH, "'en' should be in ENGLISH")
        self.assertFalse("es" in LanguageCode.ENGLISH, "'es' should not be in ENGLISH")
        self.assertFalse("en" in LanguageCode.NONE, "'en' should not be in NONE")
    
    def test_enum_iteration(self):
        codes = [code for code in LanguageCode]
        self.assertIn(LanguageCode.ENGLISH, codes)
        self.assertIn(LanguageCode.SPANISH, codes)
        
    # def test_invalid_language_code(self):
    #     with self.assertRaises(ValueError):
    #         LanguageCode.from_string("invalid_code")


    # Test case insensitivity
    def test_case_insensitivity(self):
        lang1 = LanguageCode.from_string("EN")
        lang2 = LanguageCode.from_string("en")
        self.assertEqual(lang1, lang2)

    # Test repr method
    def test_repr(self):
        lang = LanguageCode.ENGLISH
        self.assertEqual(repr(lang), "LanguageCode.ENGLISH")

    # Test str method
    def test_str(self):
        lang = LanguageCode.ENGLISH
        self.assertEqual(str(lang), "English")

    # Test equality with other object types
    def test_eq_with_other_objects(self):
        lang = LanguageCode.ENGLISH
        self.assertNotEqual(lang, object())
        self.assertNotEqual(lang, 42)
        self.assertNotEqual(lang, [])
        # Equality with another LanguageCode instance that is the same
        self.assertEqual(lang, LanguageCode.ENGLISH, "ENGLISH should be equal to another ENGLISH")
        
        # Equality with a string that matches the iso_639_1 value
        self.assertEqual(lang, "en", "ENGLISH should be equal to the string 'en'")
        
        # Test with another LanguageCode that is the same in terms of iso_639_1 value
        lang2 = LanguageCode.from_string("EN")
        self.assertEqual(lang, lang2, "ENGLISH should be equal to a LanguageCode with the same iso_639_1 value")
        
        # Test comparing with a lowercase string
        self.assertEqual(lang, "en", "ENGLISH should be equal to the lowercase string 'en'")
        
        # Test comparing with the same value (but different case)
        lang_upper = LanguageCode.from_string("EN")
        self.assertEqual(lang, lang_upper, "ENGLISH should be equal to ENGLISH with upper case")
        
        # Test comparing with a string that represents a valid iso_639_1 value
        self.assertNotEqual(lang, "es", "ENGLISH should not be equal to the string 'es'")
        
        # Test comparison with different LanguageCode instance
        lang2 = LanguageCode.from_string("es")
        self.assertNotEqual(lang, lang2, "ENGLISH should not be equal to SPANISH")
        
    def test_hash_and_sets(self):
        # Test that LanguageCode objects are hashable and behave correctly in sets
        lang_set = {LanguageCode.ENGLISH, LanguageCode.SPANISH}
        self.assertIn(LanguageCode.ENGLISH, lang_set, "ENGLISH should be in the set")
        self.assertNotIn(LanguageCode.NONE, lang_set, "NONE should not be in the set")
        
        # Test with dictionaries using LanguageCode as keys
        lang_dict = {LanguageCode.ENGLISH: "English", LanguageCode.SPANISH: "Spanish"}
        self.assertEqual(lang_dict[LanguageCode.ENGLISH], "English", "The value for ENGLISH should be 'English'")
        
    # def test_from_string_with_invalid_input(self):
    # # Invalid language codes should raise an exception or return NONE, depending on your design
    #     with self.assertRaises(ValueError):  # or whatever exception your code raises
    #         LanguageCode.from_string("invalid_code")

    # def test_invalid_iso_639_1(self):
    #     # Test with an invalid language code
    #     lang_invalid = LanguageCode.from_string("xyz")
    #     self.assertEqual(lang_invalid, LanguageCode.NONE, "Invalid language code should return NONE")

    #     # Test with an empty string
    #     lang_empty = LanguageCode.from_string("")
    #     self.assertEqual(lang_empty, LanguageCode.NONE, "Empty string should return NONE")
        
    #     # Test with a None input
    #     lang_none = LanguageCode.from_string(None)
    #     self.assertEqual(lang_none, LanguageCode.NONE, "None should return NONE")

if __name__ == "__main__":
    unittest.main()