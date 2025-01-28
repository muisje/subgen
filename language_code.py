from enum import Enum
import warnings
from typing import List, Optional

#Maybe handle the situation of regional notations like es-mx and just strip -mx and check as iso_639_1
#Maybe handle other things with synonyms like for when we encounter 'castellano' or 'latino' for spanish and 'und' for NONE (unknown)
#Maybe have a known unsupported list for things like fil and nob to be ignored. or have an option to log or not log a warning on unsupported languages. or raise error

class LanguageCode(Enum):
    # ISO 639-1, ISO 639-2/T, ISO 639-2/B, English Name, Native Name, optionally lit of synonyms
    AFRIKAANS = ("af", "afr", "afr", "Afrikaans", "Afrikaans")
    AMHARIC = ("am", "amh", "amh", "Amharic", "አማርኛ")
    ARABIC = ("ar", "ara", "ara", "Arabic", "العربية")
    ASSAMESE = ("as", "asm", "asm", "Assamese", "অসমীয়া")
    AZERBAIJANI = ("az", "aze", "aze", "Azerbaijani", "Azərbaycanca")
    BASHKIR = ("ba", "bak", "bak", "Bashkir", "Башҡортса")
    BELARUSIAN = ("be", "bel", "bel", "Belarusian", "Беларуская")
    BULGARIAN = ("bg", "bul", "bul", "Bulgarian", "Български")
    BENGALI = ("bn", "ben", "ben", "Bengali", "বাংলা")
    TIBETAN = ("bo", "bod", "tib", "Tibetan", "བོད་ཡིག")
    BRETON = ("br", "bre", "bre", "Breton", "Brezhoneg")
    BOSNIAN = ("bs", "bos", "bos", "Bosnian", "Bosanski")
    CATALAN = ("ca", "cat", "cat", "Catalan", "Català", ["valencian"])
    CZECH = ("cs", "ces", "cze", "Czech", "Čeština")
    WELSH = ("cy", "cym", "wel", "Welsh", "Cymraeg")
    DANISH = ("da", "dan", "dan", "Danish", "Dansk")
    GERMAN = ("de", "deu", "ger", "German", "Deutsch")
    GREEK = ("el", "ell", "gre", "Greek", "Ελληνικά")
    ENGLISH = ("en", "eng", "eng", "English", "English")
    SPANISH = ("es", "spa", "spa", "Spanish", "Español", ["castilian", "castellano", "latino"])
    ESTONIAN = ("et", "est", "est", "Estonian", "Eesti")
    BASQUE = ("eu", "eus", "baq", "Basque", "Euskara")
    PERSIAN = ("fa", "fas", "per", "Persian", "فارسی")
    FINNISH = ("fi", "fin", "fin", "Finnish", "Suomi")
    FAROESE = ("fo", "fao", "fao", "Faroese", "Føroyskt")
    FRENCH = ("fr", "fra", "fre", "French", "Français")
    GALICIAN = ("gl", "glg", "glg", "Galician", "Galego")
    GUJARATI = ("gu", "guj", "guj", "Gujarati", "ગુજરાતી")
    HAUSA = ("ha", "hau", "hau", "Hausa", "Hausa")
    HAWAIIAN = ("haw", "haw", "haw", "Hawaiian", "ʻŌlelo Hawaiʻi")
    HEBREW = ("he", "heb", "heb", "Hebrew", "עברית")
    HINDI = ("hi", "hin", "hin", "Hindi", "हिन्दी")
    CROATIAN = ("hr", "hrv", "hrv", "Croatian", "Hrvatski")
    HAITIAN_CREOLE = ("ht", "hat", "hat", "Haitian Creole", "Kreyòl Ayisyen", ["haitian"])
    HUNGARIAN = ("hu", "hun", "hun", "Hungarian", "Magyar")
    ARMENIAN = ("hy", "hye", "arm", "Armenian", "Հայերեն")
    INDONESIAN = ("id", "ind", "ind", "Indonesian", "Bahasa Indonesia")
    ICELANDIC = ("is", "isl", "ice", "Icelandic", "Íslenska")
    ITALIAN = ("it", "ita", "ita", "Italian", "Italiano")
    JAPANESE = ("ja", "jpn", "jpn", "Japanese", "日本語")
    JAVANESE = ("jw", "jav", "jav", "Javanese", "ꦧꦱꦗꦮ")
    GEORGIAN = ("ka", "kat", "geo", "Georgian", "ქართული")
    KAZAKH = ("kk", "kaz", "kaz", "Kazakh", "Қазақша")
    KHMER = ("km", "khm", "khm", "Khmer", "ភាសាខ្មែរ")
    KANNADA = ("kn", "kan", "kan", "Kannada", "ಕನ್ನಡ")
    KOREAN = ("ko", "kor", "kor", "Korean", "한국어")
    LATIN = ("la", "lat", "lat", "Latin", "Latina")
    LUXEMBOURGISH = ("lb", "ltz", "ltz", "Luxembourgish", "Lëtzebuergesch", ["letzeburgesch"])
    LINGALA = ("ln", "lin", "lin", "Lingala", "Lingála")
    LAO = ("lo", "lao", "lao", "Lao", "ພາສາລາວ")
    LITHUANIAN = ("lt", "lit", "lit", "Lithuanian", "Lietuvių")
    LATVIAN = ("lv", "lav", "lav", "Latvian", "Latviešu")
    MALAGASY = ("mg", "mlg", "mlg", "Malagasy", "Malagasy")
    MAORI = ("mi", "mri", "mao", "Maori", "Te Reo Māori")
    MACEDONIAN = ("mk", "mkd", "mac", "Macedonian", "Македонски")
    MALAYALAM = ("ml", "mal", "mal", "Malayalam", "മലയാളം")
    MONGOLIAN = ("mn", "mon", "mon", "Mongolian", "Монгол")
    MARATHI = ("mr", "mar", "mar", "Marathi", "मराठी")
    MALAY = ("ms", "msa", "may", "Malay", "Bahasa Melayu")
    MALTESE = ("mt", "mlt", "mlt", "Maltese", "Malti")
    BURMESE = ("my", "mya", "bur", "Burmese", "မြန်မာစာ",["myanmar"])
    NEPALI = ("ne", "nep", "nep", "Nepali", "नेपाली")
    DUTCH = ("nl", "nld", "dut", "Dutch", "Nederlands", ["flemish"])
    NORWEGIAN_NYNORSK = ("nn", "nno", "nno", "Norwegian Nynorsk", "Nynorsk")
    NORWEGIAN = ("no", "nor", "nor", "Norwegian", "Norsk")
    OCCITAN = ("oc", "oci", "oci", "Occitan", "Occitan")
    PUNJABI = ("pa", "pan", "pan", "Punjabi", "ਪੰਜਾਬੀ", ["panjabi"])
    POLISH = ("pl", "pol", "pol", "Polish", "Polski")
    PASHTO = ("ps", "pus", "pus", "Pashto", "پښتو", ["pushto"])
    PORTUGUESE = ("pt", "por", "por", "Portuguese", "Português")
    ROMANIAN = ("ro", "ron", "rum", "Romanian", "Română", ["moldavian", "moldovan"])
    RUSSIAN = ("ru", "rus", "rus", "Russian", "Русский")
    SANSKRIT = ("sa", "san", "san", "Sanskrit", "संस्कृतम्")
    SINDHI = ("sd", "snd", "snd", "Sindhi", "سنڌي")
    SINHALA = ("si", "sin", "sin", "Sinhala", "සිංහල", ["sinhalese"])
    SLOVAK = ("sk", "slk", "slo", "Slovak", "Slovenčina")
    SLOVENE = ("sl", "slv", "slv", "Slovene", "Slovenščina", ["slovenian"])
    SHONA = ("sn", "sna", "sna", "Shona", "ChiShona")
    SOMALI = ("so", "som", "som", "Somali", "Soomaaliga")
    ALBANIAN = ("sq", "sqi", "alb", "Albanian", "Shqip")
    SERBIAN = ("sr", "srp", "srp", "Serbian", "Српски")
    SUNDANESE = ("su", "sun", "sun", "Sundanese", "Basa Sunda")
    SWEDISH = ("sv", "swe", "swe", "Swedish", "Svenska")
    SWAHILI = ("sw", "swa", "swa", "Swahili", "Kiswahili")
    TAMIL = ("ta", "tam", "tam", "Tamil", "தமிழ்")
    TELUGU = ("te", "tel", "tel", "Telugu", "తెలుగు")
    TAJIK = ("tg", "tgk", "tgk", "Tajik", "Тоҷикӣ")
    THAI = ("th", "tha", "tha", "Thai", "ไทย")
    TURKMEN = ("tk", "tuk", "tuk", "Turkmen", "Türkmençe")
    TAGALOG = ("tl", "tgl", "tgl", "Tagalog", "Tagalog")
    TURKISH = ("tr", "tur", "tur", "Turkish", "Türkçe")
    TATAR = ("tt", "tat", "tat", "Tatar", "Татарча")
    UKRAINIAN = ("uk", "ukr", "ukr", "Ukrainian", "Українська")
    URDU = ("ur", "urd", "urd", "Urdu", "اردو")
    UZBEK = ("uz", "uzb", "uzb", "Uzbek", "Oʻzbek")
    VIETNAMESE = ("vi", "vie", "vie", "Vietnamese", "Tiếng Việt")
    YIDDISH = ("yi", "yid", "yid", "Yiddish", "ייִדיש")
    YORUBA = ("yo", "yor", "yor", "Yoruba", "Yorùbá")
    CHINESE = ("zh", "zho", "chi", "Chinese", "中文", ["mandarin"])
    CANTONESE = ("yue", "yue", "yue", "Cantonese", "粵語")
    NONE = (None, None, None, None, None, ["", "und", "Unknown"])  # For no language
    # und for Undetermined aka unknown language https://www.loc.gov/standards/iso639-2/faq.html#25
    
    # list of supported languages. raise an error if not supported 


    def __init__(self, iso_639_1, iso_639_2_t, iso_639_2_b, name_en, name_native, synonyms: Optional[List[str]] = None):
        self.iso_639_1 = iso_639_1
        self.iso_639_2_t = iso_639_2_t
        self.iso_639_2_b = iso_639_2_b
        self.name_en = name_en
        self.name_native = name_native
        self.synonyms = list(syn.lower() for syn in synonyms) if synonyms is not None else []
        

    def add_synonym(self, synonym: str) -> 'LanguageCode':
        """
        Add a new synonym for the language.
        
        Args:
            synonym: The new synonym to add
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If synonym is empty or not a string
        """
        if not isinstance(synonym, str):
            raise ValueError("Synonym must be a string")
        if not synonym.strip():
            raise ValueError("Synonym cannot be empty")
        
        synonym = synonym.lower().strip()
        if synonym in [self.name_en.lower(), self.name_native.lower()]:
            raise ValueError("Synonym cannot be the same as English or native name")
            
        if synonym not in self.synonyms:
            self.synonyms.append(synonym)
        return self

    def remove_synonym(self, synonym: str) -> 'LanguageCode':
        """
        Remove a synonym from the language.
        
        Args:
            synonym: The synonym to remove
            
        Returns:
            self for method chaining
        """
        synonym = synonym.lower().strip()
        if synonym in self.synonyms:
            self.synonyms.remove(synonym)
        return self

    def get_synonyms(self):
        """Get all synonyms for the language."""
        return self.synonyms

    @staticmethod
    def from_iso_639_1(code):
        if code == "und" or code == "":
            return LanguageCode.NONE
        for lang in LanguageCode:
            if lang.iso_639_1 == code:
                return lang
        warnings.warn(f"Invalid ISO 639-1 code: {code}", UserWarning, stacklevel=2)
        return LanguageCode.NONE

    @staticmethod
    def from_iso_639_2(code):
        if code == "und" or code == "":
            return LanguageCode.NONE
        for lang in LanguageCode:
            if lang.iso_639_2_t == code or lang.iso_639_2_b == code:
                return lang
        warnings.warn(f"Invalid ISO 639-2 code: {code}", UserWarning, stacklevel=2)
        return LanguageCode.NONE

    @staticmethod
    def from_name(name : str):
        """Convert a language name (either English or native) to LanguageCode enum."""
        
        if name.lower() in ['und', '', 'unknown']:
            return LanguageCode.NONE
        
        if any(name):
            for lang in LanguageCode:
                if lang is LanguageCode.NONE:
                    continue
                if lang.name_en.lower() == name.lower() or lang.name_native.lower() == name.lower():
                    return lang
        warnings.warn(f"Invalid language name: {name}", UserWarning, stacklevel=2)
        LanguageCode.NONE
        

    @staticmethod    
    def from_string(value: str):
        """
        Convert a string to a LanguageCode instance. Matches on ISO codes, English name, or native name.
        """
        if value is None or value.lower() in ['und', '', 'unknown']:
            return LanguageCode.NONE
        value = value.strip().lower()
        for lang in LanguageCode:
            if lang is LanguageCode.NONE:
                continue
            elif (
                value == lang.iso_639_1
                or value == lang.iso_639_2_t
                or value == lang.iso_639_2_b
                or value == lang.name_en.lower()
                or value == lang.name_native.lower()
                or value in lang.synonyms
            ):
                return lang
        warnings.warn(f"Invalid language code: {value}", UserWarning, stacklevel=2)
        return LanguageCode.NONE
    
    # is valid language
    @staticmethod
    def is_valid_language(string: str):
        for language in LanguageCode:
            if language is LanguageCode.NONE:
                continue
            else:
                if string in language:
                    return True
        return False
    
    def to_iso_639_1(self, allow_none=True):
        if self == LanguageCode.NONE:
            return None if allow_none else "und"
        return self.iso_639_1

    def to_iso_639_2_t(self, allow_none=True):
        if self == LanguageCode.NONE:
            return None if allow_none else "und"
        return self.iso_639_2_t

    def to_iso_639_2_b(self, allow_none=True):
        if self == LanguageCode.NONE:
            return None if allow_none else "und"
        return self.iso_639_2_b

    def to_name(self, in_english=True, allow_none=True):
        if self == LanguageCode.NONE:
            return None if allow_none else "Unknown"
        return self.name_en if in_english else self.name_native
    
    def __str__(self):
        return self.to_name(allow_none=False)
    
    def __repr__(self):
        return f"LanguageCode.{self.__str__().upper()}"
    
    def __format__(self, format_spec):
        """
        Formats the LanguageCode instance based on the provided format specification.

        Format Spec Options:
            - "name": The English name of the language (default).
            - "native": The native name of the language.
            - "iso1": The ISO 639-1 code.
            - "iso2t": The ISO 639-2 (terminology) code.
            - "iso2b": The ISO 639-2 (bibliographic) code.
            - "synonyms": A comma-separated list of synonyms.
            - "all": All attributes in a formatted string.
        """
        if self == LanguageCode.NONE:
            return "Unknown"
        
        if format_spec in ("", "name"):
            # Default to English name if no format_spec is provided
            return self.name_en
        elif format_spec == "native":
            return self.name_native
        elif format_spec == "iso1":
            return self.iso_639_1
        elif format_spec == "iso2t":
            return self.iso_639_2_t
        elif format_spec == "iso2b":
            return self.iso_639_2_b
        elif format_spec == "synonyms":
            return ", ".join(self.synonyms)
        elif format_spec == "all":
            return (
                f"Name (English): {self.name_en}, "
                f"Name (Native): {self.name_native}, "
                f"ISO 639-1: {self.iso_639_1},"
                f"ISO 639-2/T: {self.iso_639_2_t}, "
                f"ISO 639-2/B: {self.iso_639_2_b}"
            )
        else:
            return self.__str__()
    
    def __bool__(self):
        return True if self.iso_639_1 is not None else False
    
    def __eq__(self, other):
        """
        Compare the LanguageCode instance to another object.
        Explicitly handle comparison to None.
        """
        if other is None:
            # If compared to None, return False unless self is None
            # LanguageCode.NONE == None will return True
            # LanguageCode.None is None will return False
            # LanguageCode.ENGLISH == None will return False
            return self.iso_639_1 is None
        if isinstance(other, str):  # Allow comparison with a string
            return other in self
        if isinstance(other, LanguageCode):
            # Normal comparison for LanguageCode instances
            return self.iso_639_1 == other.iso_639_1
        # Otherwise, defer to the default equality
        return NotImplemented
    
    def __hash__(self):
        return hash(self.iso_639_1)
    
    def __contains__(self, raw_string):
        """
        Check if the given string is equal to ISO-639-1, ISO-639-2/T, or ISO-639-2/B codes, the English name, the native name, or any of the synonyms.

        Example:
            >>> "en" in LanguageCode.ENGLISH
            True
            >>> "Spanish" in LanguageCode.ENGLISH
            False
        """

        if isinstance(raw_string, str):
            string = LanguageCode.normalize(raw_string)
            return (
                string == self.iso_639_1 or
                string == self.iso_639_2_t or
                string == self.iso_639_2_b or
                string == self.name_en.lower() or
                string == self.name_native.lower() or
                string in self.synonyms
            )
        return NotImplemented
        
    @staticmethod
    def normalize(value):
        return value.strip().lower() if value else ""