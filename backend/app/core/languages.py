"""Language registry for MSMARCO-XI.

Ties together three naming schemes that do not agree with each other:

  * dataset parquet filenames  -- e.g. "hintrain.parquet" / "hinval.parquet"
  * FLORES-style target_lang   -- e.g. "hin_Deva"
  * Sarvam STT/TTS locales     -- e.g. "hi-IN"

Verified against the HF repo tree on 2026-08-15. Note Telugu has a validation
file but NO train file, so `has_train=False` for it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Language:
    code: str  # short ISO-639-1-ish key used across the app
    name: str  # English display name
    native_name: str  # endonym, for the UI
    flores: str  # FLORES lang_Script code (dataset target_lang)
    file_stem: str  # 3-letter stem in the parquet filename
    sarvam_locale: str | None  # Sarvam STT/TTS locale, None if unsupported
    script: str
    has_train: bool = True

    @property
    def val_file(self) -> str:
        return f"validation/{self.file_stem}val.parquet"

    @property
    def train_file(self) -> str | None:
        return f"train/{self.file_stem}train.parquet" if self.has_train else None


LANGUAGES: tuple[Language, ...] = (
    Language("as", "Assamese", "অসমীয়া", "asm_Beng", "asm", None, "Bengali"),
    Language("bn", "Bengali", "বাংলা", "ben_Beng", "ben", "bn-IN", "Bengali"),
    Language("gu", "Gujarati", "ગુજરાતી", "guj_Gujr", "guj", "gu-IN", "Gujarati"),
    Language("hi", "Hindi", "हिन्दी", "hin_Deva", "hin", "hi-IN", "Devanagari"),
    Language("kn", "Kannada", "ಕನ್ನಡ", "kan_Knda", "kan", "kn-IN", "Kannada"),
    Language("ml", "Malayalam", "മലയാളം", "mal_Mlym", "mal", "ml-IN", "Malayalam"),
    Language("mr", "Marathi", "मराठी", "mar_Deva", "mar", "mr-IN", "Devanagari"),
    Language("ne", "Nepali", "नेपाली", "npi_Deva", "npi", None, "Devanagari"),
    Language("or", "Odia", "ଓଡ଼ିଆ", "ory_Orya", "ory", "od-IN", "Odia"),
    Language("pa", "Punjabi", "ਪੰਜਾਬੀ", "pan_Guru", "pan", "pa-IN", "Gurmukhi"),
    Language("sa", "Sanskrit", "संस्कृतम्", "san_Deva", "san", None, "Devanagari"),
    Language("ta", "Tamil", "தமிழ்", "tam_Taml", "tam", "ta-IN", "Tamil"),
    # Telugu: validation only -- no train/teltrain.parquet exists in the repo.
    Language("te", "Telugu", "తెలుగు", "tel_Telu", "tel", "te-IN", "Telugu", has_train=False),
    Language("ur", "Urdu", "اردو", "urd_Arab", "urd", None, "Perso-Arabic"),
)

ENGLISH = Language("en", "English", "English", "eng_Latn", "eng", "en-IN", "Latin")

BY_CODE: dict[str, Language] = {lang.code: lang for lang in LANGUAGES}
BY_CODE[ENGLISH.code] = ENGLISH

BY_FLORES: dict[str, Language] = {lang.flores: lang for lang in LANGUAGES}
BY_FLORES[ENGLISH.flores] = ENGLISH

BY_FILE_STEM: dict[str, Language] = {lang.file_stem: lang for lang in LANGUAGES}


def get_language(code: str) -> Language | None:
    """Look up by short code, FLORES code, or file stem."""
    if code in BY_CODE:
        return BY_CODE[code]
    if code in BY_FLORES:
        return BY_FLORES[code]
    return BY_FILE_STEM.get(code)


def sarvam_locale_for(code: str) -> str:
    """Sarvam locale for a language, falling back to Hindi.

    Sarvam does not cover every language in the dataset (Assamese, Nepali,
    Sanskrit, Urdu). For those we still transcribe -- Sarvam's model often
    handles related scripts -- but callers should treat the result as
    lower-confidence.
    """
    lang = get_language(code)
    if lang and lang.sarvam_locale:
        return lang.sarvam_locale
    return "hi-IN"


def stt_supported(code: str) -> bool:
    lang = get_language(code)
    return bool(lang and lang.sarvam_locale)


ALL_CODES: tuple[str, ...] = tuple(lang.code for lang in LANGUAGES)
STT_SUPPORTED_CODES: tuple[str, ...] = tuple(
    lang.code for lang in LANGUAGES if lang.sarvam_locale
)
