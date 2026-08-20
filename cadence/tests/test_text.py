from cadence.text import STOPWORDS, decade_tags, normalize, title_tokens, tokenize


def test_normalize_strips_accents_emoji_and_case():
    assert normalize("Café Tacvba — Súper Éxitos! 🔥") == "cafe tacvba super exitos"


def test_normalize_expands_ampersand():
    assert "and" in normalize("Hall & Oates")


def test_normalize_empty():
    assert normalize("") == ""
    assert normalize("!!!") == ""


def test_decade_two_digit_forms_resolve_to_the_right_century():
    assert decade_tags("90s hits") == ["1990s"]
    assert decade_tags("80's") == ["1980s"]
    assert decade_tags("the 00s") == ["2000s"]
    assert decade_tags("2010s vibes") == ["2010s"]


def test_decade_from_explicit_year():
    assert decade_tags("best of 2005") == ["2000s"]


def test_decade_multiple_and_deduped():
    assert decade_tags("70s/80s and more 70s") == ["1970s", "1980s"]


def test_tokenize_emits_bigrams():
    toks = tokenize("rainy day study music")
    assert "rainy day" in toks
    assert "study music" in toks


def test_tokenize_drops_stopwords_from_unigrams():
    toks = tokenize("the best playlist of music")
    assert not ({t for t in toks if " " not in t} & STOPWORDS)


def test_title_tokens_is_capped():
    long_title = " ".join(f"word{i}" for i in range(40))
    assert len(title_tokens(long_title)) <= 6


def test_tokenize_is_deterministic():
    assert tokenize("90s Throwback Party") == tokenize("90s Throwback Party")
