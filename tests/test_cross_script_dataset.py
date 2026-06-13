from scripts.extract_cross_script_dataset import collect_sentences, extract_sentence


def test_extract_sentence_from_translation_dict():
    row = {"translation": {"en": "hello", "hi": "नमस्ते दुनिया"}}
    assert extract_sentence(row, language="hindi") == "नमस्ते दुनिया"


def test_collect_sentences_deduplicates_and_limits():
    rows = [
        {"translation": {"hi": "पहला वाक्य"}},
        {"translation": {"hi": "पहला वाक्य"}},
        {"translation": {"hi": "दूसरा वाक्य"}},
    ]
    assert collect_sentences(rows, language="hindi", limit=2) == ["पहला वाक्य", "दूसरा वाक्य"]


def test_extract_sentence_supports_explicit_column():
    assert extract_sentence({"target": "यह परीक्षण है"}, language="hindi", text_column="target") == "यह परीक्षण है"
