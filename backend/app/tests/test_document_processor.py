import pytest

from app.services.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=200, chunk_overlap=20)


def test_clean_text_preserves_paragraph_breaks(processor):
    raw = "First paragraph line one.\nFirst paragraph line two.\n\nSecond paragraph."
    cleaned = processor._clean_text(raw)

    assert "\n\n" in cleaned
    paragraphs = cleaned.split("\n\n")
    assert paragraphs == [
        "First paragraph line one. First paragraph line two.",
        "Second paragraph.",
    ]


def test_clean_text_collapses_intra_paragraph_whitespace(processor):
    raw = "Word1   \t  Word2\nWord3"
    cleaned = processor._clean_text(raw)

    assert cleaned == "Word1 Word2 Word3"


def test_clean_text_collapses_excess_blank_lines_to_one_paragraph_break(processor):
    raw = "Paragraph one.\n\n\n\nParagraph two."
    cleaned = processor._clean_text(raw)

    assert cleaned == "Paragraph one.\n\nParagraph two."


def test_create_chunks_prefers_paragraph_boundary(processor):
    para_a = "A" * 150
    para_b = "B" * 150
    text = f"{para_a}\n\n{para_b}"

    chunks = processor._create_chunks(text, "doc.txt", [])

    assert len(chunks) >= 2
    assert chunks[0].text.strip() == para_a


def test_process_file_empty_txt_produces_no_chunks(tmp_path, processor):
    filepath = tmp_path / "empty.txt"
    filepath.write_text("   \n\n  ", encoding="utf-8")

    chunks = processor.process_file(str(filepath))

    assert chunks == []


def test_process_file_single_sentence(tmp_path, processor):
    filepath = tmp_path / "short.txt"
    filepath.write_text("Just one short sentence.", encoding="utf-8")

    chunks = processor.process_file(str(filepath))

    assert len(chunks) == 1
    assert chunks[0].text == "Just one short sentence."
    assert chunks[0].metadata["source"] == "short.txt"


def test_process_file_unsupported_extension_raises(tmp_path, processor):
    filepath = tmp_path / "file.docx.bak"
    filepath.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError):
        processor.process_file(str(filepath))


def test_process_file_missing_raises(processor):
    with pytest.raises(FileNotFoundError):
        processor.process_file("does-not-exist.txt")


def test_process_file_source_name_override(tmp_path, processor):
    filepath = tmp_path / "20260101_000000_report.txt"
    filepath.write_text("Some content here.", encoding="utf-8")

    chunks = processor.process_file(str(filepath), source_name="report.txt", session_id="abc")

    assert all(c.metadata["source"] == "report.txt" for c in chunks)
    assert all(c.metadata["session_id"] == "abc" for c in chunks)


def test_chunk_overlap_shares_trailing_text(processor):
    # Each token is unique across the whole text, so any word shared between
    # consecutive chunks must have come from the overlap region.
    text = " ".join(f"sentence{i}." for i in range(60))
    chunks = processor._create_chunks(text, "doc.txt", [])

    assert len(chunks) > 1
    shared_words = set(chunks[0].text.split()) & set(chunks[1].text.split())
    assert shared_words
