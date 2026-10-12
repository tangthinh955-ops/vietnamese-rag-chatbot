import pytest

from rag.loaders import load_directory, load_file


def test_load_markdown(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# Tiêu đề\n\nNội dung   \n\n\n", encoding="utf-8")
    docs = load_file(f)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "a.md"
    assert docs[0].text.endswith("Nội dung")


def test_load_directory_skips_unsupported(tmp_path):
    (tmp_path / "ok.txt").write_text("nội dung", encoding="utf-8")
    (tmp_path / "skip.exe").write_bytes(b"x")
    docs = load_directory(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "ok.txt"


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(f)
