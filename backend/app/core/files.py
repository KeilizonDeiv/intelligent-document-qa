import re
import unicodedata


def secure_filename(filename: str) -> str:
    """Strip path components and any character that isn't safe on disk."""
    filename = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("ascii")
    filename = filename.replace("/", " ").replace("\\", " ")
    filename = re.sub(r"[^A-Za-z0-9._ -]", "", filename).strip().strip(".")
    filename = re.sub(r"\s+", "_", filename)
    return filename or "file"
