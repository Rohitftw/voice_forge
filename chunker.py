import re

def chunk_text(text: str, limit: int = 240) -> list[str]:
    """Split a script into sentence-sized chunks. Emotion tags like [laugh] stay inside the text."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = []
    for sent in re.split(r"(?<=[.!?…])\s+", text):
        # a very long sentence gets split again on commas so the model never sees a wall of text
        if len(sent) > limit:
            pieces += re.split(r"(?<=[,;:])\s+", sent)
        else:
            pieces.append(sent)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks
