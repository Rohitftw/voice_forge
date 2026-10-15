"""Resumable model download. Re-run safely: finished files are skipped, partial files continue."""
import os, time
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from huggingface_hub import snapshot_download

REPO = "ResembleAI/chatterbox-turbo"
for attempt in range(1, 41):
    try:
        print("Done:", snapshot_download(REPO, max_workers=2))
        break
    except Exception as e:
        print(f"Attempt {attempt} interrupted: {e}\nRetrying in 5s. Progress is kept.")
        time.sleep(5)
else:
    print("Still failing after 40 tries. Check your connection, or try a different network.")
