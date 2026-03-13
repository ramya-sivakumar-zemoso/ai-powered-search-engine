import os
 
MEILI_URL   = os.environ.get("MEILI_URL", "http://127.0.0.1:7700")
MASTER_KEY  = os.environ.get("MEILI_MASTER_KEY", "aSampleMasterKey")
OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "sampleAPIkey")
 
MASTER_HEADERS = {
    "Content-Type":  "application/json",
    "Authorization": f"Bearer {MASTER_KEY}"
}
 
SEARCH_HEADERS = {
    "Content-Type":  "application/json",
    "Authorization": f"Bearer {MASTER_KEY}"
}