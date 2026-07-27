import uvicorn
import os

if __name__ == "__main__":
    # Hugging Face Spaces exposes port 7860 by default for Gradio spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("route:app", host="0.0.0.0", port=port)
