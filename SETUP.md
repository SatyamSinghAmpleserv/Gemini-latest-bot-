# Gemini Chatbot Project Setup

Follow these instructions to set up and run the project locally.

## Prerequisites
- Python 3.10 or higher installed on your system.

## Setup Instructions

### 1. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

**For Mac/Linux:**
```bash
python3 -m venv venv
```

**For Windows:**
```cmd
python -m venv venv
```

### 2. Activate the Virtual Environment

**For Mac/Linux:**
```bash
source venv/bin/activate
```

**For Windows:**
```cmd
venv\Scripts\activate
```

### 3. Install Dependencies
Once the virtual environment is activated, install all required packages using the `requirement.txt` file.

```bash
pip install -r requirement.txt
```

### 4. Set Environment Variables
Ensure you have a `.env` file in the root of your project directory containing the necessary API keys and database credentials (e.g., `GEMINI_API_KEY`).

### 5. Run the Application
Start the FastAPI server. 

**Using the provided script:**
```bash
python run_app.py
```
*(The server will start locally on `http://0.0.0.0:6542`)*

---
**Note:** On the first run, the HuggingFace AI models (like `BAAI/bge-large-en-v1.5`) will be downloaded automatically. This can take a few minutes depending on your internet speed.
