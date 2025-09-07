# Project Setup and Run Instructions

This guide provides step-by-step instructions to set up and run the project, including both the backend and frontend components.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python**: Version 3.x (recommended: 3.8 or higher).
- **Next.js**: Requires Node.js (version 16.x or higher) for the frontend.
- **Ollama**: Download and install Ollama from [https://ollama.com/](https://ollama.com/).

## Backend Setup

1. **Pull Required Ollama Models**:
   Open PowerShell and run the following commands to download the necessary models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. **Start Ollama Server**:
   Run the following command to start the Ollama server:
   ```bash
   ollama serve
   ```
   - If you see a message indicating that the socket is already in use, the server is ready, and you can proceed.

3. **Install Python Dependencies**:
   Navigate to the `backend` folder, create variable environment for python
   ```bash
   python -m venv venv
   ```

   Navigate to the `backend` folder, which contains a `requirements.txt` file listing all required Python libraries. Install them using the following command:
   ```bash
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Activate Virtual Environment** if you haven't:
   In the `backend` directory, activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```

5. **Run the Backend**:
   With the virtual environment activated, start the backend server using Uvicorn:
   ```bash
   uvicorn app.main:app --reload
   ```

## Frontend Setup

1. **Install Node.js Dependencies**:
   Navigate to the `frontend` directory and install the required Node.js packages:
   ```bash
   npm i
   ```

2. **Run the Frontend**:
   In the `frontend` directory, start the development server:
   ```bash
   npm run dev
   ```

## Notes

- Ensure that both the backend and frontend servers are running simultaneously for the project to function correctly.
- If you encounter any issues, verify that all prerequisites are installed correctly and that the Ollama server is running.
