# 🇩🇪 German Tutor - Real-Time Conversational Speech Agent

A voice-based German language learning application with real-time speech-to-text, AI conversation, corrections, and text-to-speech feedback.

## Features

- **Push-to-Talk Voice Input** - Hold button to speak, release to send
- **Real-Time Speech Processing** - Whisper API for German transcription
- **AI Conversation** - GPT-4 powered German tutor
- **Automatic Corrections** - Grammar and vocabulary corrections with explanations
- **Text-to-Speech Playback** - OpenAI TTS (nova voice) for natural German pronunciation
- **Chat Interface** - WhatsApp-style bubbles with inline corrections
- **Session Management** - View, create, and delete past learning sessions
- **Progress Tracking** - Learner profile with level tracking

## Tech Stack

**Backend:**
- FastAPI (async web framework)
- OpenAI Whisper API (speech-to-text)
- OpenAI GPT-4 (conversation AI)
- OpenAI TTS (text-to-speech)
- WebSocket (real-time communication)

**Frontend:**
- Vanilla JavaScript (no frameworks)
- MediaRecorder API (audio recording)
- WebSocket client
- Responsive UI optimized for mobile

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key
- Anthropic API key (optional, for future features)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd german_tutor
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment file:
```bash
cp backend/.env.example backend/.env
```

5. Add your API keys to `backend/.env`:
```
OPENAI_API_KEY=your_openai_key_here
CLAUDE_API_KEY=your_anthropic_key_here
```

### Running the App

1. Start the backend server:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

2. Open your browser to:
```
http://localhost:8000
```

## Usage

1. **Grant microphone permission** when prompted
2. **Hold** the "🎤 Hold to Speak" button
3. **Say something in German** (e.g., "Hallo, wie geht es dir?")
4. **Release** the button to send
5. View your corrected message and AI response
6. Listen to the AI's voice response automatically

### Session Management

- **New** - Start a fresh conversation
- **Dropdown** - View past sessions (date, time, exchanges)
- **Delete** - Remove old sessions

## Project Structure

```
german_tutor/
├── backend/
│   ├── main.py          # FastAPI app, WebSocket, API endpoints
│   └── .env             # API keys (not in git)
├── frontend/
│   └── index.html       # Single-page app with CSS & JS
├── sessions/            # Session JSON files (not in git)
├── learner_profile.json # User progress (not in git)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## API Endpoints

- `GET /` - Serve frontend
- `GET /api/health` - Health check
- `GET /api/profile` - Get learner profile
- `GET /api/sessions` - List all sessions
- `GET /api/sessions/{id}` - Get session details
- `POST /api/sessions/new` - Create new session
- `DELETE /api/sessions/{filename}` - Delete session
- `WS /ws/session` - WebSocket for real-time conversation

## Cost Estimates

**Per Session (~10 exchanges):**
- Whisper API: ~$0.006 (60 seconds total)
- GPT-4: ~$0.02 (2,000 tokens)
- TTS: ~$0.04 (2,500 characters)
- **Total: ~$0.07 per session**

## Future Enhancements

- OpenAI Realtime API (1-2s latency speech-to-speech)
- Word-level pronunciation scoring
- Claude-powered adaptive difficulty
- Progress dashboard with analytics
- Spaced repetition vocabulary system

## License

MIT

## Version

v0.1 - Initial release with core conversational features
