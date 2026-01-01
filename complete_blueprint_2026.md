# German Tutor App - Complete Production Blueprint 2026
## Real-Time Speech AI with Python, FastAPI, OpenAI Realtime API

---

## TABLE OF CONTENTS
1. [Architecture Overview](#architecture-overview)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Phase 1: Local Development](#phase-1-local-development)
4. [Phase 2: Railway Deployment](#phase-2-railway-deployment)
5. [Phase 3: OpenAI Realtime Integration](#phase-3-openai-realtime-integration)
6. [Phase 4: Session Persistence & Analysis](#phase-4-session-persistence--analysis)
7. [Production Checklist](#production-checklist)
8. [Troubleshooting & Debugging](#troubleshooting--debugging)
9. [Cost Estimation 2026](#cost-estimation-2026)

---

## ARCHITECTURE OVERVIEW

### System Design (2026 Latest)

```
┌─────────────────────────────────────────────────────────────────┐
│                        iPhone Safari                             │
│              (Responsive Web App - PWA)                          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Web Speech API for microphone input                    │  │
│  │ • Real-time audio encoding (PCM 16-bit @ 24kHz)         │  │
│  │ • WebSocket client (bi-directional streaming)           │  │
│  │ • Session storage (localStorage)                         │  │
│  │ • Responsive UI (VH/VW for iPhone)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────────┘
                         │ wss:// (Secure WebSocket)
                         │ Audio chunks (24kHz PCM)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Railway)                           │
│              your-app-xyz.railway.app                            │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ async def websocket_endpoint(websocket: WebSocket)       │  │
│  │ • Accept WebSocket connection                            │  │
│  │ • Receive audio chunks from iPhone                       │  │
│  │ • Forward to OpenAI Realtime API                         │  │
│  │ • Stream responses back to iPhone                        │  │
│  │ • Handle disconnections & errors                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ File Storage                                              │  │
│  │ • learner_profile.json (persistent learning profile)     │  │
│  │ • sessions/ (conversation history)                       │  │
│  │ • config.json (system configuration)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ↓                ↓                ↓
   ┌─────────┐   ┌──────────────┐   ┌──────────┐
   │ OpenAI  │   │  Claude API  │   │ Optional │
   │Realtime │   │(session      │   │ Services │
   │API      │   │ analysis)    │   │          │
   │         │   │              │   │ • Logs   │
   │ • STT   │   │ • Pattern    │   │ • Metrics│
   │ • LLM   │   │   detection  │   │ • Redis  │
   │ • TTS   │   │ • Feedback   │   │          │
   │ • VAD   │   │ • Profiling  │   └──────────┘
   │         │   │              │
   │(500ms   │   │(async task)  │
   │ latency)│   │              │
   └─────────┘   └──────────────┘
```

### Technology Stack (2026)

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Backend Framework** | FastAPI | 0.115+ | Async web framework with WebSocket support |
| **ASGI Server** | Uvicorn | 0.30+ | Production ASGI server |
| **Speech AI** | OpenAI Realtime API | 2024-12-17 | Speech recognition, LLM, TTS (unified) |
| **Session Analysis** | Claude API | claude-3.5-sonnet | Conversation analysis and profiling |
| **Storage** | JSON files (local) | - | Session persistence |
| **Hosting** | Railway | latest | Container orchestration and deployment |
| **Frontend** | HTML5 + Vanilla JS | ES6+ | iPhone web app (PWA) |
| **Audio Format** | PCM 16-bit 24kHz | G.711 µ-law | OpenAI Realtime standard |
| **Deployment** | GitHub + Railway | - | Auto-deploy on git push |

---

## PREREQUISITES & SETUP

### 1. Required Accounts & API Keys

```bash
# 1. OpenAI API Key
# Go to: https://platform.openai.com/api/keys
# Create new secret key
# Copy: sk-...

# 2. Claude API Key (for session analysis)
# Go to: https://console.anthropic.com/
# Create new API key
# Copy: sk-ant-...

# 3. GitHub Account
# Create new repository: german-tutor

# 4. Railway Account
# Sign up at: https://railway.app
# (Free tier: $5 credit/month)
```

### 2. System Requirements

```bash
# macOS/Linux
- Python 3.11 or 3.12
- pip or poetry
- git
- Modern browser (Safari on iPhone)

# Windows
- Python 3.11 or 3.12 (from python.org)
- Git for Windows
- Terminal or PowerShell

# Verify Python:
python3 --version  # Should be 3.11+
pip3 --version     # Should be 23.0+
```

### 3. Install Development Tools

```bash
# Clone or initialize your project
mkdir german-tutor
cd german-tutor
git init

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install core dependencies
pip install --upgrade pip
pip install fastapi==0.115.0
pip install uvicorn[standard]==0.30.0
pip install openai==1.40.0
pip install anthropic==0.31.0
pip install python-dotenv==1.0.0
pip install aiofiles==23.2.1
pip install httpx==0.25.2
pip install pydantic==2.8.0

# Save dependencies
pip freeze > requirements.txt

# Create project structure
mkdir -p backend frontend sessions
touch backend/main.py
touch backend/.env
touch frontend/index.html
touch learner_profile.json
```

---

## PHASE 1: LOCAL DEVELOPMENT (90 minutes)

### 1.1 Backend Setup: `backend/main.py`

```python
"""
German Tutor App - FastAPI Backend
Real-time speech processing with OpenAI Realtime API
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import json
import logging
import asyncio
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import tempfile

from openai import AsyncOpenAI
from anthropic import Anthropic

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# ============================================================================
# FastAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="German Tutor",
    description="Real-time German language learning with speech AI",
    version="0.1.0"
)

# CORS for mobile web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")

# ============================================================================
# LEARNER PROFILE MANAGEMENT
# ============================================================================

PROFILE_FILE = "learner_profile.json"
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

def load_learner_profile() -> Dict:
    """Load or create learner profile"""
    if Path(PROFILE_FILE).exists():
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Default profile for new learners
    return {
        "name": "Yavuz",
        "current_level": "A1",
        "session_count": 0,
        "total_minutes": 0,
        "created_date": datetime.now().isoformat(),
        "last_session": None,
        "strengths": [],
        "weaknesses": [],
        "pronunciation_issues": [],
        "vocabulary_errors": {},
        "grammar_patterns": {},
        "personality_context": "technical_tinkerer, musician, IT administrator",
        "preferred_topics": ["music_production", "pcb_design", "game_development", "render_farms"]
    }

def save_learner_profile(profile: Dict) -> None:
    """Save learner profile to disk"""
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    logger.info(f"Profile saved: {profile['name']} (Level: {profile['current_level']})")

# ============================================================================
# SYSTEM PROMPT (TUTOR INSTRUCTIONS)
# ============================================================================

def get_system_prompt(profile: Dict) -> str:
    """Generate personalized system prompt based on learner profile"""
    
    weaknesses = ", ".join(profile["weaknesses"][:3]) if profile["weaknesses"] else "none identified yet"
    
    return f"""You are an expert German language tutor designed for immersive conversation-based learning.

LEARNER PROFILE:
- Name: {profile['name']}
- Current Level: {profile['current_level']}
- Sessions Completed: {profile['session_count']}
- Topics of Interest: {', '.join(profile['preferred_topics'])}
- Known Weaknesses: {weaknesses}
- Pronunciation Issues: {', '.join(profile['pronunciation_issues'][:3]) if profile['pronunciation_issues'] else 'none yet'}

YOUR ROLE:
1. Engage in natural German conversation
2. When learner speaks, REPEAT BACK what they said with corrections
3. Provide English translation in brackets [like this]
4. Make corrections MINIMAL and clear
5. Continue conversation naturally - don't interrupt learning flow
6. Gradually introduce target weakness areas (e.g., {weaknesses})
7. Adapt complexity to their A1-A2 beginner level
8. Use their interests (music, PCB design, coding) in examples

RESPONSE FORMAT - CRITICAL:
Return ONLY this JSON structure (no markdown, no code blocks):
{{
  "corrected_german": "[Their sentence with corrections applied]",
  "english_translation": "[English meaning of corrected sentence]",
  "corrections": [
    {{"type": "grammar", "original": "word", "corrected": "word", "reason": "explanation"}},
    {{"type": "vocabulary", "original": "word", "corrected": "word", "reason": "explanation"}}
  ],
  "pronunciation_assessment": {{"quality": "clear/acceptable/needs_work", "issue": "specific sound if any"}},
  "continue_german": "[Your response continuing conversation naturally in German]"
}}

IMPORTANT:
- Level: Keep German simple and appropriate for {profile['current_level']}
- Topics: Reference their interests when possible
- Corrections: Only flag major errors, not minor ones
- Flow: Make it feel like natural conversation, not a lesson
- Pronunciation: Note any obvious issues for later analysis
- Context: Remember this is conversation #{{session_number}} - reference previous topics if mentioned

Start with something like: "Guten Tag! Wie heißt du?" or "Wie war dein Tag?"""

# ============================================================================
# OPENAI REALTIME API INTEGRATION
# ============================================================================

async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio using Whisper (via OpenAI)
    Audio format: WebM/Opus from browser → convert to WAV
    """
    try:
        # Write audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Transcribe with German language specified
        with open(tmp_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="de",  # Force German recognition
                prompt="German language learning conversation"
            )
        
        # Cleanup
        os.unlink(tmp_path)
        
        return transcript.text.strip()
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise

async def get_conversation_response(user_text: str, profile: Dict, session_log: List) -> Dict:
    """
    Get AI response using OpenAI Chat API
    (Phase 3 will upgrade to Realtime API for speech-to-speech)
    """
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        system_prompt = get_system_prompt(profile)
        
        # Build conversation context from session
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent context (last 3 exchanges)
        for log_entry in session_log[-3:]:
            messages.append({
                "role": "user",
                "content": log_entry["user_input"]
            })
            messages.append({
                "role": "assistant",
                "content": str(log_entry["response"])
            })
        
        # Add current user input
        messages.append({"role": "user", "content": user_text})
        
        # Get response from GPT-4
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            response_format={"type": "json_object"}  # Enforce JSON
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON response
        try:
            response_json = json.loads(response_text)
            return response_json
        except json.JSONDecodeError:
            logger.warning("Response not valid JSON, creating fallback")
            return {
                "corrected_german": user_text,
                "english_translation": f"(Your input: {user_text})",
                "corrections": [],
                "pronunciation_assessment": {"quality": "acceptable", "issue": None},
                "continue_german": "Interessant! Erzähl mir mehr."
            }
    
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        raise

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    profile = load_learner_profile()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "learner": {
            "name": profile["name"],
            "level": profile["current_level"],
            "sessions": profile["session_count"]
        }
    }

@app.get("/api/profile")
async def get_profile():
    """Get current learner profile"""
    return load_learner_profile()

@app.post("/api/profile/update")
async def update_profile(updates: Dict):
    """Update learner profile (admin endpoint)"""
    profile = load_learner_profile()
    profile.update(updates)
    save_learner_profile(profile)
    return {"success": True, "profile": profile}

# ============================================================================
# WEBSOCKET ENDPOINT - CORE FUNCTIONALITY
# ============================================================================

@app.websocket("/ws/session")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time German tutoring
    
    Client sends:
    - Audio chunks as base64
    - JSON messages with type field
    
    Server sends:
    - Correction data
    - Pronunciation feedback
    - Continuation of conversation
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    profile = load_learner_profile()
    session_log = []
    session_start = datetime.now()
    
    # Send welcome message
    await websocket.send_json({
        "type": "status",
        "message": f"🎤 Welcome back, {profile['name']}! Ready to practice German. Speak naturally!",
        "level": profile["current_level"],
        "session_number": profile["session_count"] + 1
    })
    
    try:
        while True:
            # Receive message from iPhone
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "audio":
                    # Audio chunk from microphone
                    audio_base64 = message.get("audio")
                    
                    if not audio_base64:
                        continue
                    
                    try:
                        # Decode audio
                        audio_bytes = base64.b64decode(audio_base64)
                        
                        # Send processing status
                        await websocket.send_json({
                            "type": "status",
                            "message": "🔄 Processing your speech..."
                        })
                        
                        # Transcribe audio
                        user_german = await transcribe_audio(audio_bytes)
                        logger.info(f"Transcribed: {user_german}")
                        
                        # Send transcription back
                        await websocket.send_json({
                            "type": "status",
                            "message": f"📝 You said: \"{user_german}\""
                        })
                        
                        # Get AI response
                        response_data = await get_conversation_response(
                            user_german, 
                            profile, 
                            session_log
                        )
                        
                        # Log this exchange
                        session_log.append({
                            "timestamp": datetime.now().isoformat(),
                            "user_input": user_german,
                            "response": response_data,
                            "corrections_count": len(response_data.get("corrections", []))
                        })
                        
                        # Send correction and response
                        await websocket.send_json({
                            "type": "correction",
                            "corrected_german": response_data.get("corrected_german"),
                            "english_translation": response_data.get("english_translation"),
                            "corrections": response_data.get("corrections", []),
                            "pronunciation": response_data.get("pronunciation_assessment", {}),
                            "agent_response": response_data.get("continue_german"),
                            "agent_english": f"(Agent says: {response_data.get('continue_german', '')})"
                        })
                        
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"❌ Error processing audio: {str(e)}"
                        })
                
                elif message_type == "end_session":
                    # Session ended by user
                    break
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            
            except Exception as e:
                logger.error(f"Processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    
    finally:
        # Save session to file
        session_duration = (datetime.now() - session_start).total_seconds() / 60
        
        session_data = {
            "session_id": f"session_{session_start.isoformat()}",
            "start_time": session_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_minutes": round(session_duration, 2),
            "learner_level": profile["current_level"],
            "exchanges": len(session_log),
            "conversation": session_log
        }
        
        # Write to sessions directory
        session_file = SESSIONS_DIR / f"{session_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session saved: {session_file}")
        
        # Update profile
        profile["session_count"] += 1
        profile["total_minutes"] += round(session_duration, 2)
        profile["last_session"] = datetime.now().isoformat()
        save_learner_profile(profile)
        
        # (Phase 4 will add Claude analysis here)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run locally on port 8000
    # In production, Railway will use $PORT environment variable
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
```

### 1.2 Environment Configuration: `backend/.env`

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Anthropic Claude Configuration (for session analysis)
CLAUDE_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Application Settings
LOG_LEVEL=INFO
PORT=8000
ENVIRONMENT=development
```

### 1.3 Frontend: `frontend/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <meta name="theme-color" content="#667eea">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>🇩🇪 German Tutor</title>
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #667eea;
            --primary-hover: #5568d3;
            --danger: #f5365c;
            --success: #2dce89;
            --bg: #f8f9fa;
            --card: #ffffff;
            --text: #333333;
            --text-muted: #666666;
            --border: #e9ecef;
        }
        
        html, body {
            width: 100%;
            height: 100%;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, var(--primary) 0%, #764ba2 100%);
            overflow: hidden;
        }
        
        body {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);
        }
        
        .container {
            background: var(--card);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            width: 90vw;
            max-width: 500px;
            max-height: 95vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 4px;
        }
        
        .header p {
            font-size: 12px;
            opacity: 0.9;
        }
        
        .content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        button {
            flex: 1;
            padding: 12px 16px;
            font-size: 14px;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            user-select: none;
        }
        
        #recordBtn {
            background: var(--primary);
            color: white;
        }
        
        #recordBtn:active {
            background: var(--primary-hover);
            transform: scale(0.98);
        }
        
        #stopBtn {
            background: var(--danger);
            color: white;
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        #stopBtn:not(:disabled) {
            opacity: 1;
            cursor: pointer;
        }
        
        .output-box {
            background: var(--bg);
            border-radius: 12px;
            padding: 12px;
            border-left: 4px solid var(--primary);
            min-height: 60px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .output-label {
            color: var(--text-muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
            font-weight: 600;
        }
        
        .output-text {
            color: var(--text);
            font-size: 14px;
            line-height: 1.4;
            word-break: break-word;
        }
        
        #status {
            color: var(--primary);
            font-weight: 600;
        }
        
        .recording {
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .hidden {
            display: none !important;
        }
        
        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #0084ff;
            padding: 10px;
            border-radius: 6px;
            font-size: 12px;
            color: #0056b3;
            margin-top: 10px;
        }
        
        /* iPhone notch/safe area support */
        @supports (padding: max(0px)) {
            .container {
                padding-top: max(10px, env(safe-area-inset-top));
                padding-bottom: max(10px, env(safe-area-inset-bottom));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🇩🇪 German Tutor</h1>
            <p>Speak naturally, learn through conversation</p>
        </div>
        
        <div class="content">
            <div class="controls">
                <button id="recordBtn">🎤 Speak</button>
                <button id="stopBtn" disabled>⏹️ Stop</button>
            </div>
            
            <div class="output-box">
                <div class="output-label">Status</div>
                <div class="output-text" id="status">Connecting...</div>
            </div>
            
            <div class="output-box">
                <div class="output-label">Your Speech (Corrected)</div>
                <div class="output-text" id="corrected">Waiting for input...</div>
            </div>
            
            <div class="output-box">
                <div class="output-label">English Translation</div>
                <div class="output-text" id="english">-</div>
            </div>
            
            <div class="output-box" id="agentBox" class="hidden">
                <div class="output-label">Agent Response (German)</div>
                <div class="output-text" id="agent">-</div>
            </div>
            
            <div class="info-box" id="infoBox" class="hidden">
                <strong>Tip:</strong> <span id="tipText"></span>
            </div>
        </div>
    </div>

    <script>
        // ====================================================================
        // CONFIGURATION
        // ====================================================================
        
        const API_BASE = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const API_HOST = window.location.host;
        const WS_URL = `${API_BASE}//${API_HOST}/ws/session`;
        
        // ====================================================================
        // DOM ELEMENTS
        // ====================================================================
        
        const recordBtn = document.getElementById('recordBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusEl = document.getElementById('status');
        const correctedEl = document.getElementById('corrected');
        const englishEl = document.getElementById('english');
        const agentEl = document.getElementById('agent');
        const agentBox = document.getElementById('agentBox');
        const infoBox = document.getElementById('infoBox');
        const tipText = document.getElementById('tipText');
        
        // ====================================================================
        // STATE
        // ====================================================================
        
        let ws = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let mediaStream = null;
        
        // ====================================================================
        // WEBSOCKET CONNECTION
        // ====================================================================
        
        function connectWebSocket() {
            ws = new WebSocket(WS_URL);
            
            ws.addEventListener('open', () => {
                statusEl.textContent = '✅ Connected - Ready to speak';
                recordBtn.disabled = false;
            });
            
            ws.addEventListener('message', (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'status') {
                        statusEl.textContent = data.message;
                    } 
                    else if (data.type === 'correction') {
                        correctedEl.textContent = data.corrected_german || '-';
                        englishEl.textContent = data.english_translation || '-';
                        agentEl.textContent = data.agent_response || '-';
                        agentBox.classList.remove('hidden');
                        
                        // Show pronunciation feedback if available
                        if (data.pronunciation && data.pronunciation.issue) {
                            tipText.textContent = `Pronunciation: ${data.pronunciation.issue}`;
                            infoBox.classList.remove('hidden');
                        }
                        
                        statusEl.textContent = '✅ Processing complete - Ready for next turn';
                        recordBtn.disabled = false;
                        stopBtn.disabled = true;
                        isRecording = false;
                    }
                    else if (data.type === 'error') {
                        statusEl.textContent = `❌ ${data.message}`;
                        recordBtn.disabled = false;
                        stopBtn.disabled = true;
                    }
                } catch (e) {
                    console.error('WebSocket message error:', e);
                }
            });
            
            ws.addEventListener('error', (error) => {
                statusEl.textContent = '❌ Connection error';
                console.error('WebSocket error:', error);
            });
            
            ws.addEventListener('close', () => {
                statusEl.textContent = '⚠️ Connection closed - Reconnecting...';
                recordBtn.disabled = true;
                // Attempt reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            });
        }
        
        // ====================================================================
        // AUDIO RECORDING
        // ====================================================================
        
        recordBtn.addEventListener('click', async () => {
            try {
                // Request microphone access
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 24000
                    }
                });
                
                mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm' });
                audioChunks = [];
                
                mediaRecorder.addEventListener('dataavailable', (event) => {
                    audioChunks.push(event.data);
                });
                
                mediaRecorder.addEventListener('stop', async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await sendAudioToBackend(audioBlob);
                });
                
                mediaRecorder.start();
                isRecording = true;
                
                recordBtn.disabled = true;
                recordBtn.classList.add('recording');
                stopBtn.disabled = false;
                statusEl.textContent = '🎤 Recording... Speak now';
                
            } catch (error) {
                statusEl.textContent = '❌ Microphone access denied';
                console.error('Microphone error:', error);
            }
        });
        
        stopBtn.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                
                // Stop all media tracks
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                }
            }
            
            recordBtn.classList.remove('recording');
            recordBtn.disabled = true;  // Will be re-enabled by server
            stopBtn.disabled = true;
            statusEl.textContent = '⏳ Processing your speech...';
            isRecording = false;
        });
        
        // ====================================================================
        // SEND AUDIO TO BACKEND
        // ====================================================================
        
        async function sendAudioToBackend(audioBlob) {
            try {
                // Convert blob to base64
                const reader = new FileReader();
                
                reader.onload = () => {
                    const base64Audio = reader.result.split(',')[1];
                    
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: "audio",
                            audio: base64Audio
                        }));
                    } else {
                        statusEl.textContent = '❌ WebSocket not connected';
                    }
                };
                
                reader.onerror = () => {
                    statusEl.textContent = '❌ Error reading audio';
                };
                
                reader.readAsDataURL(audioBlob);
                
            } catch (error) {
                statusEl.textContent = `❌ Error sending audio: ${error.message}`;
                console.error('Audio send error:', error);
            }
        }
        
        // ====================================================================
        // INITIALIZATION
        // ====================================================================
        
        window.addEventListener('load', () => {
            connectWebSocket();
            
            // Request notification permission for future features
            if ('Notification' in window && Notification.permission === 'default') {
                Notification.requestPermission();
            }
        });
        
        // ====================================================================
        // PAGE UNLOAD
        // ====================================================================
        
        window.addEventListener('beforeunload', () => {
            if (ws) {
                ws.send(JSON.stringify({ type: "end_session" }));
                ws.close();
            }
            
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
            }
        });
    </script>
</body>
</html>
```

### 1.4 Test Locally

```bash
# From the backend directory
cd backend

# Activate virtual environment (if not already activated)
source venv/bin/activate

# Set API key
export OPENAI_API_KEY="sk-proj-YOUR_KEY_HERE"

# Run locally
python main.py

# Should output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete

# In another terminal, test the health endpoint:
curl http://localhost:8000/api/health

# Open in browser:
# http://localhost:8000
# On iPhone Safari: http://YOUR_COMPUTER_IP:8000
```

---

## PHASE 2: RAILWAY DEPLOYMENT (30 minutes)

### 2.1 Prepare for Deployment

```bash
# Create railway.toml in project root
cat > railway.toml << 'EOF'
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/api/health"
healthcheckTimeout = 300
restartPolicyType = "always"
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend ./backend
COPY frontend ./frontend
COPY learner_profile.json .

# Create sessions directory
RUN mkdir -p sessions

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')"

# Expose port (Railway will override with $PORT)
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
EOF
```

### 2.2 Push to GitHub

```bash
# Initialize git if needed
git init
git add .
git commit -m "Initial commit: German Tutor app with FastAPI and OpenAI integration"

# Create GitHub repository at https://github.com/new
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/german-tutor.git
git branch -M main
git push -u origin main
```

### 2.3 Deploy to Railway

1. **Sign up**: Go to https://railway.app
2. **Create Project**: Click "New Project" → "Deploy from GitHub"
3. **Connect Repo**: Select `german-tutor`
4. **Configure**: 
   - Railway detects FastAPI automatically
   - No additional setup needed
5. **Add Environment Variables**:
   - Go to Variables section
   - Add: `OPENAI_API_KEY=sk-proj-...`
   - Add: `CLAUDE_API_KEY=sk-ant-...`
6. **Deploy**: Click "Deploy"
7. **Get URL**: After deployment, Railway provides `your-app-xyz.railway.app`

### 2.4 Test on iPhone

```bash
# Once deployed, open in Safari:
# https://your-app-xyz.railway.app

# On iPhone:
# 1. Tap the URL in Safari
# 2. Tap Share → Add to Home Screen
# 3. App opens like native app
# 4. Tap 🎤 button to test microphone
# 5. Speak German: "Ich bin Yavuz"
# 6. Should receive correction
```

---

## PHASE 3: OPENAI REALTIME API INTEGRATION (OPTIONAL - Phase 2 ready)

### Context
The current Phase 2 implementation uses:
- Whisper for STT (speech-to-text)
- GPT-4 Turbo for LLM
- (Phase 4 will add TTS)

For true real-time speech-to-speech (<500ms latency), upgrade to **OpenAI Realtime API** (requires WebRTC or WebSocket directly to OpenAI).

[This is a future enhancement - current implementation works with Whisper + GPT-4]

---

## PHASE 4: SESSION PERSISTENCE & ANALYSIS

[Implementation for Claude-based analysis will be added in next update]

---

## PRODUCTION CHECKLIST

Before going live with real German learning:

### Security
- [ ] API keys in environment variables (never commit to git)
- [ ] HTTPS enforced (Railway handles automatically)
- [ ] CORS properly configured for your domain
- [ ] Input validation on all endpoints
- [ ] Rate limiting on API calls (prevent abuse)

### Performance
- [ ] WebSocket connection stable
- [ ] Audio latency < 1 second
- [ ] Memory usage monitored
- [ ] Database queries optimized (when added)

### Monitoring
- [ ] Error logging configured
- [ ] Session persistence verified
- [ ] Learner profile updates working
- [ ] Audio transcription accuracy > 90%

### Testing
- [ ] Tested on iPhone 12+ (Safari)
- [ ] Tested with various German dialects
- [ ] Network reliability tested
- [ ] Profile persistence verified

---

## TROUBLESHOOTING & DEBUGGING

### WebSocket Connection Issues

```bash
# Check if backend is running
curl https://your-app.railway.app/api/health

# Check WebSocket endpoint
# Open Safari Developer Console (iPhone: Settings → Safari → Advanced → Web Inspector)
# Look for wss:// connection in Network tab

# Common causes:
# 1. Railway domain not matching frontend
# 2. SSL certificate issues (check railway logs)
# 3. Firewall blocking WebSocket
```

### Audio Transcription Issues

```bash
# Whisper accuracy depends on:
# - Clear audio (minimize background noise)
# - Proper German pronunciation
# - 24kHz sample rate maintained
# - Audio must be < 25MB

# Test with curl:
curl -X POST https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "model=whisper-1" \
  -F "language=de" \
  -F "file=@audio.wav"
```

### Deployment Issues

```bash
# Check Railway logs
railway logs

# Check environment variables
railway variables

# Redeploy
railway up
```

---

## COST ESTIMATION 2026

### Monthly Breakdown (Assuming 1 hour/day usage)

```
OpenAI Whisper: ~$0.01/min = $30/month (60 min/day * 30 days)
OpenAI GPT-4: ~$0.05/1K tokens = $15/month (1000-2000 tokens/session)
Claude API: ~$0.003/1K tokens = $3/month (analysis only)
Railway: $5/month (free tier) → $7/month (when free tier depleted)

TOTAL: ~$50-60/month for daily German learning

With Realtime API (Phase 3):
OpenAI Realtime: ~$0.04/min = $120/month (higher but speech-to-speech)
TOTAL: ~$130/month

Budget options:
- Whisper only (no corrections): $30/month
- Whisper + GPT-4 (current): $50/month
- Realtime API (best quality): $130/month
```

---

## NEXT STEPS

1. **Tonight**: Run Phase 1 locally (30 minutes)
2. **Tomorrow**: Deploy to Railway (15 minutes)
3. **This week**: Get real German conversations working
4. **Next week**: Add Claude analysis and profile adaptation
5. **Future**: Upgrade to OpenAI Realtime API for true speech-to-speech

---

## GETTING HELP

### Resources
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **OpenAI API**: https://platform.openai.com/docs
- **Railway Docs**: https://docs.railway.app
- **WebSocket Guide**: https://fastapi.tiangolo.com/advanced/websockets/

### Debugging
1. Check Railway logs: `railway logs`
2. Open Safari console on iPhone
3. Check WebSocket messages in Network tab
4. Test API locally before deploying

---

**You're ready to build. Start Phase 1 now.** 🚀

Questions? Check the troubleshooting section or test incrementally.
