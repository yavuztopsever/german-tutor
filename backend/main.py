"""
German Tutor App - FastAPI Backend
Real-time speech processing with OpenAI Realtime API
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

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

# Session management configuration
AUTO_SAVE_INTERVAL = 3  # Save session every N exchanges
CONTEXT_WINDOW_SIZE = 8  # Number of past exchanges to include in GPT-4 context

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

RESPONSE FORMAT - ABSOLUTELY CRITICAL:
You MUST return ONLY valid JSON. No markdown, no code blocks, no extra text, no backticks.
Start your response with {{ and end with }}.

Required JSON structure:
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

EXAMPLE VALID RESPONSE:
{{"corrected_german": "Hallo, ich bin Yavuz.", "english_translation": "Hello, I am Yavuz.", "corrections": [], "pronunciation_assessment": {{"quality": "clear", "issue": null}}, "continue_german": "Schön dich kennenzulernen! Wie geht es dir heute?"}}

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

def save_session_checkpoint(session_start: datetime, session_log: List, profile: Dict, session_file_path: Path):
    """Save session checkpoint to prevent data loss"""
    try:
        session_data = {
            "session_id": f"session_{session_start.isoformat()}",
            "start_time": session_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_minutes": round((datetime.now() - session_start).total_seconds() / 60, 2),
            "learner_level": profile["current_level"],
            "exchanges": len(session_log),
            "conversation_log": session_log
        }

        with open(session_file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Session checkpoint saved: {len(session_log)} exchanges")
    except Exception as e:
        logger.error(f"Failed to save session checkpoint: {e}")
        # Don't raise - non-critical failure

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

        transcribed_text = transcript.text.strip()

        # Check for poor audio quality (empty or very short transcription)
        if not transcribed_text or len(transcribed_text) < 3:
            raise ValueError("Audio quality too poor for transcription")

        return transcribed_text

    except Exception as e:
        error_msg = str(e).lower()

        # Check for specific audio quality errors
        if "audio quality" in error_msg or "could not transcribe" in error_msg or len(str(e)) < 3:
            logger.warning(f"Poor audio quality: {e}")
            raise ValueError("AUDIO_QUALITY_ERROR")

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

        # Add recent context (configurable window size)
        for log_entry in session_log[-CONTEXT_WINDOW_SIZE:]:
            messages.append({
                "role": "user",
                "content": log_entry["user_input"]
            })
            messages.append({
                "role": "assistant",
                "content": str(log_entry["agent_response"])
            })

        # Add current user input
        messages.append({"role": "user", "content": user_text})

        # Get response from GPT-4
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,  # Large buffer for complete JSON with corrections and explanations
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

async def generate_speech(text: str, voice: str = "nova") -> bytes:
    """
    Generate speech audio using OpenAI TTS API

    Args:
        text: German text to speak
        voice: OpenAI voice (alloy, echo, fable, onyx, nova, shimmer)

    Returns:
        bytes: MP3 audio data
    """
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        response = await client.audio.speech.create(
            model="tts-1",  # Use tts-1-hd for higher quality (2x cost)
            voice=voice,
            input=text,
            response_format="mp3",  # mp3 for compatibility
            speed=0.9  # Slightly slower for language learning
        )

        # Read the audio bytes from the response
        audio_bytes = response.content

        logger.info(f"Generated TTS audio: {len(audio_bytes)} bytes for {len(text)} chars")
        return audio_bytes

    except Exception as e:
        logger.error(f"TTS generation error: {e}", exc_info=True)
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

@app.get("/api/test-tts")
async def test_tts():
    """Test TTS generation - debug endpoint"""
    try:
        test_text = "Hallo, wie geht es dir?"
        logger.info(f"Testing TTS with: {test_text}")
        audio_bytes = await generate_speech(test_text, voice="nova")
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        logger.info(f"TTS successful: {len(audio_bytes)} bytes")
        return {
            "success": True,
            "text": test_text,
            "audio_size": len(audio_bytes),
            "base64_size": len(audio_base64),
            "sample": audio_base64[:100]
        }
    except Exception as e:
        logger.error(f"TTS test failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

# ============================================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/sessions")
async def list_sessions():
    """List all available sessions"""
    sessions = []
    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "filename": session_file.name,
                    "start_time": data["start_time"],
                    "duration_minutes": data.get("duration_minutes", 0),
                    "exchanges": data.get("exchanges", 0),
                    "level": data.get("learner_level", "A1")
                })
        except Exception as e:
            logger.warning(f"Error reading session {session_file}: {e}")
            continue

    # Sort by start time (newest first)
    sessions.sort(key=lambda x: x["start_time"], reverse=True)
    return {"sessions": sessions}

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full session details"""
    # Find session file by ID
    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data["session_id"] == session_id:
                    return data
        except Exception as e:
            logger.warning(f"Error reading session {session_file}: {e}")
            continue

    return {"error": "Session not found"}, 404

@app.delete("/api/sessions/{filename}")
async def delete_session(filename: str):
    """Delete a session file"""
    import re

    # Validate filename format
    if not re.match(r"^\d{8}_\d{6}\.json$", filename):
        return {"error": "Invalid filename format"}, 400

    # Prevent directory traversal
    if ".." in filename or "/" in filename:
        return {"error": "Invalid filename"}, 400

    session_file = SESSIONS_DIR / filename

    if not session_file.exists():
        return {"error": "Session not found"}, 404

    session_file.unlink()
    logger.info(f"Deleted session: {filename}")

    return {"success": True, "deleted": filename}

@app.post("/api/sessions/new")
async def create_new_session():
    """Signal that a new session should start (returns session metadata)"""
    profile = load_learner_profile()
    session_start = datetime.now()

    return {
        "session_id": f"session_{session_start.isoformat()}",
        "start_time": session_start.isoformat(),
        "learner_level": profile["current_level"],
        "session_number": profile["session_count"] + 1
    }

# ============================================================================
# SERVE FRONTEND (AFTER API ROUTES)
# ============================================================================

# Serve frontend - use a root route instead of mount to avoid catching WebSocket
frontend_path = Path(__file__).parent.parent / "frontend"

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML"""
    html_file = frontend_path / "index.html"
    if html_file.exists():
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"message": "Frontend not found"}

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

                        # Generate TTS audio for AI response
                        agent_german_text = response_data.get("continue_german", "")
                        tts_base64 = None

                        try:
                            tts_audio_bytes = await generate_speech(agent_german_text, voice="nova")
                            tts_base64 = base64.b64encode(tts_audio_bytes).decode('utf-8')
                        except Exception as e:
                            logger.warning(f"TTS failed: {e}, continuing without audio")

                        # Log this exchange
                        session_log.append({
                            "timestamp": datetime.now().isoformat(),
                            "user_input": user_german,
                            "user_corrected": response_data.get("corrected_german"),
                            "user_translation": response_data.get("english_translation"),
                            "corrections": response_data.get("corrections", []),
                            "agent_response": agent_german_text,
                            "pronunciation": response_data.get("pronunciation_assessment", {}),
                            "had_tts": tts_base64 is not None
                        })

                        # Auto-save every N exchanges to prevent data loss
                        if len(session_log) % AUTO_SAVE_INTERVAL == 0:
                            session_file = SESSIONS_DIR / f"{session_start.strftime('%Y%m%d_%H%M%S')}.json"
                            save_session_checkpoint(session_start, session_log, profile, session_file)

                        # Send enhanced conversation response
                        await websocket.send_json({
                            "type": "conversation",
                            "timestamp": datetime.now().isoformat(),
                            "user_message": {
                                "original_german": user_german,
                                "corrected_german": response_data.get("corrected_german"),
                                "english_translation": response_data.get("english_translation"),
                                "corrections": response_data.get("corrections", []),
                                "pronunciation": response_data.get("pronunciation_assessment", {})
                            },
                            "agent_message": {
                                "german_text": agent_german_text,
                                "english_translation": "",
                                "tts_audio": tts_base64
                            }
                        })

                    except Exception as e:
                        error_str = str(e)

                        # Log failed exchange to maintain history integrity
                        session_log.append({
                            "timestamp": datetime.now().isoformat(),
                            "user_input": user_german if 'user_german' in locals() else None,
                            "user_corrected": None,
                            "user_translation": None,
                            "corrections": [],
                            "agent_response": None,
                            "pronunciation": {},
                            "had_tts": False,
                            "error": error_str
                        })

                        # Check for audio quality error
                        if "AUDIO_QUALITY_ERROR" in error_str or "Audio quality too poor" in error_str:
                            logger.warning(f"Audio quality error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "error_category": "audio_quality",
                                "message": "Untertitelung aufgrund der Audioqualität nicht möglich. / Subtitling not possible due to audio quality. Try speaking closer to the microphone or in a quieter environment."
                            })
                        else:
                            logger.error(f"Audio processing error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Error processing audio: {str(e)}"
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
        # Save session and update profile with error handling
        try:
            # Calculate duration
            session_duration = (datetime.now() - session_start).total_seconds() / 60

            # Update profile FIRST (before session save for consistency)
            profile["session_count"] += 1
            profile["total_minutes"] += round(session_duration, 2)
            profile["last_session"] = datetime.now().isoformat()
            save_learner_profile(profile)
            logger.info("Profile updated successfully")

            # Build session data
            session_data = {
                "session_id": f"session_{session_start.isoformat()}",
                "start_time": session_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_minutes": round(session_duration, 2),
                "learner_level": profile["current_level"],
                "exchanges": len(session_log),
                "conversation_log": session_log
            }

            # Only save if session has exchanges
            if len(session_log) > 0:
                session_file = SESSIONS_DIR / f"{session_start.strftime('%Y%m%d_%H%M%S')}.json"
                with open(session_file, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Session saved: {session_file} ({len(session_log)} exchanges)")
            else:
                logger.info("Empty session - not saved to disk")

        except Exception as e:
            logger.error(f"Error saving session or profile: {e}", exc_info=True)
            # Session ends even if save fails

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
