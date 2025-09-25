#!/usr/bin/env python3
"""
Parakeet WebSocket Server - AssemblyAI Compatible API

Provides the exact same WebSocket API format as AssemblyAI using NVIDIA Parakeet.
"""

import asyncio
import json
import time
import threading
from collections import deque
from typing import Dict, Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol
from dotenv import load_dotenv
import os
import numpy as np
import tempfile as _session_tmp
from silero_vad import load_silero_vad

# Import model once and reuse
import nemo.collections.asr as nemo_asr
import torch


class ParakeetWebSocketServer:
    """WebSocket server providing AssemblyAI-compatible API with Parakeet."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        model_name: str = "nvidia/parakeet-tdt_ctc-1.1b"
    ):
        self.host = host
        self.port = port
        self.model_name = model_name
        
        # Connected clients
        self.clients = set()
        
        # Shared model
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # PnC handled by the selected ASR model (built-in). No external PnC pipeline.
        
        # Turn tracking
        self.current_turn_order = 0
        
        print(f"🚀 Initializing Parakeet WebSocket Server")
        print(f"📡 Server will run on ws://{host}:{port}")
    
    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register a new client connection."""
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"👤 Client connected: {client_info}")
        print(f"📊 Total clients: {len(self.clients)}")
        
        # Send welcome message
        welcome_msg = {
            "type": "connection_established",
            "message": "Connected to Parakeet ASR WebSocket Server",
            "model": self.model_name,
            "timestamp": int(time.time() * 1000)
        }
        await websocket.send(json.dumps(welcome_msg))
    
    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister a client connection."""
        if websocket in self.clients:
            self.clients.remove(websocket)
            client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            print(f"👋 Client disconnected: {client_info}")
            print(f"📊 Total clients: {len(self.clients)}")

        # Optionally unload model when no clients
        if not self.clients and self.model is not None:
            print("🛑 No clients connected. Model stays loaded (fast re-connect).")
    
    def format_text_for_final(self, text: str) -> tuple[str, bool]:
        """Model emits PnC; return as-is and mark applied."""
        if not text:
            return text, False
        return text.strip(), True
    
    async def send_transcription(self, websocket: WebSocketServerProtocol, result: Dict[str, Any]):
        """Send transcription to a single client - AssemblyAI compatible format."""
        # AssemblyAI-compatible response format
        message = {
            "type": "transcription",
            "text": result["text"],
            "is_final": result["is_final"],
            "isTurnFormatted": result["is_final"],
            "source": "mic",  # Default to mic, could be enhanced to track source
            "turnOrder": result.get("segment_id", 0),
            "confidence": 1.0 if result["is_final"] else 0.8,
        }
        
        if result["is_final"]:
            formatted_text, applied = self.format_text_for_final(result["text"])
            message["text"] = formatted_text
            message["isPnCApplied"] = applied
            print(
                f"📝 Final turn={message['turnOrder']} PnC={applied} before_len={len(result['text'])} after_len={len(formatted_text)}"
            )
        else:
            message["isPnCApplied"] = False
        await websocket.send(json.dumps(message))
    
    class ConnectionSession:
        """Per-connection session: buffers, VAD-lite, and turn tracking."""
        def __init__(self, websocket: WebSocketServerProtocol, server: "ParakeetWebSocketServer"):
            self.websocket = websocket
            self.server = server
            self.sample_rate = 16000
            self.auth_data = None  # Store AssemblyAI authentication data
            # Adaptive noise-resistant settings
            self.chunk_min_samples = int(1.0 * self.sample_rate)   # 1.0s to first interim (capture beginning)
            self.min_speech_duration = int(0.5 * self.sample_rate)  # 0.5s minimum speech before finalization
            self.interim_count = 0  # Track number of interim results for accumulating context
            final_silence_duration = float(os.getenv('FINAL_SILENCE_DURATION', '2.0'))  # Increased to 2.0s for more tolerance
            max_silence_duration = float(os.getenv('MAX_SILENCE_DURATION', '4.0'))  # Increased to 4.0s for longer pauses
            self.final_silence_samples = int(final_silence_duration * self.sample_rate)
            self.max_silence_samples = int(max_silence_duration * self.sample_rate)
            
            # Hybrid VAD System: Silero + Energy Analysis
            self.silero_vad = load_silero_vad()
            self.vad_confidence_threshold = 0.7  # Higher threshold to reduce false positives from office noise
            
            # Energy-based user proximity detection
            self.energy_history = deque(maxlen=100)  # Track ambient energy
            self.ambient_energy = 0.0
            self.energy_threshold_multiplier = float(os.getenv('ENERGY_THRESHOLD_MULTIPLIER', '2.5'))
            self.min_energy_threshold = float(os.getenv('MIN_ENERGY_THRESHOLD', '50.0'))
            self.max_energy_threshold = float(os.getenv('MAX_ENERGY_THRESHOLD', '500.0'))
            
            self.buffer = deque(maxlen=self.sample_rate * 5)  # 5s ring
            self.speech_buffer = []
            self.in_speech = False
            self.silence_count = 0
            self.segment_id = 0
            # Debug settings
            self.debug_hybrid_vad = os.getenv('DEBUG_HYBRID_VAD', 'false').lower() == 'true'

            # Reusable temp file (on fast drive) to avoid churn
            tmp = _session_tmp.NamedTemporaryFile(suffix=".wav", delete=False)
            self._temp_wav = tmp.name
            tmp.close()

            # Worker control
            self._active = True
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

        def append_audio(self, pcm_bytes: bytes):
            audio = np.frombuffer(pcm_bytes, dtype=np.int16)
            self.buffer.extend(audio)

        def pop_block(self, block_size: int) -> Optional[np.ndarray]:
            if len(self.buffer) < block_size:
                return None
            block = [self.buffer.popleft() for _ in range(block_size)]
            return np.array(block, dtype=np.int16)

        def calculate_energy(self, audio: np.ndarray) -> float:
            """Calculate RMS energy of audio block"""
            if audio.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        
        def update_ambient_energy(self, energy: float):
            """Update ambient energy baseline for adaptive thresholding"""
            self.energy_history.append(energy)
            if len(self.energy_history) >= 10:  # Need some history
                self.ambient_energy = float(np.median(list(self.energy_history)))
        
        def get_adaptive_threshold(self) -> float:
            """Get dynamic energy threshold based on ambient noise"""
            if self.ambient_energy <= 0:
                return self.min_energy_threshold
            
            # Calculate threshold as multiple of ambient energy
            threshold = self.ambient_energy * self.energy_threshold_multiplier
            
            # Clamp to reasonable bounds
            threshold = max(self.min_energy_threshold, min(threshold, self.max_energy_threshold))
            
            # Safety reset if threshold gets too high (indicates mic issues)
            if threshold > 400:
                self.ambient_energy = 0.0
                self.energy_history.clear()
                threshold = self.min_energy_threshold
            
            return threshold
        
        def is_primary_user_speech(self, audio: np.ndarray) -> bool:
            """Check if speech is from the primary user (close, loud) using energy analysis"""
            if audio.size == 0:
                return False
            
            # Calculate energy
            energy = self.calculate_energy(audio)
            
            # Update ambient energy baseline
            self.update_ambient_energy(energy)
            
            # Get adaptive threshold
            threshold = self.get_adaptive_threshold()
            
            # Check if energy indicates primary user (close/loud)
            is_primary = energy > threshold
            return is_primary
        
        
        def is_speech(self, audio: np.ndarray) -> bool:
            """Hybrid VAD: Combine Silero (any speech) + Energy (primary user)"""
            if audio.size == 0:
                return False
            
            try:
                # Step 1: Silero VAD - detects ANY speech (user + others)
                audio_float = audio.astype(np.float32) / 32768.0
                if len(audio_float) < 160:  # Need at least 10ms at 16kHz
                    return False
                
                audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)
                speech_prob = self.silero_vad(audio_tensor, 16000)
                silero_speech = speech_prob > self.vad_confidence_threshold
                
                # Step 2: Check if it's from the PRIMARY user (close, loud)
                is_primary_user = self.is_primary_user_speech(audio)
                
                # Step 3: Hybrid decision logic
                # Only transcribe if: Silero detects speech AND Energy indicates primary user
                hybrid_decision = bool(silero_speech.item()) and is_primary_user
                
                return hybrid_decision
            except Exception as e:
                print(f"❌ Hybrid VAD error: {e}")
                return False


        def is_sentence_complete(self, text: str) -> bool:
            """Check if the text appears to be a complete sentence."""
            if not text or len(text.strip()) < 3:
                return False
            
            text = text.strip()
            
            # Ends with sentence-ending punctuation
            if text.endswith(('.', '!', '?')):
                return True
            
            # Check for incomplete patterns
            incomplete_patterns = [
                ' and', ' but', ' so', ' which', ' that', ' who', ' where', ' when',
                ' because', ' although', ' however', ' therefore', ' meanwhile',
                ' first', ' second', ' next', ' then', ' also', ' furthermore'
            ]
            
            text_lower = text.lower()
            for pattern in incomplete_patterns:
                if text_lower.endswith(pattern):
                    return False
            
            # If it's a reasonable length and doesn't end with incomplete patterns, consider it complete
            return len(text.split()) >= 4

        async def process(self):
            # No-op: processing is handled by the background worker to avoid event-loop stalls
            return

        def _worker_loop(self):
            import soundfile as sf
            import os as _os
            print("🔄 Worker loop started")
            while self._active:
                try:
                    block = self.pop_block(512)
                    if block is None:
                        time.sleep(0.02)
                        continue
                    
                    # Debug: Log every block processing
                    if hasattr(self, '_debug_counter'):
                        self._debug_counter += 1
                    else:
                        self._debug_counter = 0
                    
                    # Use Hybrid VAD: Silero + Energy
                    is_speech = self.is_speech(block)
                    
                    if self._debug_counter % 50 == 0 and self.debug_hybrid_vad:  # Every ~1 second
                        # Get individual VAD results for debugging
                        energy = self.calculate_energy(block)
                        threshold = self.get_adaptive_threshold()
                        is_primary = self.is_primary_user_speech(block)
                        print(f"🎤 Hybrid VAD: Speech={is_speech} | Energy={energy:.1f}/{threshold:.1f} | Primary={is_primary} | Block size: {len(block)}")
                        if self.in_speech:
                            print(f"🔇 Silence count: {self.silence_count}/{self.final_silence_samples} (final) / {self.max_silence_samples} (max)")

                    if is_speech:
                        if not self.in_speech:
                            # Start new speech segment
                            self.in_speech = True
                            self.silence_count = 0
                            self.segment_id += 1
                            self.speech_buffer = []  # Start fresh buffer
                            self._last_interim_ms = 0  # Reset interim timing
                            self._finalized = False  # Flag to prevent multiple finals
                            self.interim_count = 0  # Reset interim counter for new segment
                            # Pre-fill buffer with current block to capture beginning
                            self.speech_buffer.extend(block.tolist())
                            print(f"🎤 SPEECH STARTED - Segment {self.segment_id}")
                        
                        # Accumulate speech audio
                        self.speech_buffer.extend(block.tolist())
                        
                        # Send interim results (throttled to avoid spam)
                        if len(self.speech_buffer) >= self.chunk_min_samples:
                            now_ms = int(time.time() * 1000)
                            if now_ms - self._last_interim_ms >= 500:  # 500ms throttle
                                self._last_interim_ms = now_ms
                                self.interim_count += 1
                                
                                # Accumulating context: each interim uses progressively more audio
                                # Interim 1: 1.0s, Interim 2: 2.0s, Interim 3: 3.0s, etc.
                                context_duration = min(1.0 + (self.interim_count - 1) * 1.0, len(self.speech_buffer) / self.sample_rate)
                                context_samples = int(context_duration * self.sample_rate)
                                context_audio = self.speech_buffer[:context_samples]  # Use from beginning, not recent
                                
                                print(f"🔄 SENDING INTERIM #{self.interim_count} - Context: {context_duration:.1f}s ({len(context_audio)} samples)")
                                audio = np.array(context_audio, dtype=np.int16)
                                audio_float = audio.astype(np.float32) / 32768.0
                                try:
                                    sf.write(self._temp_wav, audio_float, self.sample_rate)
                                    from functools import partial
                                    transcribe_call = partial(
                                        self.server.model.transcribe,
                                        [self._temp_wav],
                                        batch_size=1,
                                        num_workers=0,
                                        return_hypotheses=False,
                                    )
                                    res = transcribe_call()
                                    text = ""
                                    if res and len(res) > 0:
                                        text = res[0].text if hasattr(res[0], "text") else str(res[0])
                                        text = (text or "").strip()
                                    if text:
                                        payload = {
                                            "text": text,
                                            "is_final": False,
                                            "segment_id": self.segment_id,
                                        }
                                        asyncio.run_coroutine_threadsafe(
                                            self.server.send_transcription(self.websocket, payload),
                                            self.server.loop,
                                        )
                                        print(f"⚡ INTERIM SENT: {text}")
                                except Exception:
                                    pass
                    else:
                        if self.in_speech:
                            self.silence_count += block.size
                            # Don't add silence audio to speech buffer - it corrupts the final transcription
                            
                            # Simple finalization: transcribe when silence threshold reached
                            should_finalize = False
                            
                            # Force finalization if we've hit max silence (prevent hanging)
                            if self.silence_count >= self.max_silence_samples and not self._finalized:
                                should_finalize = True
                                print(f"🔚 FORCE FINALIZING SEGMENT {self.segment_id} - Max silence reached: {self.silence_count}/{self.max_silence_samples}")
                            # Normal finalization when silence threshold reached
                            elif self.silence_count >= self.final_silence_samples and not self._finalized:
                                should_finalize = True
                            
                            if should_finalize:
                                # Check if we have enough speech content
                                if len(self.speech_buffer) >= self.min_speech_duration:
                                    self._finalized = True  # Prevent multiple finals
                                    print(f"🔚 FINALIZING SEGMENT {self.segment_id} - Silence: {self.silence_count}/{self.final_silence_samples}, Speech: {len(self.speech_buffer)} samples")
                                    
                                    # Transcribe using the same approach as interim - use all accumulated context
                                    # This ensures consistency between interim and final results
                                    audio = np.array(self.speech_buffer, dtype=np.int16)
                                    audio_float = audio.astype(np.float32) / 32768.0
                                    try:
                                        sf.write(self._temp_wav, audio_float, self.sample_rate)
                                        from functools import partial
                                        transcribe_call = partial(
                                            self.server.model.transcribe,
                                            [self._temp_wav],
                                            batch_size=1,
                                            num_workers=0,
                                            return_hypotheses=False,
                                        )
                                        res = transcribe_call()
                                        text = ""
                                        if res and len(res) > 0:
                                            text = res[0].text if hasattr(res[0], "text") else str(res[0])
                                            text = (text or "").strip()
                                        
                                        if text:
                                            print(f"✅ FINAL RESULT: {text}")
                                            print(f"🔍 AUDIO INFO - Duration: {len(self.speech_buffer)/self.sample_rate:.2f}s, Samples: {len(self.speech_buffer)}")
                                            payload = {
                                                "text": text,
                                                "is_final": True,
                                                "segment_id": self.segment_id,
                                            }
                                            asyncio.run_coroutine_threadsafe(
                                                self.server.send_transcription(self.websocket, payload),
                                                self.server.loop,
                                            )
                                        else:
                                            print(f"❌ NO TEXT GENERATED")
                                    except Exception as e:
                                        print(f"❌ TRANSCRIPTION ERROR: {e}")
                                    
                                    # Reset for next segment
                                    self.in_speech = False
                                    self.speech_buffer = []
                                    self.silence_count = 0
                                    print(f"🔄 READY FOR NEXT SEGMENT")
                                else:
                                    # Speech too short, don't finalize yet
                                    print(f"⏳ SEGMENT {self.segment_id} TOO SHORT - Speech: {len(self.speech_buffer)}/{self.min_speech_duration} samples")
                        else:
                            # No speech detected - just continue
                            pass
                except Exception:
                    # Never crash worker
                    time.sleep(0.02)

        async def emit_transcription(self, is_final: bool):
            if not self.speech_buffer:
                return
            audio = np.array(self.speech_buffer, dtype=np.int16)
            # Prepare audio for model
            # For interims, only send the last ~2.0s to preserve context
            if not is_final:
                max_interim = int(2.0 * self.sample_rate)
                if audio.size > max_interim:
                    audio = audio[-max_interim:]
            audio_float = audio.astype(np.float32) / 32768.0
            import soundfile as sf
            import tempfile
            import os as _os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_float, self.sample_rate)
                try:
                    # Offload blocking ASR to a worker thread; minimize DataLoader overhead
                    from functools import partial
                    transcribe_call = partial(
                        self.server.model.transcribe,
                        [tmp.name],
                        batch_size=1,
                        num_workers=0,
                        return_hypotheses=False,
                    )
                    res = await asyncio.to_thread(transcribe_call)
                finally:
                    try:
                        _os.unlink(tmp.name)
                    except Exception:
                        pass
            text = ""
            if res and len(res) > 0:
                text = res[0].text if hasattr(res[0], "text") else str(res[0])
                text = (text or "").strip()
            if not text:
                return
            payload = {
                "text": text,
                "is_final": is_final,
                "segment_id": self.segment_id,
            }
            await self.server.send_transcription(self.websocket, payload)
    
    async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming messages from clients - AssemblyAI compatible."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            session: ParakeetWebSocketServer.ConnectionSession = getattr(websocket, "_session", None)
            
            # AssemblyAI Authentication Flow
            if not session and (data.get("token") or data.get("tenant_id")):
                # This is an authentication message from AssemblyAI client
                print(f"🔐 AssemblyAI Authentication: {data.get('tenant_id', 'unknown')}")
                
                # Create session with authentication data
                websocket._session = ParakeetWebSocketServer.ConnectionSession(websocket, self)
                session = websocket._session
                
                # Store authentication data
                session.auth_data = {
                    'token': data.get('token'),
                    'tenant_id': data.get('tenant_id'),
                    'session_id': data.get('session_id'),
                    'meeting_id': data.get('meeting_id'),
                    'location': data.get('location'),
                    'timezone': data.get('timezone'),
                    'is_ai_intelligence_enabled': data.get('is_ai_intelligence_enabled', False)
                }
                
                # Send AssemblyAI-compatible connection confirmation
                await websocket.send(json.dumps({
                    "type": "connect",
                    "message": "Connected to Parakeet ASR service",
                    "timestamp": int(time.time() * 1000)
                }))
                return
            
            # Legacy start_transcription support
            elif msg_type == "start_transcription":
                if session is None:
                    websocket._session = ParakeetWebSocketServer.ConnectionSession(websocket, self)
                await websocket.send(json.dumps({
                    "type": "transcription_started",
                    "timestamp": int(time.time() * 1000)
                }))
            
            # AssemblyAI audio_data message
            elif msg_type == "audio_data":
                if session is None:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Not authenticated. Send authentication first.",
                        "timestamp": int(time.time() * 1000)
                    }))
                    return
                
                # Process base64 audio data
                audio_data = data.get("data", {})
                base64_audio = audio_data.get("audio_data")
                sample_rate = audio_data.get("sample_rate", 16000)
                source = data.get("source", "mic")
                
                if base64_audio:
                    try:
                        import base64
                        # Decode base64 to bytes
                        audio_bytes = base64.b64decode(base64_audio)
                        # Convert to numpy array
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        
                        # Add to session buffer
                        session.append_audio(audio_array.tobytes())
                        await session.process()
                        
                    except Exception as e:
                        print(f"❌ Error processing audio data: {e}")
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Error processing audio: {str(e)}",
                            "timestamp": int(time.time() * 1000)
                        }))
            
            elif msg_type == "stop_transcription":
                websocket._session = None
                await websocket.send(json.dumps({
                    "type": "transcription_stopped",
                    "timestamp": int(time.time() * 1000)
                }))
            
            elif msg_type == "ping":
                response = {
                    "type": "pong",
                    "timestamp": int(time.time() * 1000)
                }
                await websocket.send(json.dumps(response))
            
            else:
                response = {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "timestamp": int(time.time() * 1000)
                }
                await websocket.send(json.dumps(response))
                
        except json.JSONDecodeError:
            response = {
                "type": "error",
                "message": "Invalid JSON format",
                "timestamp": int(time.time() * 1000)
            }
            await websocket.send(json.dumps(response))
        except Exception as e:
            response = {
                "type": "error",
                "message": f"Error processing message: {str(e)}",
                "timestamp": int(time.time() * 1000)
            }
            await websocket.send(json.dumps(response))
    
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle individual client connections."""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)):
                    # Binary PCM chunk from this client
                    session: ParakeetWebSocketServer.ConnectionSession = getattr(websocket, "_session", None)
                    if session is None:
                        # Ignore audio until start_transcription received
                        continue
                    session.append_audio(message)
                    await session.process()
                else:
                    await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"❌ Client error: {e}")
        finally:
            # Stop worker and cleanup temp file
            session: ParakeetWebSocketServer.ConnectionSession = getattr(websocket, "_session", None)
            if session is not None:
                session._active = False
                try:
                    import os as _os
                    _os.unlink(session._temp_wav)
                except Exception:
                    pass
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start the WebSocket server."""
        print(f"🚀 Starting WebSocket server on {self.host}:{self.port}")
        print("⚙️  Loading ASR model...")
        # Preload ASR model
        if self.model is None:
            try:
                self.model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name
                )
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                print("✅ ASR model ready")
            except Exception as e:
                raise RuntimeError(f"ASR model load failed: {e}")
        print("🟢 Model loaded. Accepting connections...")
        
        # Store event loop for ASR thread
        self.loop = asyncio.get_event_loop()
        
        # Speed optimizations
        try:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
            torch.set_num_threads(2)
            # Prefer greedy decoding for lower latency if supported
            if hasattr(self.model, "change_decoding_strategy"):
                self.model.change_decoding_strategy({"strategy": "greedy"})
        except Exception as _:
            pass

        # Warm-up: run a tiny inference once to initialize kernels
        try:
            import tempfile as _tf
            import soundfile as _sf
            import numpy as _np
            warm = (_np.zeros(int(0.5 * 16000), dtype=_np.int16)).astype(_np.float32) / 32768.0
            with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp:
                _sf.write(_tmp.name, warm, 16000)
                try:
                    _ = await asyncio.to_thread(self.model.transcribe, [_tmp.name], batch_size=1, num_workers=0)
                finally:
                    try:
                        import os as _os
                        _os.unlink(_tmp.name)
                    except Exception:
                        pass
        except Exception as _:
            pass

        # Start WebSocket server
        server = await websockets.serve(
            # websockets 15+ no longer passes path to handler; keep compat
            lambda ws, *args, **kwargs: self.handle_client(ws),
            self.host,
            self.port,
            ping_interval=25,
            ping_timeout=25
        )
        
        print("✅ WebSocket server started!")
        print(f"📡 Connect to: ws://{self.host}:{self.port}")
        print("🎤 ASR system is running and ready for connections")
        print("🛑 Press Ctrl+C to stop")
        
        # Keep server running
        await server.wait_closed()
    
    def stop(self):
        """Stop the server and cleanup."""
        print("🛑 Stopping server...")
        # nothing persistent to cleanup here


async def main():
    """Main server function."""
    load_dotenv()
    
    # Configuration
    host = os.getenv("WEBSOCKET_HOST", "localhost")
    port = int(os.getenv("WEBSOCKET_PORT", "8080"))
    model_name = os.getenv("MODEL_NAME", "nvidia/parakeet-tdt_ctc-1.1b")
    
    # Create and start server
    server = ParakeetWebSocketServer(host=host, port=port, model_name=model_name)
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\n⚠️ Server interrupted by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        server.stop()
        print("👋 Server stopped!")


if __name__ == "__main__":
    print("🎤 NVIDIA Parakeet WebSocket Server")
    print("=" * 65)
    asyncio.run(main())
