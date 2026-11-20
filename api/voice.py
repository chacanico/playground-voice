"""Vercel endpoint for independent Playground voice turns."""
import hmac
from http.server import BaseHTTPRequestHandler
import os

from tools.voice_relay import Conversation, MAX_RECORDING, RealtimeError


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b"Playground voice relay")

    def do_POST(self):
        token = os.environ.get("PLAYGROUND_VOICE_TOKEN", "").encode()
        key = os.environ.get("OPENAI_API_KEY")
        if len(token) < 24 or not key:
            self.send_error(503)
            return

        provided = self.headers.get("X-Playground-Token", "").encode()
        if not hmac.compare_digest(provided, token):
            self.send_error(401)
            return

        try:
            size = int(self.headers.get("Content-Length", ""))
        except ValueError:
            size = 0
        if not 800 <= size <= MAX_RECORDING or self.headers.get_content_type() != "audio/basic":
            self.send_error(400)
            return

        audio = self.rfile.read(size)
        if len(audio) != size:
            self.send_error(400)
            return

        conversation = Conversation(key, "gpt-realtime-2.1-mini")
        try:
            reply = conversation.ask(audio)
        except Exception as error:
            detail = str(error) if isinstance(error, RealtimeError) else type(error).__name__
            self.log_error("Voice request failed (%s)", detail)
            self.send_error(502)
            return
        finally:
            conversation.close()

        self.send_response(200)
        self.send_header("Content-Type", "audio/basic")
        self.send_header("Content-Length", str(len(reply)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(reply)
