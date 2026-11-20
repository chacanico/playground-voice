"""Relay one Playground's push-to-talk turns to an OpenAI Realtime session."""
import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import os
import re
import time
from urllib.parse import urlencode

from websocket import create_connection

MAX_RECORDING = 80000
MAX_REPLY = 240000


class RealtimeError(RuntimeError):
    """Report only machine-readable fields, never audio, transcripts or secrets."""

    def __init__(self, category, details):
        fields = [category]
        for key in ("type", "code", "reason", "param"):
            value = details.get(key)
            if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", value):
                fields.append(f"{key}={value}")
        super().__init__(" ".join(fields))


class Conversation:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.socket = None
        self.started = 0

    def send(self, event, **payload):
        self.socket.send(json.dumps({"type": event, **payload}))

    def receive(self):
        message = json.loads(self.socket.recv())
        if message["type"] == "error":
            raise RealtimeError("Realtime event rejected", message.get("error") or {})
        return message

    def close(self):
        if self.socket:
            self.socket.close(timeout=1)
            self.socket = None

    def ask(self, audio):
        if not self.socket or time.monotonic() - self.started >= 45 * 60:
            self.close()
            query = urlencode({"model": self.model})
            self.socket = create_connection(
                f"wss://api.openai.com/v1/realtime?{query}", timeout=20,
                header={"Authorization": f"Bearer {self.api_key}"}
            )
            self.started = time.monotonic()
            self.send("session.update", session={
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": "You are Playground's voice assistant. Reply in the user's language, in one or two short sentences.",
                "audio": {
                    "input": {"format": {"type": "audio/pcmu"}, "turn_detection": None},
                    "output": {"format": {"type": "audio/pcmu"}, "voice": "marin"},
                },
            })
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                self.socket.settimeout(max(0.1, deadline - time.monotonic()))
                if self.receive()["type"] == "session.updated":
                    break
            else:
                raise TimeoutError("Realtime session setup timed out")

        self.send("input_audio_buffer.append", audio=base64.b64encode(audio).decode("ascii"))
        self.send("input_audio_buffer.commit")
        self.send("response.create")
        reply = bytearray()
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            self.socket.settimeout(max(0.1, deadline - time.monotonic()))
            event = self.receive()
            if event["type"] == "response.output_audio.delta":
                reply.extend(base64.b64decode(event["delta"], validate=True))
                if len(reply) > MAX_REPLY:
                    raise RuntimeError("Reply exceeds audio limit")
            elif event["type"] == "response.done":
                if event["response"]["status"] == "completed" and reply:
                    return reply
                details = event["response"].get("status_details") or {}
                raise RealtimeError("Realtime response incomplete", {
                    **details,
                    **(details.get("error") or {}),
                })
        raise TimeoutError("Realtime response timed out")


class VoiceHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def do_POST(self):
        if self.path != "/voice":
            self.send_error(404)
            return
        token = self.headers.get("X-Playground-Token", "").encode()
        if not hmac.compare_digest(token, self.server.token):
            self.send_error(401)
            return
        try:
            count = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            count = 0
        if self.headers.get_content_type() != "audio/basic" or not 800 <= count <= MAX_RECORDING:
            self.send_error(400)
            return
        try:
            audio = self.rfile.read(count)
            if len(audio) != count:
                self.send_error(400)
                return
            reply = self.server.conversation.ask(audio)
        except Exception as error:
            self.server.conversation.close()
            self.log_error("Voice request failed (%s)", type(error).__name__)
            self.send_error(502)
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/basic")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True, type=ipaddress.IPv4Address)
    parser.add_argument("--port", default=8766, type=int)
    parser.add_argument("--model", default="gpt-realtime-2.1-mini")
    options = parser.parse_args()
    token = os.environ.get("PLAYGROUND_VOICE_TOKEN", "").encode()
    api_key = os.environ.get("OPENAI_API_KEY")
    if len(token) < 24 or not api_key:
        parser.error("Set OPENAI_API_KEY and a PLAYGROUND_VOICE_TOKEN of at least 24 bytes")
    if not options.bind.is_private or options.bind.is_unspecified:
        parser.error("Bind to a specific private address")

    conversation = Conversation(api_key, options.model)
    with HTTPServer((str(options.bind), options.port), VoiceHandler) as server:
        server.token = token
        server.conversation = conversation
        print(f"Voice relay: http://{options.bind}:{options.port}/voice", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            conversation.close()


if __name__ == "__main__":
    main()
