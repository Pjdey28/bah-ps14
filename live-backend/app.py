from __future__ import annotations

import os
import threading
import time

import eventlet

eventlet.monkey_patch()

from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from stream_source import SolarStormReplaySource


STREAM_PORT = int(os.getenv("SOLAR_STREAM_PORT", "5000"))
STREAM_REFRESH_SECONDS = float(os.getenv("SOLAR_STREAM_REFRESH_SECONDS", "2.0"))


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

source = SolarStormReplaySource()
state = source.snapshot()

stream_lock = threading.Lock()
stream_started = False


def to_snapshot():
    return state


def next_packet():
    global state
    with stream_lock:
        packet = source.next_packet()
        state = source.snapshot()
        return packet


def stream_loop():
    while True:
        socketio.emit("packet", next_packet())
        time.sleep(STREAM_REFRESH_SECONDS)


def ensure_stream():
    global stream_started
    if stream_started:
        return

    stream_started = True
    threading.Thread(target=stream_loop, daemon=True).start()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": state.get("mode", "historical"), "rows": state.get("rows", 0)})


@app.route("/snapshot", methods=["GET"])
def snapshot():
    return jsonify(to_snapshot())


@socketio.on("connect")
def on_connect():
    ensure_stream()
    emit("snapshot", to_snapshot())


if __name__ == "__main__":
    ensure_stream()
    socketio.run(app, host="0.0.0.0", port=STREAM_PORT, debug=True)