import json
import re
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_from_directory

app = Flask(__name__)

OUTPUTS = Path(__file__).parent.parent / "outputs"
VALID_YTID = re.compile(r"^[\w-]{1,32}$")
VALID_MODELS = {"sd-turbo", "sdxl-turbo"}


def _valid_ytid(ytid: str) -> bool:
    return bool(VALID_YTID.match(ytid))


def _available_clips() -> list[str]:
    ytids = []
    for prompt_file in sorted((OUTPUTS / "prompts").glob("*.json")):
        ytid = prompt_file.stem
        if (
            (OUTPUTS / "images-sd-turbo" / f"{ytid}.png").exists()
            and (OUTPUTS / "images-sdxl-turbo" / f"{ytid}.png").exists()
        ):
            ytids.append(ytid)
    return ytids


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/clips")
def api_clips():
    return jsonify(_available_clips())


@app.get("/api/clip/<ytid>")
def api_clip(ytid: str):
    if not _valid_ytid(ytid):
        abort(400)
    path = OUTPUTS / "prompts" / f"{ytid}.json"
    if not path.exists():
        abort(404)
    return jsonify(json.loads(path.read_text()))


@app.get("/audio/<ytid>.wav")
def serve_audio(ytid: str):
    if not _valid_ytid(ytid):
        abort(400)
    return send_from_directory(OUTPUTS / "audio", f"{ytid}.wav")


@app.get("/image/<model>/<ytid>.png")
def serve_image(model: str, ytid: str):
    if model not in VALID_MODELS or not _valid_ytid(ytid):
        abort(400)
    return send_from_directory(OUTPUTS / f"images-{model}", f"{ytid}.png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
