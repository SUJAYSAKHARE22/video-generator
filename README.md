# OpenVid AI — Autonomous Cinematic Demo Engine

Turns a raw screen-recording into a cinematic SaaS-style promo video, fully automatically. An AI vision model (NVIDIA NIM, free tier) looks at your video and decides everything a human would manually configure in an editor like OpenVid — background, device mockup, zoom/pan camera moves, captions — then a Python/MoviePy renderer applies it.

## Project structure

```
video-generator/
├── app.py              # Flask server (upload -> AI plan -> render -> download)
├── ai_planner.py        # Calls NVIDIA NIM vision model, returns edit plan JSON
├── renderer.py           # MoviePy/PIL rendering engine (background, mockup, zoom)
├── requirements.txt
├── .env.example          # Copy to .env and add your NVIDIA API key
├── templates/
│   └── index.html        # Upload UI
├── uploads/               # Temp uploaded videos (auto-created)
├── output/                # Rendered output videos (auto-created)
└── assets/                # Optional: background.mp4 / cursor.png / music.mp3
```

## 1. Extract the files

- Unzip `video-generator-ai.zip` — this is the complete, ready-to-run project. Use this instead of copy-pasting the individual files I sent separately (those were just for you to preview each file's content).
- If you already have your old `video-generator-main` folder, replace `app.py` and `requirements.txt`, and add `ai_planner.py`, `renderer.py`, `.env.example` into it, then replace `templates/index.html`.

## 2. Install Python dependencies

Requires Python 3.10+.

```bash
cd video-generator
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` installs: Flask, MoviePy, NumPy, Pillow, OpenCV, Requests, python-dotenv.

If MoviePy complains about FFmpeg missing, install it:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
# Windows: download from ffmpeg.org and add to PATH
```

## 3. Get a free NVIDIA NIM API key

1. Go to https://build.nvidia.com
2. Sign in / create a free account
3. Open any model page (e.g. search "llama-3.2-11b-vision-instruct")
4. Click **Get API Key** — this gives you a free key starting with `nvapi-...`

## 4. Configure your API key

```bash
cp .env.example .env
```

Edit `.env`:
```
NVIDIA_API_KEY=nvapi-your-real-key-here
```

> If you skip this step, the app still works — it just falls back to a sensible default edit plan instead of an AI-generated one.

## 5. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## 6. Use it

1. Drag & drop (or browse) your raw screen-recording (MP4/MOV/WebM, up to 200MB).
2. Click **"Let AI Direct & Generate Video"**.
3. Wait — the AI samples frames from your video, decides background/mockup/zoom/caption, and MoviePy renders the final 1920×1080 cinematic video.
4. Download the finished `.mp4` when it's ready.

## Optional: static assets

Drop these into `assets/` if you want them used as fallbacks/extras:
- `background.mp4` — looping background video (otherwise a gradient is generated)
- `cursor.png` — custom cursor icon
- `music.mp3` — background music track

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` inside your activated venv |
| MoviePy/FFmpeg error | Install ffmpeg system-wide (see step 2) |
| AI plan always falls back to default | Check `.env` has a valid `NVIDIA_API_KEY` and you have internet access |
| 500 error on render | Check the terminal running `app.py` — it prints the exact exception |
| Upload fails / too large | File exceeds 200MB limit (`MAX_CONTENT_LENGTH` in `app.py`) — raise it if needed |

## How it works (short version)

1. `app.py` receives the uploaded video, probes its duration.
2. `ai_planner.py` extracts a few sample frames, sends them + a JSON-schema prompt to NVIDIA's free `meta/llama-3.2-11b-vision-instruct` vision model, and parses/validates the returned edit plan (background, mockup style, zoom fragments, caption).
3. `renderer.py` builds a static "shell" (gradient background + browser mockup frame with URL bar), then for every video frame computes the zoom/pan state from the plan, crops/zooms the screen content, composites it into the shell with rounded corners, and optionally draws a caption.
4. The final video is rendered with MoviePy/FFmpeg and sent back for download.
