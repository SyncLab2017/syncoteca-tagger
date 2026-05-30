import csv
import os
import json
import tempfile
import time
import io
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


def _ytdlp() -> str:
    script_dir = Path(__file__).parent
    venv_bin = script_dir / "venv" / "bin" / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    found = shutil.which("yt-dlp")
    if found:
        return found
    raise FileNotFoundError("yt-dlp not found — run: venv/bin/pip install yt-dlp")


import streamlit as st
import pandas as pd
import numpy as np
import soundfile as sf
import scipy.signal
import anthropic

try:
    from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor
    _ESSENTIA_OK = True
except Exception:
    _ESSENTIA_OK = False

st.set_page_config(
    page_title="Syncoteca Tagger",
    page_icon="🎵",
    layout="wide",
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
YANDEX_TOKEN = os.environ.get("YANDEX_MUSIC_TOKEN", "")
BASE = Path(__file__).parent

# ─── Vocabulary ───────────────────────────────────────────────────────────────

def _load_disco_cat(cat: str) -> list[dict]:
    result = []
    with open(BASE / "DISCO Tags.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            en = row.get("Name", "").strip()
            ru_col = list(row.values())[1] if len(row.values()) > 1 else ""
            ru = ru_col.strip() if ru_col else ""
            c = row.get("Category", "").strip()
            if en and c == cat:
                result.append({"en": en, "ru": ru or en})
    return result


@st.cache_data(show_spinner=False)
def _build_vocab() -> dict[str, dict[str, dict]]:
    """Returns {category: {en_key: {en,ru}}} lookup dicts."""

    def idx(tags: list[dict]) -> dict[str, dict]:
        return {t["en"]: t for t in tags}

    # Genres — same as human_tagger.py
    all_genres = [
        {"ru": "Альтернатива", "en": "Alternative"},
        {"ru": "Инди", "en": "Indie"},
        {"ru": "Метал", "en": "Metal"},
        {"ru": "Панк", "en": "Punk"},
        {"ru": "Рок", "en": "Rock"},
        {"ru": "Дэнс", "en": "Dance"},
        {"ru": "Поп", "en": "Pop"},
        {"ru": "Поп-рок", "en": "Pop-Rock"},
        {"ru": "Фанк", "en": "Funk"},
        {"ru": "Синти-поп", "en": "Synth-pop"},
        {"ru": "Хип-хоп / Рэп", "en": "Hip-hop / Rap"},
        {"ru": "Электронная", "en": "Electronic"},
        {"ru": "R&B", "en": "R&B"},
        {"ru": "Амбиент", "en": "Ambient"},
        {"ru": "Блюз", "en": "Blues"},
        {"ru": "Госпел", "en": "Gospel"},
        {"ru": "Джаз", "en": "Jazz"},
        {"ru": "Дроны", "en": "Drones"},
        {"ru": "Звуковой пейзаж", "en": "Soundscape"},
        {"ru": "Классика", "en": "Classical"},
        {"ru": "Нью-эйдж / Медитация", "en": "New Age / Meditation"},
        {"ru": "Регги", "en": "Reggae"},
        {"ru": "Рок-н-ролл", "en": "Rock'n'Roll"},
        {"ru": "Соул", "en": "Soul"},
        {"ru": "Авторская песня", "en": "Singer-songwriter"},
        {"ru": "Детская", "en": "Children's"},
        {"ru": "Диско", "en": "Disco"},
        {"ru": "Кантри", "en": "Country"},
        {"ru": "Латин", "en": "Latin"},
        {"ru": "Английская", "en": "British"},
        {"ru": "Мировая музыка", "en": "World"},
        {"ru": "Немецкая", "en": "German"},
        {"ru": "Русская", "en": "Russian"},
        {"ru": "Народная", "en": "Traditional / Folk"},
        {"ru": "Ретро", "en": "Retro"},
        {"ru": "Ретро / Винтаж", "en": "Vintage"},
        {"ru": "Романс", "en": "Romance"},
        {"ru": "Саундтрек", "en": "Soundtrack"},
        {"ru": "Фолк", "en": "Folk"},
        {"ru": "Шансон", "en": "Chanson / Russian bard"},
        {"ru": "Эстрада", "en": "Estrada (Soviet pop)"},
    ]

    # Moods — from DISCO Tags.csv + extras
    _mood_raw = _load_disco_cat("Mood/feel") + _load_disco_cat("Mood/feel / Fun")
    _extra = [
        {"ru": "Душевное",      "en": "Soulful"},
        {"ru": "Лирическое",    "en": "Lyrical"},
        {"ru": "Медитативное",  "en": "Meditative"},
        {"ru": "Меланхоличное", "en": "Melancholic"},
        {"ru": "Мрачное",       "en": "Gloomy"},
        {"ru": "Спокойное",     "en": "Calm"},
    ]
    _seen: set[str] = set()
    all_moods: list[dict] = []
    for t in _mood_raw + _extra:
        if t["en"] not in _seen:
            _seen.add(t["en"])
            all_moods.append(t)

    # Era
    all_era = [
        {"ru": "30-е", "en": "1930s"}, {"ru": "40-е", "en": "1940s"},
        {"ru": "50-е", "en": "1950s"}, {"ru": "60-е", "en": "1960s"},
        {"ru": "70-е", "en": "1970s"}, {"ru": "80-е", "en": "1980s"},
        {"ru": "90-е", "en": "1990s"}, {"ru": "2000-е", "en": "2000s"},
        {"ru": "2010-е", "en": "2010s"}, {"ru": "2020-е", "en": "2020s"},
    ]

    # Tempo — from DISCO CSV + Variable tempo (custom, used in human_tagger.py)
    all_tempo = _load_disco_cat("Tempo") + [{"ru": "Переменчивый", "en": "Variable tempo"}]

    # Vocal
    all_vocal = [
        {"ru": "Акапелла",          "en": "A cappella"},
        {"ru": "Бэк-вокал",         "en": "Background vocals"},
        {"ru": "Вокализ",           "en": "Vocalise"},
        {"ru": "Детский вокал",     "en": "Children's vocal"},
        {"ru": "Дуэт",              "en": "Duet"},
        {"ru": "Женский вокал",     "en": "Female vocal"},
        {"ru": "Инструментальная",  "en": "Instrumental"},
        {"ru": "Мужской вокал",     "en": "Male vocal"},
        {"ru": "Речитатив / Рэп",   "en": "Rap / Spoken"},
        {"ru": "Хор",               "en": "Choir"},
        {"ru": "Шёпотом",           "en": "Whisper"},
    ]

    # Instruments
    all_instr = [
        {"ru": "Акустическая гитара", "en": "Acoustic guitar"},
        {"ru": "Барабаны",            "en": "Drums"},
        {"ru": "Бас-гитара",          "en": "Bass guitar"},
        {"ru": "Виолончель",          "en": "Cello"},
        {"ru": "Духовые",             "en": "Brass / Horns"},
        {"ru": "Оркестр",             "en": "Orchestra"},
        {"ru": "Перкуссия",           "en": "Percussion"},
        {"ru": "Саксофон",            "en": "Saxophone"},
        {"ru": "Синтезатор",          "en": "Synth"},
        {"ru": "Скрипка",             "en": "Violin"},
        {"ru": "Струнные",            "en": "Strings"},
        {"ru": "Труба",               "en": "Trumpet"},
        {"ru": "Флейта",              "en": "Flute"},
        {"ru": "Фортепиано",          "en": "Piano"},
        {"ru": "Электрогитара",       "en": "Electric guitar"},
    ]

    # Themes — from DISCO Tags.csv
    _THEME_EXCLUDE = {
        "Attachment", "Believe", "Change", "Commitment", "Empowerment", "Escape",
        "Expedition", "Goal", "Hustle", "Loss", "Rebellion", "Success", "Vision", "Voyage",
    }
    _theme_raw = (
        _load_disco_cat("Lyric themes") +
        _load_disco_cat("Lyric themes / Adventure") +
        _load_disco_cat("Lyric themes / Ambition") +
        _load_disco_cat("Lyric themes / Love") +
        [
            {"ru": "Мат / Ненормативная лексика", "en": "Explicit / Profanity"},
            {"ru": "Новый год", "en": "New Year"},
            {"ru": "Ссора",     "en": "Quarrel"},
        ]
    )
    _ts: set = set()
    all_theme: list[dict] = []
    for t in _theme_raw:
        if t["en"] in _THEME_EXCLUDE:
            continue
        k = (t["ru"], t["en"])
        if k not in _ts:
            _ts.add(k)
            all_theme.append(t)

    return {
        "genre": idx(all_genres),
        "mood":  idx(all_moods),
        "era":   idx(all_era),
        "tempo": idx(all_tempo),
        "vocal": idx(all_vocal),
        "instr": idx(all_instr),
        "theme": idx(all_theme),
    }


# ─── Audio analysis ───────────────────────────────────────────────────────────

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODES = ["мажор", "минор"]

_MAJ_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _chroma_from_stft(y: np.ndarray, sr: int, n_fft: int = 4096) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    hop = n_fft // 4
    frames = [np.fft.rfft(y[i:i + n_fft] * np.hanning(n_fft))
              for i in range(0, len(y) - n_fft, hop)]
    if not frames:
        return np.ones(12) / 12.0
    mag = np.abs(np.array(frames))
    chroma = np.zeros((len(frames), 12), dtype=np.float64)
    for fi, freq in enumerate(freqs):
        if freq < 27.5:
            continue
        midi = 12 * np.log2(max(freq, 1) / 440.0) + 69
        bin_idx = int(round(midi)) % 12
        chroma[:, bin_idx] += mag[:, fi]
    total = chroma.sum(axis=1, keepdims=True) + 1e-10
    return (chroma / total).mean(axis=0)


def _hpss_ratio(y: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    hop = n_fft // 4
    frames = [y[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(y) - n_fft, hop)]
    if not frames:
        return 0.5
    S = np.abs(np.array([np.fft.rfft(f) for f in frames]))
    H = scipy.signal.medfilt(S, kernel_size=(1, 17))
    P = scipy.signal.medfilt(S, kernel_size=(17, 1))
    h_energy = float(np.mean(H ** 2))
    t_energy = float(np.mean(S ** 2)) + 1e-10
    return float(np.clip(h_energy / t_energy, 0.0, 1.0))


def analyze_audio(file_path: str) -> dict:
    try:
        if not _ESSENTIA_OK:
            return {"audio_analyzed": False, "error": "essentia not available"}

        loader = MonoLoader(filename=file_path, sampleRate=44100)
        y = loader()
        y = y[:44100 * 60]
        if len(y) < 44100 * 5:
            return {"error": "слишком короткий файл"}

        rhythm = RhythmExtractor2013(method="multifeature")
        bpm, ticks, _, _, _ = rhythm(y)
        bpm = float(bpm)

        if len(ticks) > 4:
            intervals = np.diff(ticks.astype(float))
            dance = float(np.clip(1.0 - intervals.std() / (intervals.mean() + 1e-6), 0.0, 1.0))
        else:
            dance = 0.5

        key_ext = KeyExtractor()
        key_name, scale, _ = key_ext(y)
        mode = "мажор" if scale == "major" else "минор"

        rms = float(np.sqrt(np.mean(y ** 2)))
        energy = float(np.clip(rms * 20, 0.0, 1.0))
        vocal_presence = _hpss_ratio(y, 44100)

        return {
            "bpm": round(bpm, 1),
            "key": f"{key_name} {mode}",
            "energy": round(energy, 3),
            "vocal_presence": round(vocal_presence, 3),
            "danceability": round(dance, 3),
            "audio_analyzed": True,
            "error": None,
        }
    except Exception as e:
        return {"audio_analyzed": False, "error": str(e)}


# ─── Yandex.Music download ────────────────────────────────────────────────────

def download_from_yandex(url: str, out_dir: str) -> tuple[dict, str | None, str | None]:
    ytdlp = _ytdlp()
    try:
        result = subprocess.run(
            [ytdlp, "--no-playlist", "--dump-single-json", url],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(result.stdout) if result.stdout.strip() else {}
        if not isinstance(info, dict):
            info = {}
    except Exception:
        info = {}

    audio_path = None
    dl_error = None
    try:
        dl = subprocess.run([
            ytdlp, "--no-playlist",
            "--format", "bestaudio/best",
            "-x", "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", f"{out_dir}/track.%(ext)s",
            url,
        ], capture_output=True, timeout=90)
        mp3 = Path(out_dir) / "track.mp3"
        if mp3.exists():
            audio_path = str(mp3)
        elif dl.returncode != 0:
            dl_error = dl.stderr.decode(errors="replace").strip().split("\n")[-1][:120]
    except Exception as e:
        dl_error = str(e)

    return info, audio_path, dl_error


# ─── Claude enrichment ────────────────────────────────────────────────────────

_FEW_SHOT = """EXAMPLES (use as reference for tag quality and quantity):

Track: "Sail On" by T-Bone Walker
{"genre":["Funk","Soul"],"mood":["Positive","Rhythmic","Energetic","Percussive","Fun","Swagger"],"era":["1970s"],"tempo":["Fast"],"vocal":["Male vocal"],"instr":["Drums","Bass guitar","Electric guitar","Synth","Percussion"],"theme":["Passion"]}

Track: "Crime in Action" by Paolo Vivaldi
{"genre":["Classical","World"],"mood":["Dramatic","Intense","Epic"],"era":["2010s"],"tempo":["Fast"],"vocal":["Instrumental"],"instr":["Orchestra","Strings","Drums"],"theme":[]}

Track: "Fake Luv" by Rozalia
{"genre":["Pop","Electronic"],"mood":["Energetic","Sexy","Swagger"],"era":["2020s"],"tempo":["Midtempo"],"vocal":["Female vocal"],"instr":["Synth","Bass guitar"],"theme":[]}

Track: "Мария Магдалена" by Филипп Киркоров
{"genre":["Pop-Rock","Estrada (Soviet pop)"],"mood":["Epic","Dramatic","Romantic"],"era":["1990s"],"tempo":["Midtempo"],"vocal":["Male vocal","Background vocals"],"instr":[],"theme":[]}
"""


def _audio_context(m: dict) -> str:
    if not m.get("audio_analyzed"):
        return ""
    energy_label = "высокая" if m["energy"] > 0.6 else "средняя" if m["energy"] > 0.3 else "низкая"
    vocal_label = "вокальный" if m["vocal_presence"] > 0.50 else "инструментальный" if m["vocal_presence"] < 0.15 else "смешанный"
    return (
        f"Audio: BPM={m['bpm']}, key={m['key']}, "
        f"energy={m['energy']:.2f}({energy_label}), "
        f"vocal={m['vocal_presence']:.2f}({vocal_label}), "
        f"danceability={m['danceability']:.2f}"
    )


def enrich_with_claude(
    title: str,
    artist: str,
    audio_metrics: dict,
    api_key: str,
    vocab: dict,
) -> dict:
    """Returns {"genre":[{en,ru},...], "mood":[...], "era":[...], "tempo":[...], "vocal":[...], "instr":[...], "theme":[...]}"""
    client = anthropic.Anthropic(api_key=api_key)

    genre_list  = ", ".join(sorted(vocab["genre"].keys()))
    mood_list   = ", ".join(sorted(vocab["mood"].keys()))
    era_list    = ", ".join(vocab["era"].keys())  # keep chronological order
    tempo_list  = ", ".join(vocab["tempo"].keys())
    vocal_list  = ", ".join(sorted(vocab["vocal"].keys()))
    instr_list  = ", ".join(sorted(vocab["instr"].keys()))
    theme_list  = ", ".join(sorted(vocab["theme"].keys()))

    audio_ctx = _audio_context(audio_metrics)

    prompt = f"""You are a sync music licensing expert tagging tracks for a professional catalog.
Tag this track using ONLY tags from the vocabulary below.
Return a single JSON object — no markdown, no explanation.

Track: "{title}" by {artist}
{audio_ctx}

VOCABULARY (use exact spelling):
genre: {genre_list}
mood: {mood_list}
era: {era_list}
tempo: {tempo_list}
vocal: {vocal_list}
instr: {instr_list}
theme: {theme_list}

RULES:
- genre: 1-3 tags
- mood: 3-7 tags that best describe the emotional feel
- era: 1 tag (decade the track sounds like, not release year)
- tempo: 1 tag
- vocal: 1-2 tags (use "Instrumental" if no vocals)
- instr: 2-6 prominent instruments (empty array if unclear)
- theme: 0-3 lyric themes (empty array if instrumental or unclear)

{_FEW_SHOT}
Now tag this track. Return JSON only:"""

    def _map(en_list: list, cat: str) -> list[dict]:
        result = []
        for en in en_list:
            tag = vocab[cat].get(en)
            if tag:
                result.append(tag)
        return result

    for attempt in range(3):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            return {
                "genre": _map(parsed.get("genre", []), "genre"),
                "mood":  _map(parsed.get("mood",  []), "mood"),
                "era":   _map(parsed.get("era",   []), "era"),
                "tempo": _map(parsed.get("tempo", []), "tempo"),
                "vocal": _map(parsed.get("vocal", []), "vocal"),
                "instr": _map(parsed.get("instr", []), "instr"),
                "theme": _map(parsed.get("theme", []), "theme"),
            }
        except Exception as e:
            err = str(e)
            if "529" in err or "overloaded" in err:
                time.sleep(5 * (attempt + 1))
                continue
            return {"error": err}
    return {"error": "Claude API перегружен, попробуй позже"}


# ─── Supabase save ────────────────────────────────────────────────────────────

@st.cache_resource
def _supabase():
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None


def save_to_supabase(row: dict) -> str | None:
    sb = _supabase()
    if not sb:
        return "Supabase не настроен"
    try:
        tags = {k: json.dumps(row.get(k, []), ensure_ascii=False)
                for k in ("genre", "mood", "era", "tempo", "vocal", "instr", "theme")}
        sb.table("human_tags").insert({
            "track_id": str(row.get("track_id", "")),
            "yandex_url": row.get("source", ""),
            "title": row.get("title", ""),
            "artist": row.get("artist", ""),
            "employee": "auto",
            "tagged_at": datetime.utcnow().isoformat(timespec="seconds"),
            "notes": row.get("notes", ""),
            **tags,
        }).execute()
        return None
    except Exception as e:
        return str(e)


# ─── Process single track ─────────────────────────────────────────────────────

def _extract_track_id(url: str) -> str:
    m = re.search(r"/track/(\d+)", url)
    return m.group(1) if m else ""


def process_track(
    source: str,
    audio_file=None,
    title: str = "",
    artist: str = "",
    api_key: str = "",
    tmp_dir: str = "",
    vocab: dict = None,
) -> dict:
    row = {
        "title": title or "—",
        "artist": artist or "—",
        "source": source,
        "track_id": _extract_track_id(source) if source.startswith("http") else "",
        "bpm": "",
        "key": "",
        "energy": "",
        "vocal_presence": "",
        "danceability": "",
        "audio_analyzed": False,
        "genre": [],
        "mood": [],
        "era": [],
        "tempo": [],
        "vocal": [],
        "instr": [],
        "theme": [],
        "status": "⏳",
        "notes": "",
    }

    audio_path = None

    if source.startswith("http"):
        info, audio_path, dl_error = download_from_yandex(source, tmp_dir)
        if not row["title"] or row["title"] == "—":
            row["title"] = info.get("title") or info.get("track") or title or "—"
        if not row["artist"] or row["artist"] == "—":
            row["artist"] = info.get("artist") or info.get("uploader") or artist or "—"
        if dl_error and not audio_path:
            row["status"] = f"⚠️ аудио: {dl_error}"
    elif audio_file is not None:
        ext = Path(audio_file.name).suffix
        audio_path = f"{tmp_dir}/uploaded{ext}"
        with open(audio_path, "wb") as f:
            f.write(audio_file.getvalue())

    if audio_path and Path(audio_path).exists():
        metrics = analyze_audio(audio_path)
        if not metrics.get("error"):
            row.update({
                "bpm": metrics["bpm"],
                "key": metrics["key"],
                "energy": metrics["energy"],
                "vocal_presence": metrics["vocal_presence"],
                "danceability": metrics["danceability"],
                "audio_analyzed": True,
            })
    else:
        metrics = {"audio_analyzed": False}

    if api_key and vocab:
        tags = enrich_with_claude(row["title"], row["artist"], metrics, api_key, vocab)
        if "error" not in tags:
            row.update(tags)
            row["status"] = "✅"
        else:
            row["status"] = "⚠️ " + tags.get("error", "")
    else:
        row["status"] = "⚠️ нет API ключа"

    return row


# ─── Display helpers ──────────────────────────────────────────────────────────

def _tags_text(tags: list[dict]) -> str:
    return " · ".join(f"{t['ru']}" for t in tags) if tags else "—"


def _render_track_card(row: dict):
    st.markdown(f"**{row['title']}** — {row['artist']}")
    if row.get("bpm"):
        st.caption(f"BPM: {row['bpm']} · {row.get('key', '')} · Энергия: {row.get('energy', '')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"🎸 **Жанр:** {_tags_text(row.get('genre', []))}")
        st.markdown(f"😊 **Настроение:** {_tags_text(row.get('mood', []))}")
        st.markdown(f"📅 **Эпоха:** {_tags_text(row.get('era', []))}")
        st.markdown(f"⏱ **Темп:** {_tags_text(row.get('tempo', []))}")
    with col2:
        st.markdown(f"🎤 **Вокал:** {_tags_text(row.get('vocal', []))}")
        st.markdown(f"🎹 **Инструменты:** {_tags_text(row.get('instr', []))}")
        st.markdown(f"📝 **Тема:** {_tags_text(row.get('theme', []))}")


def _results_to_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "status": r.get("status", ""),
            "title": r.get("title", ""),
            "artist": r.get("artist", ""),
            "genre": _tags_text(r.get("genre", [])),
            "mood": _tags_text(r.get("mood", [])),
            "era": _tags_text(r.get("era", [])),
            "tempo": _tags_text(r.get("tempo", [])),
            "vocal": _tags_text(r.get("vocal", [])),
            "instr": _tags_text(r.get("instr", [])),
            "theme": _tags_text(r.get("theme", [])),
            "bpm": r.get("bpm", ""),
            "key": r.get("key", ""),
            "source": r.get("source", ""),
        })
    return pd.DataFrame(rows)


def _to_csv_export(results: list[dict]) -> bytes:
    """CSV with JSON arrays (same format as human_tags table)."""
    rows = []
    for r in results:
        rows.append({
            "track_id": r.get("track_id", ""),
            "yandex_url": r.get("source", ""),
            "title": r.get("title", ""),
            "artist": r.get("artist", ""),
            "employee": "auto",
            "tagged_at": datetime.utcnow().isoformat(timespec="seconds"),
            "genre": json.dumps(r.get("genre", []), ensure_ascii=False),
            "mood": json.dumps(r.get("mood", []), ensure_ascii=False),
            "era": json.dumps(r.get("era", []), ensure_ascii=False),
            "tempo": json.dumps(r.get("tempo", []), ensure_ascii=False),
            "vocal": json.dumps(r.get("vocal", []), ensure_ascii=False),
            "instr": json.dumps(r.get("instr", []), ensure_ascii=False),
            "theme": json.dumps(r.get("theme", []), ensure_ascii=False),
            "bpm": r.get("bpm", ""),
            "key": r.get("key", ""),
            "notes": r.get("notes", ""),
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# ─── Main UI ──────────────────────────────────────────────────────────────────

vocab = _build_vocab()

st.title("🎵 Syncoteca Tagger")
st.caption("Авто-тегирование музыкального каталога для sync-лицензирования")

with st.sidebar:
    st.header("⚙️ Настройки")
    api_key = st.text_input(
        "Anthropic API Key",
        value=ANTHROPIC_API_KEY,
        type="password",
        help="sk-ant-...",
    )
    save_to_db = st.checkbox("Сохранять в Supabase", value=True,
                             help="Результаты пишутся в таблицу human_tags с employee='auto'")
    st.divider()
    st.caption("Форматы аудио: MP3, WAV, FLAC, OGG")
    st.caption("URL: Яндекс.Музыка, SoundCloud, YouTube")
    st.divider()
    st.caption(f"Жанров: {len(vocab['genre'])} · Настроений: {len(vocab['mood'])} · Тем: {len(vocab['theme'])}")


tab_url, tab_upload = st.tabs(["🔗 Яндекс.Музыка URL", "📁 Загрузить файлы"])


def _run_batch(sources: list, audio_files: list = None):
    """Run tagging for a batch, yield results one by one."""
    results = []
    progress = st.progress(0)
    status_text = st.empty()
    table_placeholder = st.empty()

    with tempfile.TemporaryDirectory() as tmp:
        for i, item in enumerate(sources):
            if audio_files:
                uf, title, artist = item
                label = uf.name
                row = process_track(
                    source=uf.name, audio_file=uf,
                    title=title or uf.name, artist=artist,
                    api_key=api_key, tmp_dir=tmp, vocab=vocab,
                )
            else:
                url = item
                label = url[:60]
                row = process_track(
                    source=url,
                    api_key=api_key, tmp_dir=tmp, vocab=vocab,
                )

            status_text.text(f"Обрабатываю {i+1}/{len(sources)}: {label}...")
            results.append(row)
            progress.progress((i + 1) / len(sources))
            table_placeholder.dataframe(_results_to_df(results), use_container_width=True)

            if save_to_db and row["status"] == "✅":
                err = save_to_supabase(row)
                if err:
                    st.warning(f"Supabase: {err}")

    status_text.text(f"✅ Готово: {len(results)} треков")
    return results


# ─── Tab 1: URLs ──────────────────────────────────────────────────────────────

with tab_url:
    st.subheader("Вставь ссылки на треки")
    urls_input = st.text_area(
        "По одной ссылке на строку",
        placeholder="https://music.yandex.ru/album/19399075/track/95265265",
        height=150,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        run_url = st.button("▶ Анализировать", type="primary", key="run_url")

    if run_url:
        urls = [u.strip() for u in urls_input.splitlines() if u.strip().startswith("http")]
        if not urls:
            st.warning("Нет валидных URL")
        elif not api_key:
            st.error("Укажи Anthropic API Key в боковой панели")
        else:
            results = _run_batch(urls)
            st.session_state["results_url"] = results

    if st.session_state.get("results_url"):
        results = st.session_state["results_url"]

        st.divider()
        st.subheader("Результаты")
        for r in results:
            with st.expander(f"{r['status']} {r['title']} — {r['artist']}"):
                _render_track_card(r)

        csv_bytes = _to_csv_export(results)
        st.download_button(
            "⬇️ Скачать CSV (human_tags формат)",
            csv_bytes,
            "syncoteca_auto_tags.csv",
            "text/csv",
            key="dl_url",
        )


# ─── Tab 2: File upload ───────────────────────────────────────────────────────

with tab_upload:
    st.subheader("Загрузи аудио-файлы")

    uploaded_files = st.file_uploader(
        "MP3 / WAV / FLAC",
        accept_multiple_files=True,
        type=["mp3", "wav", "flac", "ogg", "m4a"],
    )

    if uploaded_files:
        st.write(f"Загружено файлов: {len(uploaded_files)}")

        meta_rows = []
        with st.expander("➕ Добавить метаданные вручную (опционально)"):
            for uf in uploaded_files:
                c1, c2 = st.columns(2)
                with c1:
                    t = st.text_input("Название", key=f"t_{uf.name}", placeholder=uf.name)
                with c2:
                    a = st.text_input("Артист", key=f"a_{uf.name}")
                meta_rows.append((uf, t, a))
        if not meta_rows:
            meta_rows = [(uf, "", "") for uf in uploaded_files]

        col1, _ = st.columns([1, 3])
        with col1:
            run_upload = st.button("▶ Анализировать", type="primary", key="run_upload")

        if run_upload:
            if not api_key:
                st.error("Укажи Anthropic API Key в боковой панели")
            else:
                results = _run_batch(meta_rows, audio_files=True)
                st.session_state["results_upload"] = results

    if st.session_state.get("results_upload"):
        results = st.session_state["results_upload"]

        st.divider()
        st.subheader("Результаты")
        for r in results:
            with st.expander(f"{r['status']} {r['title']} — {r['artist']}"):
                _render_track_card(r)

        csv_bytes = _to_csv_export(results)
        st.download_button(
            "⬇️ Скачать CSV (human_tags формат)",
            csv_bytes,
            "syncoteca_auto_tags.csv",
            "text/csv",
            key="dl_upload",
        )
