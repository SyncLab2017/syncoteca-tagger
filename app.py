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


def _detect_vocal_gender(y: np.ndarray, sr: int,
                         f0_female: float = 180.0,
                         f0_male: float = 140.0) -> dict:
    """Estimate vocal gender via F0 autocorrelation on voiced frames.
    Voiced frames: ZCR in 0.03–0.15 (voice range) AND RMS > 40th-percentile.
    Returns gender, median F0, and voiced_ratio."""
    n_fft = 2048
    hop = n_fft // 4
    frames_y = [y[i:i + n_fft] for i in range(0, len(y) - n_fft, hop)]
    if len(frames_y) < 10:
        return {"gender": "unclear", "f0_median": 0.0, "voiced_ratio": 0.0}

    zcr_frames = np.array([np.mean(np.abs(np.diff(np.sign(f))) / 2) for f in frames_y])
    rms_frames = np.array([np.sqrt(np.mean(f ** 2)) for f in frames_y])
    rms_thresh = float(np.percentile(rms_frames, 40))

    voiced = (zcr_frames > 0.03) & (zcr_frames < 0.15) & (rms_frames > rms_thresh)
    voiced_ratio = float(voiced.sum() / len(voiced))

    if voiced.sum() < 10:
        return {"gender": "instrumental", "f0_median": 0.0, "voiced_ratio": round(voiced_ratio, 3)}

    pitches = []
    for i in np.where(voiced)[0][:200]:
        f = frames_y[i]
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        lo = max(1, int(sr / 400))
        hi = min(len(ac) - 1, int(sr / 80))
        if hi > lo:
            p = int(np.argmax(ac[lo:hi + 1])) + lo
            pitches.append(sr / p)

    if not pitches:
        return {"gender": "unclear", "f0_median": 0.0, "voiced_ratio": round(voiced_ratio, 3)}

    f0 = float(np.median(pitches))
    if f0 > f0_female:
        gender = "female"
    elif f0 < f0_male:
        gender = "male"
    else:
        gender = "unclear"

    return {
        "gender":       gender,
        "f0_median":    round(f0, 1),
        "voiced_ratio": round(voiced_ratio, 3),
    }


def _spectral_features(y: np.ndarray, sr: int) -> dict:
    """ZCR, spectral centroid, rolloff, 3-band energy (bass/mid/high)."""
    zcr = float(np.mean(np.abs(np.diff(np.sign(y))) / 2))

    n_fft = 2048
    hop = n_fft // 4
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    frames = [y[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(y) - n_fft, hop)]

    if not frames:
        return {"zcr": round(zcr, 4), "centroid": sr // 4, "rolloff": sr // 4,
                "bass_energy": 0.33, "mid_energy": 0.33, "high_energy": 0.33}

    mags = np.abs(np.array([np.fft.rfft(f) for f in frames]))
    mag_sum = mags.sum(axis=1, keepdims=True) + 1e-10

    centroid = float(np.mean((mags * freqs).sum(axis=1) / mag_sum.squeeze()))

    cum = np.cumsum(mags, axis=1)
    total = cum[:, -1:] + 1e-10
    rolloff_idx = np.argmax(cum / total >= 0.85, axis=1)
    rolloff = float(np.mean(freqs[rolloff_idx.clip(0, len(freqs) - 1)]))

    bass_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = freqs >= 4000
    total_e = mags.mean() + 1e-10

    return {
        "zcr": round(zcr, 4),
        "centroid": int(centroid),
        "rolloff": int(rolloff),
        "bass_energy": round(float(mags[:, bass_mask].mean() / total_e), 3),
        "mid_energy":  round(float(mags[:, mid_mask].mean()  / total_e), 3),
        "high_energy": round(float(mags[:, high_mask].mean() / total_e), 3),
    }


def _estimate_bpm_scipy(y: np.ndarray, sr: int,
                        bpm_min: float = 55.0, bpm_max: float = 200.0,
                        anti_double: float = 140.0) -> float:
    """BPM via energy-onset autocorrelation. Anti-doubling: if BPM > anti_double,
    check half-period; if it's strong (>50% of main peak) use half BPM instead."""
    hop = 512
    n_hops = (len(y) - hop) // hop
    if n_hops < 8:
        return 120.0
    energy = np.array([np.mean(y[i * hop:(i + 1) * hop] ** 2) for i in range(n_hops)])
    onset = np.maximum(0, np.diff(energy))
    ac = np.correlate(onset, onset, mode="full")
    ac = ac[len(ac) // 2:]
    fps = sr / hop
    lo = max(1, int(fps * 60 / bpm_max))
    hi = min(len(ac) - 1, int(fps * 60 / bpm_min))
    if hi <= lo:
        return 120.0
    peak = int(np.argmax(ac[lo:hi + 1])) + lo
    bpm = fps * 60.0 / peak
    # Anti-doubling: check if half-BPM period is also a strong peak
    if bpm > anti_double:
        half_peak = peak * 2
        if half_peak < len(ac) - 1:
            if ac[half_peak] / (ac[peak] + 1e-10) > 0.5:
                bpm = bpm / 2
    return float(np.clip(bpm, bpm_min, bpm_max))


def _detect_key(y: np.ndarray, sr: int) -> str:
    """Key via chroma + Krumhansl-Schmuckler profiles (no essentia required)."""
    chroma = _chroma_from_stft(y, sr)
    chroma = chroma / (chroma.sum() + 1e-10)
    maj = _MAJ_PROFILE / _MAJ_PROFILE.sum()
    min_ = _MIN_PROFILE / _MIN_PROFILE.sum()
    best_score, best_key, best_mode = -np.inf, "C", "мажор"
    for i in range(12):
        rc = np.roll(chroma, -i)
        for profile, mode in ((maj, "мажор"), (min_, "минор")):
            score = float(np.corrcoef(rc, profile)[0, 1])
            if score > best_score:
                best_score, best_key, best_mode = score, KEYS[i], mode
    return f"{best_key} {best_mode}"


def analyze_audio(file_path: str, tuning: dict | None = None) -> dict:
    t = tuning or {}
    try:
        TARGET_SR = 44100
        bpm_min = float(t.get("bpm_min", 55))
        bpm_max = float(t.get("bpm_max", 200))
        anti_double = float(t.get("bpm_anti_double", 140))
        energy_baseline = float(t.get("energy_baseline", 0.3))

        if _ESSENTIA_OK:
            loader = MonoLoader(filename=file_path, sampleRate=TARGET_SR)
            y = loader()
            y = y[:TARGET_SR * 60]
            if len(y) < TARGET_SR * 5:
                return {"error": "слишком короткий файл"}
            rhythm = RhythmExtractor2013(method="multifeature")
            bpm, ticks, _, _, _ = rhythm(y)
            bpm = float(bpm)
            dance = float(np.clip(
                1.0 - np.diff(ticks.astype(float)).std() / (np.diff(ticks.astype(float)).mean() + 1e-6),
                0.0, 1.0,
            )) if len(ticks) > 4 else 0.5
            key_ext = KeyExtractor()
            key_name, scale, _ = key_ext(y)
            key = f"{key_name} {'мажор' if scale == 'major' else 'минор'}"
        else:
            import soundfile as sf
            y, sr = sf.read(file_path, always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y.astype(np.float32)
            if sr != TARGET_SR:
                n_target = int(len(y) * TARGET_SR / sr)
                y = np.interp(np.linspace(0, len(y) - 1, n_target), np.arange(len(y)), y)
            y = y[:TARGET_SR * 60]
            if len(y) < TARGET_SR * 5:
                return {"error": "слишком короткий файл"}
            bpm = _estimate_bpm_scipy(y, TARGET_SR, bpm_min, bpm_max, anti_double)
            dance = 0.5
            key = _detect_key(y, TARGET_SR)

        rms = float(np.sqrt(np.mean(y ** 2)))
        energy = float(np.clip(rms / energy_baseline, 0.0, 1.0))
        vocal_presence = _hpss_ratio(y, TARGET_SR)
        spec = _spectral_features(y, TARGET_SR)
        gender_info = _detect_vocal_gender(
            y, TARGET_SR,
            f0_female=float(t.get("f0_female", 180.0)),
            f0_male=float(t.get("f0_male", 140.0)),
        )

        return {
            "bpm":            round(bpm, 1),
            "key":            key,
            "energy":         round(energy, 3),
            "vocal_presence": round(vocal_presence, 3),
            "danceability":   round(dance, 3),
            "audio_analyzed": True,
            "error":          None,
            **spec,
            **gender_info,
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

_FALLBACK_EXAMPLES = [
    {"title": "Sail On", "artist": "T-Bone Walker",
     "genre": ["Funk","Soul"], "mood": ["Positive","Rhythmic","Energetic","Fun","Swagger"],
     "era": ["1970s"], "tempo": ["Fast"], "vocal": ["Male vocal"],
     "instr": ["Drums","Bass guitar","Electric guitar","Synth","Percussion"], "theme": ["Passion"]},
    {"title": "Crime in Action", "artist": "Paolo Vivaldi",
     "genre": ["Classical","World"], "mood": ["Dramatic","Intense","Epic"],
     "era": ["2010s"], "tempo": ["Fast"], "vocal": ["Instrumental"],
     "instr": ["Orchestra","Strings","Drums"], "theme": []},
    {"title": "Fake Luv", "artist": "Rozalia",
     "genre": ["Pop","Electronic"], "mood": ["Energetic","Sexy","Swagger"],
     "era": ["2020s"], "tempo": ["Midtempo"], "vocal": ["Female vocal"],
     "instr": ["Synth","Bass guitar"], "theme": []},
    {"title": "Мария Магдалена", "artist": "Филипп Киркоров",
     "genre": ["Pop-Rock","Estrada (Soviet pop)"], "mood": ["Epic","Dramatic","Romantic"],
     "era": ["1990s"], "tempo": ["Midtempo"], "vocal": ["Male vocal","Background vocals"],
     "instr": [], "theme": []},
]


@st.cache_data(ttl=3600, show_spinner=False)
def _load_catalog_examples() -> list[dict]:
    """Load fully-tagged human examples from Supabase for dynamic few-shot selection."""
    sb = _supabase()
    if not sb:
        return _FALLBACK_EXAMPLES
    try:
        resp = (sb.table("human_tags")
                .select("title,artist,genre,mood,era,tempo,vocal,instr,theme")
                .neq("employee", "auto")
                .neq("genre", "[]")
                .neq("vocal", "[]")
                .neq("era", "[]")
                .neq("mood", "[]")
                .limit(600)
                .execute())
        rows = resp.data or []
        result = []
        for r in rows:
            try:
                ex = {
                    "title":  r.get("title", ""),
                    "artist": r.get("artist", ""),
                }
                for cat in ("genre", "mood", "era", "tempo", "vocal", "instr", "theme"):
                    val = r.get(cat, "[]")
                    parsed = json.loads(val) if isinstance(val, str) else (val or [])
                    ex[cat] = [t["en"] for t in parsed if isinstance(t, dict) and "en" in t]
                if ex["genre"] and ex["vocal"]:
                    result.append(ex)
            except Exception:
                pass
        return result if result else _FALLBACK_EXAMPLES
    except Exception:
        return _FALLBACK_EXAMPLES


def _select_examples(audio_metrics: dict, catalog: list[dict], n: int = 5,
                     artist: str = "") -> list[dict]:
    """Pick n catalog examples relevant to this track's audio profile.

    Priority order:
    1. Same artist (artist-specific learning — same artist → similar tags)
    2. Same vocal gender + tempo bucket
    3. Same vocal gender
    4. Anything else
    """
    import random
    if len(catalog) <= n:
        return catalog

    gender = audio_metrics.get("gender", "unclear")
    tempo_label = ("Fast" if audio_metrics.get("bpm", 0) > 130
                   else "Slow" if audio_metrics.get("bpm", 0) < 80
                   else "Midtempo")

    def vocal_en(ex):
        return (ex.get("vocal") or [""])[0]

    # Tier 0: same artist (case-insensitive, up to 2 slots reserved)
    artist_clean = (artist or "").strip().lower()
    artist_examples = []
    if artist_clean and artist_clean not in ("—", "-", ""):
        artist_examples = [e for e in catalog
                           if (e.get("artist") or "").strip().lower() == artist_clean]
    artist_slots = min(2, len(artist_examples))
    artist_picks = artist_examples[:artist_slots]
    artist_keys  = {(e["title"], e["artist"]) for e in artist_picks}

    # Remaining slots from gender/tempo matching (exclude artist picks)
    remaining_n = n - artist_slots
    rest_catalog = [e for e in catalog if (e["title"], e["artist"]) not in artist_keys]

    target_vocal = ("Female vocal" if gender == "female"
                    else "Male vocal" if gender == "male"
                    else "Instrumental" if gender == "instrumental"
                    else None)

    if target_vocal:
        matching = [e for e in rest_catalog if vocal_en(e) == target_vocal]
        other    = [e for e in rest_catalog if vocal_en(e) != target_vocal]
    else:
        matching = rest_catalog
        other    = []

    tempo_match = [e for e in matching if tempo_label in (e.get("tempo") or [])]
    tempo_other = [e for e in matching if tempo_label not in (e.get("tempo") or [])]

    pool = tempo_match + tempo_other + other
    seen, deduped = set(), []
    for e in pool:
        key = (e["title"], e["artist"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    best  = deduped[:max(0, remaining_n - 1)]
    rest2 = deduped[max(0, remaining_n - 1):]
    extra = [random.choice(rest2)] if rest2 else []

    return artist_picks + best + extra


def _format_examples(examples: list[dict], target_artist: str = "") -> str:
    lines = ["EXAMPLES FROM OUR CATALOG — match this tagging style exactly:"]
    artist_clean = (target_artist or "").strip().lower()
    for ex in examples:
        tag = {cat: ex.get(cat, []) for cat in ("genre","mood","era","tempo","vocal","instr","theme")}
        ex_artist = (ex.get("artist") or "").strip()
        same = (artist_clean and ex_artist.lower() == artist_clean)
        label = " ⭐ SAME ARTIST — highest priority" if same else ""
        lines.append(f'\nTrack: "{ex["title"]}" by {ex_artist}{label}')
        lines.append(json.dumps(tag, ensure_ascii=False))
    return "\n".join(lines)


def _audio_context(m: dict, tuning: dict | None = None) -> str:
    if not m.get("audio_analyzed"):
        return ""
    t = tuning or {}
    energy_label = "высокая" if m["energy"] > 0.6 else "средняя" if m["energy"] > 0.3 else "низкая"

    vp = m.get("vocal_presence", 0.5)
    zcr = m.get("zcr", 0.05)
    vocal_hi  = float(t.get("vocal_hi",       0.50))
    vocal_lo  = float(t.get("vocal_lo",       0.15))
    zcr_vocal = float(t.get("zcr_vocal_hi",   0.07))
    if vp > vocal_hi or (vp > vocal_lo and zcr > zcr_vocal):
        vocal_label = "вокальный"
    elif vp < vocal_lo and zcr < zcr_vocal * 0.7:
        vocal_label = "инструментальный"
    else:
        vocal_label = "смешанный/неясно"

    # Vocal presence: HPSS+ZCR determines if vocal EXISTS; F0 is a weak fallback.
    # F0 detection on mixed audio is unreliable for gender (synths/instruments
    # produce pitch in vocal range). Gender must come from Claude's artist knowledge.
    f0 = m.get("f0_median", 0.0)
    vr = m.get("voiced_ratio", 0.0)
    gender_raw = m.get("gender", "unclear")

    # Strict instrumental: HPSS/ZCR says no vocal AND very few voiced frames
    if vocal_label == "инструментальный" and vr < 0.10:
        gender_hint = "instrumental — no vocal signal"
    elif gender_raw == "instrumental" and vr < 0.03:
        gender_hint = "instrumental — voiced_ratio near zero"
    else:
        # Vocal is present (or uncertain) — report audio metrics only.
        # Claude determines gender from ARTIST NAME, not F0.
        # F0 provided as last-resort hint for completely unknown artists.
        f0_note = f", F0_estimate={f0:.0f}Hz" if f0 > 50 else ""
        strength = "strong" if vocal_label == "вокальный" else "moderate"
        gender_hint = f"vocal present ({strength} signal, HPSS={vp:.2f}, voiced_ratio={vr:.2f}{f0_note})"

    ctx = (
        f"Audio: BPM={m['bpm']}, key={m['key']}, "
        f"energy={m['energy']:.2f}({energy_label}), "
        f"vocal_hpss={vp:.2f} zcr={zcr:.3f} → {vocal_label}, "
        f"voice={gender_hint} voiced_ratio={vr:.2f}, "
        f"danceability={m['danceability']:.2f}"
    )
    if m.get("centroid"):
        brightness = "яркий" if m["centroid"] > 3000 else "тёплый" if m["centroid"] < 1500 else "нейтральный"
        ctx += (f", centroid={m['centroid']}Hz({brightness})"
                f", bass/mid/high={m.get('bass_energy',0):.2f}/"
                f"{m.get('mid_energy',0):.2f}/{m.get('high_energy',0):.2f}")
    return ctx


def enrich_with_claude(
    title: str,
    artist: str,
    audio_metrics: dict,
    api_key: str,
    vocab: dict,
    tuning: dict | None = None,
    catalog_examples: list | None = None,
) -> dict:
    """Returns {"genre":[{en,ru},...], "mood":[...], "era":[...], "tempo":[...], "vocal":[...], "instr":[...], "theme":[...]}"""
    t = tuning or {}
    client = anthropic.Anthropic(api_key=api_key)

    genre_list  = ", ".join(sorted(vocab["genre"].keys()))
    mood_list   = ", ".join(sorted(vocab["mood"].keys()))
    era_list    = ", ".join(vocab["era"].keys())
    tempo_list  = ", ".join(vocab["tempo"].keys())
    vocal_list  = ", ".join(sorted(vocab["vocal"].keys()))
    instr_list  = ", ".join(sorted(vocab["instr"].keys()))
    theme_list  = ", ".join(sorted(vocab["theme"].keys()))

    audio_ctx = _audio_context(audio_metrics, t)

    n_genre = t.get("n_genre", "1-3")
    n_mood  = t.get("n_mood",  "3-7")
    n_instr = t.get("n_instr", "2-6")
    n_theme = t.get("n_theme", "0-3")
    n_examples = int(t.get("n_examples", 5))
    temperature = float(t.get("claude_temp", 0.2))

    # Dynamic few-shot from catalog
    cat = catalog_examples or _FALLBACK_EXAMPLES
    selected = _select_examples(audio_metrics, cat, n=n_examples, artist=artist)
    few_shot_block = _format_examples(selected, target_artist=artist)

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
- genre: {n_genre} tags
- mood: {n_mood} tags that best describe the emotional feel
- era: 1 tag (decade the track sounds like, not release year)
- tempo: 1 tag
- vocal: 1-2 tags — STRICT RULES: (1) "Instrumental" and any gender tag are MUTUALLY EXCLUSIVE. (2) if voice=instrumental* → ["Instrumental"] ONLY. (3) if voice=vocal present* → determine gender in PRIORITY ORDER: FIRST your knowledge of the ARTIST NAME (most reliable — you know artists like Jane Air, Ягода, GAFT, Techcrasher, EMMA M etc.); SECOND track title context; THIRD F0_estimate as last resort (>180Hz female, <140Hz male). (4) DEFAULT TO VOCAL — most tracks have singers; only choose Instrumental when voice=instrumental in audio data.
- instr: {n_instr} prominent instruments (empty array if unclear)
- theme: {n_theme} lyric themes (empty array if instrumental or unclear)

{few_shot_block}
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
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            result = {
                "genre": _map(parsed.get("genre", []), "genre"),
                "mood":  _map(parsed.get("mood",  []), "mood"),
                "era":   _map(parsed.get("era",   []), "era"),
                "tempo": _map(parsed.get("tempo", []), "tempo"),
                "vocal": _map(parsed.get("vocal", []), "vocal"),
                "instr": _map(parsed.get("instr", []), "instr"),
                "theme": _map(parsed.get("theme", []), "theme"),
            }
            if t.get("debug_claude"):
                result["_claude_raw"] = raw
            return result
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
    tuning: dict | None = None,
    catalog: list | None = None,
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
        # yt-dlp result takes priority; manual title/artist used as fallback
        row["title"] = info.get("title") or info.get("track") or title or "—"
        row["artist"] = info.get("artist") or info.get("uploader") or artist or "—"
        if dl_error and not audio_path:
            row["status"] = f"⚠️ аудио: {dl_error}"
    elif audio_file is not None:
        ext = Path(audio_file.name).suffix
        audio_bytes = audio_file.getvalue()
        row["_audio_bytes"] = audio_bytes
        row["_audio_mime"] = {".mp3": "audio/mpeg", ".wav": "audio/wav",
                               ".flac": "audio/flac", ".ogg": "audio/ogg",
                               ".m4a": "audio/mp4"}.get(ext.lower(), "audio/mpeg")
        audio_path = f"{tmp_dir}/uploaded{ext}"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

    if audio_path and Path(audio_path).exists():
        metrics = analyze_audio(audio_path, tuning)
        if not metrics.get("error"):
            row.update({
                "bpm":            metrics["bpm"],
                "key":            metrics["key"],
                "energy":         metrics["energy"],
                "vocal_presence": metrics["vocal_presence"],
                "danceability":   metrics["danceability"],
                "zcr":            metrics.get("zcr", ""),
                "centroid":       metrics.get("centroid", ""),
                "rolloff":        metrics.get("rolloff", ""),
                "bass_energy":    metrics.get("bass_energy", ""),
                "mid_energy":     metrics.get("mid_energy", ""),
                "high_energy":    metrics.get("high_energy", ""),
                "gender":         metrics.get("gender", ""),
                "f0_median":      metrics.get("f0_median", ""),
                "voiced_ratio":   metrics.get("voiced_ratio", ""),
                "audio_analyzed": True,
            })
            row["_audio_metrics"] = metrics  # cached for re-tag without re-analysis
    else:
        metrics = {"audio_analyzed": False}

    if api_key and vocab:
        tags = enrich_with_claude(row["title"], row["artist"], metrics, api_key, vocab, tuning, catalog)
        if "error" not in tags:
            row.update(tags)
            row["status"] = "✅"
        else:
            row["status"] = "⚠️ " + tags.get("error", "")
    else:
        row["status"] = "⚠️ нет API ключа"

    return row


# ─── Filename parsing ─────────────────────────────────────────────────────────

def _parse_filename(name: str) -> tuple[str, str]:
    """Extract (title, artist) from 'Artist_-_Title.mp3' style filenames."""
    stem = Path(name).stem
    stem = re.sub(r'\([^)]*\)', '', stem)   # remove (TheMP3.Info) etc.
    stem = re.sub(r'\[[^\]]*\]', '', stem)  # remove [320kbps] etc.
    stem = stem.replace('_', ' ').strip()
    parts = re.split(r'\s*-\s*', stem, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[1].strip(), parts[0].strip()  # title, artist
    return stem, ""


# ─── Display helpers ──────────────────────────────────────────────────────────

import streamlit.components.v1 as components


def _tags_text(tags: list[dict]) -> str:
    return " · ".join(t['ru'] for t in tags) if tags else "—"


def _tag_labels(tags: list[dict]) -> list[str]:
    return [f"{t['ru']} · {t['en']}" for t in tags]


def _labels_to_tags(labels: list[str], vocab_cat: dict) -> list[dict]:
    result = []
    for label in (labels or []):
        parts = label.split(" · ", 1)
        if len(parts) == 2:
            tag = vocab_cat.get(parts[1].strip())
            if tag:
                result.append(tag)
    return result


def _yandex_player(url: str):
    tid = re.search(r"/track/(\d+)", url)
    aid = re.search(r"/album/(\d+)", url)
    if not tid:
        return
    src = f"https://music.yandex.ru/iframe/#track/{tid.group(1)}"
    if aid:
        src += f"/{aid.group(1)}"
    components.html(
        f'<iframe style="border:none;width:100%;height:88px;border-radius:8px" '
        f'src="{src}" allow="autoplay"></iframe>', height=94,
    )


def _correction_form(row: dict, idx: int, results_key: str):
    """Inline tag correction form with tabs. Saves to session_state + Supabase."""
    k = f"{results_key}_{idx}"

    g_all  = sorted(_tag_labels(list(vocab["genre"].values())))
    m_all  = sorted(_tag_labels(list(vocab["mood"].values())))
    e_all  = _tag_labels(list(vocab["era"].values()))
    t_all  = _tag_labels(list(vocab["tempo"].values()))
    v_all  = sorted(_tag_labels(list(vocab["vocal"].values())))
    i_all  = sorted(_tag_labels(list(vocab["instr"].values())))
    th_all = sorted(_tag_labels(list(vocab["theme"].values())))

    tab_g, tab_m, tab_et, tab_vi, tab_th = st.tabs([
        "🎸 Жанр", "😊 Настроение", "📅 Эпоха + Темп",
        "🎤 Вокал + Инструменты", "📝 Темы",
    ])

    with tab_g:
        new_g = st.pills("Жанр", g_all, default=_tag_labels(row.get("genre", [])),
                         selection_mode="multi", key=f"cg_{k}")
    with tab_m:
        new_m = st.pills("Настроение", m_all, default=_tag_labels(row.get("mood", [])),
                         selection_mode="multi", key=f"cm_{k}")
    with tab_et:
        new_e = st.pills("Эпоха", e_all, default=_tag_labels(row.get("era", [])),
                         selection_mode="multi", key=f"ce_{k}")
        st.divider()
        new_t = st.pills("Темп", t_all, default=_tag_labels(row.get("tempo", [])),
                         selection_mode="multi", key=f"ct_{k}")
    with tab_vi:
        new_v = st.pills("Вокал", v_all, default=_tag_labels(row.get("vocal", [])),
                         selection_mode="multi", key=f"cv_{k}")
        st.divider()
        new_i = st.pills("Инструменты", i_all, default=_tag_labels(row.get("instr", [])),
                         selection_mode="multi", key=f"ci_{k}")
    with tab_th:
        new_th = st.pills("Темы текста", th_all, default=_tag_labels(row.get("theme", [])),
                          selection_mode="multi", key=f"cth_{k}")

    if st.button("💾 Сохранить исправления", type="primary", key=f"save_{k}"):
        corrected = {
            "genre": _labels_to_tags(st.session_state.get(f"cg_{k}", []),  vocab["genre"]),
            "mood":  _labels_to_tags(st.session_state.get(f"cm_{k}", []),  vocab["mood"]),
            "era":   _labels_to_tags(st.session_state.get(f"ce_{k}", []),  vocab["era"]),
            "tempo": _labels_to_tags(st.session_state.get(f"ct_{k}", []),  vocab["tempo"]),
            "vocal": _labels_to_tags(st.session_state.get(f"cv_{k}", []),  vocab["vocal"]),
            "instr": _labels_to_tags(st.session_state.get(f"ci_{k}", []),  vocab["instr"]),
            "theme": _labels_to_tags(st.session_state.get(f"cth_{k}", []), vocab["theme"]),
        }
        st.session_state[results_key][idx].update(corrected)
        err = save_to_supabase({**st.session_state[results_key][idx], "notes": "corrected"})
        if err:
            st.warning(f"Supabase: {err}")
        else:
            st.success("Сохранено!")
        st.rerun()


def _render_track_card(row: dict, idx: int = 0, results_key: str = ""):
    if row.get("source", "").startswith("http") and "yandex" in row.get("source", ""):
        _yandex_player(row["source"])

    # Audio player for uploaded files
    if row.get("_audio_bytes"):
        st.audio(row["_audio_bytes"], format=row.get("_audio_mime", "audio/mpeg"))

    st.markdown(f"**{row['title']}** — {row['artist']}")
    if row.get("bpm"):
        gender_str = ""
        if row.get("gender") in ("female", "male"):
            g_ru = "♀ жен." if row["gender"] == "female" else "♂ муж."
            gender_str = f" · Голос: {g_ru} F0={row.get('f0_median','')}Hz"
        st.caption(f"BPM: {row['bpm']} · {row.get('key', '')} · Энергия: {row.get('energy', '')}{gender_str}")

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

    tuning = st.session_state.get("tuning", {})
    if tuning.get("debug_metrics") and row.get("audio_analyzed"):
        with st.expander("🔬 Сырые аудио-метрики"):
            debug_cols = ["bpm", "key", "energy", "vocal_presence", "zcr",
                          "gender", "f0_median", "voiced_ratio",
                          "centroid", "rolloff", "bass_energy", "mid_energy", "high_energy", "danceability"]
            st.json({k: row.get(k, "—") for k in debug_cols})
    if tuning.get("debug_claude") and row.get("_claude_raw"):
        with st.expander("🤖 Ответ Claude (raw JSON)"):
            st.code(row["_claude_raw"], language="json")

    # Re-tag button: re-runs only Claude with current tuning (audio metrics cached)
    if results_key and row.get("_audio_metrics"):
        k = f"{results_key}_{idx}"
        if st.button("🔄 Перетегировать с текущими настройками", key=f"retag_{k}"):
            ak = st.session_state.get("api_key", "")
            cat = st.session_state.get("catalog_examples", _FALLBACK_EXAMPLES)
            if ak:
                with st.spinner("Перетегирую..."):
                    new_tags = enrich_with_claude(
                        row["title"], row["artist"],
                        row["_audio_metrics"], ak, vocab, tuning, cat,
                    )
                if "error" not in new_tags:
                    st.session_state[results_key][idx].update(new_tags)
                    st.session_state[results_key][idx]["status"] = "✅"
                    st.rerun()
                else:
                    st.error(new_tags["error"])

    if results_key:
        with st.expander("✏️ Исправить теги"):
            _correction_form(row, idx, results_key)


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

# Load catalog examples for dynamic few-shot (cached 1h)
# Must load after _supabase() is available (it uses @st.cache_resource)
if "catalog_examples" not in st.session_state:
    with st.spinner("Загружаю каталог для few-shot..."):
        st.session_state["catalog_examples"] = _load_catalog_examples()
_catalog_size = len(st.session_state["catalog_examples"])

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

    with st.expander("🎚 Пульт настройки алгоритмов", expanded=False):
        st.markdown("**BPM**")
        bpm_min      = st.slider("BPM мин",              40,  100,  55, key="s_bpm_min")
        bpm_max      = st.slider("BPM макс",             100, 250, 200, key="s_bpm_max")
        anti_double  = st.slider("Антиудвоение: порог",  80,  200, 140, key="s_anti_double",
                                 help="Если BPM > порога — проверить, не является ли это удвоением")

        st.divider()
        st.markdown("**Энергия**")
        energy_base  = st.slider("RMS-норма (0.3 = громкий трек)", 0.05, 0.80, 0.30,
                                 step=0.05, key="s_energy_base")

        st.divider()
        st.markdown("**Вокал (аудио-сигнал)**")
        vocal_hi     = st.slider("HPSS порог → вокальный",         0.10, 1.00, 0.50,
                                 step=0.05, key="s_vocal_hi",
                                 help="Выше этого значения = вокал есть")
        vocal_lo     = st.slider("HPSS порог → инструментальный",  0.00, 0.50, 0.15,
                                 step=0.05, key="s_vocal_lo",
                                 help="Ниже этого значения + низкий ZCR = инструментал")
        zcr_vocal    = st.slider("ZCR порог вокала",                0.01, 0.20, 0.07,
                                 step=0.01, key="s_zcr_vocal",
                                 help="ZCR (zero-crossing rate) речи/вокала обычно 0.04–0.12")
        st.markdown("**Определение пола голоса (F0)**")
        f0_female    = st.slider("F0 порог: женский голос (Гц)",  150, 280, 180,
                                 key="s_f0_female",
                                 help="Женский вокал обычно 160–300 Гц; снизь если пропускает сопрано")
        f0_male      = st.slider("F0 порог: мужской голос (Гц)",   80, 180, 140,
                                 key="s_f0_male",
                                 help="Мужской вокал обычно 80–150 Гц; подними если путает баритон с женским")

        st.divider()
        st.markdown("**Claude + Каталог**")
        n_examples   = st.slider("Примеров из каталога на трек", 2, 10, 5,
                                 key="s_n_examples",
                                 help="Сколько похожих треков из ваших 600 показывать Claude как образец")
        claude_temp  = st.slider("Температура (0=точный, 1=творческий)", 0.0, 1.0, 0.2,
                                 step=0.1, key="s_claude_temp")
        n_genre      = st.select_slider("Жанров",      ["1", "1-2", "1-3", "2-4"], value="1-3", key="s_n_genre")
        n_mood       = st.select_slider("Настроений",  ["2-4", "3-5", "3-7", "4-8"], value="3-7", key="s_n_mood")
        n_instr      = st.select_slider("Инструментов",["1-3", "2-4", "2-6", "3-8"], value="2-6", key="s_n_instr")
        n_theme      = st.select_slider("Тем",         ["0", "0-2", "0-3", "1-4"],   value="0-3", key="s_n_theme")

        st.divider()
        st.markdown("**Отладка**")
        debug_metrics = st.checkbox("Показать аудио-метрики под треком", key="s_debug_metrics")
        debug_claude  = st.checkbox("Показать JSON от Claude",           key="s_debug_claude")

    # Write tuning dict into session_state — read by _run_batch + _render_track_card
    current_tuning = {
        "bpm_min":         bpm_min,
        "bpm_max":         bpm_max,
        "bpm_anti_double": anti_double,
        "energy_baseline": energy_base,
        "vocal_hi":        vocal_hi,
        "vocal_lo":        vocal_lo,
        "zcr_vocal_hi":    zcr_vocal,
        "f0_female":       f0_female,
        "f0_male":         f0_male,
        "claude_temp":     claude_temp,
        "n_examples":      n_examples,
        "n_genre":         n_genre,
        "n_mood":          n_mood,
        "n_instr":         n_instr,
        "n_theme":         n_theme,
        "debug_metrics":   debug_metrics,
        "debug_claude":    debug_claude,
    }
    st.session_state["tuning"] = current_tuning
    st.session_state["api_key"] = api_key  # accessible from _render_track_card

    st.divider()

    # ─── Presets ──────────────────────────────────────────────────────────────
    if "presets" not in st.session_state:
        st.session_state["presets"] = {}

    with st.expander("🗂 Пресеты настроек", expanded=False):
        preset_name = st.text_input("Название пресета", placeholder="Например: инди-женский")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Сохранить", key="preset_save"):
                if preset_name.strip():
                    st.session_state["presets"][preset_name.strip()] = dict(current_tuning)
                    st.success(f"Сохранён: {preset_name.strip()}")
        with c2:
            preset_json = json.dumps(st.session_state["presets"], ensure_ascii=False, indent=2)
            st.download_button("⬇️ Экспорт", preset_json.encode("utf-8"),
                               "syncoteca_presets.json", "application/json", key="preset_export")

        if st.session_state["presets"]:
            st.divider()
            selected = st.selectbox("Загрузить пресет", list(st.session_state["presets"].keys()),
                                    key="preset_select")
            c3, c4 = st.columns(2)
            with c3:
                if st.button("📂 Загрузить", key="preset_load"):
                    p = st.session_state["presets"][selected]
                    _key_map = {
                        "bpm_min": "s_bpm_min", "bpm_max": "s_bpm_max",
                        "bpm_anti_double": "s_anti_double", "energy_baseline": "s_energy_base",
                        "vocal_hi": "s_vocal_hi", "vocal_lo": "s_vocal_lo",
                        "zcr_vocal_hi": "s_zcr_vocal", "f0_female": "s_f0_female",
                        "f0_male": "s_f0_male", "claude_temp": "s_claude_temp",
                        "n_genre": "s_n_genre", "n_mood": "s_n_mood",
                        "n_instr": "s_n_instr", "n_theme": "s_n_theme",
                    }
                    for tk, sk in _key_map.items():
                        if tk in p:
                            st.session_state[sk] = p[tk]
                    st.rerun()
            with c4:
                if st.button("🗑 Удалить", key="preset_delete"):
                    del st.session_state["presets"][selected]
                    st.rerun()

        uploaded_presets = st.file_uploader("⬆️ Импорт JSON", type="json", key="preset_import")
        if uploaded_presets:
            try:
                imported = json.loads(uploaded_presets.getvalue())
                st.session_state["presets"].update(imported)
                st.success(f"Импортировано: {len(imported)} пресетов")
            except Exception as e:
                st.error(f"Ошибка: {e}")

    st.divider()
    st.caption("Форматы аудио: MP3, WAV, FLAC, OGG")
    st.caption("URL: Яндекс.Музыка, SoundCloud, YouTube")
    st.divider()
    st.caption(f"Жанров: {len(vocab['genre'])} · Настроений: {len(vocab['mood'])} · Тем: {len(vocab['theme'])}")
    st.caption(f"📚 Каталог: {_catalog_size} треков-образцов")


tab_url, tab_upload = st.tabs(["🔗 Яндекс.Музыка URL", "📁 Загрузить файлы"])


def _run_batch(sources: list, audio_files: list = None, fallback_title: str = "", fallback_artist: str = ""):
    """Run tagging for a batch, yield results one by one."""
    tuning   = st.session_state.get("tuning", {})
    catalog  = st.session_state.get("catalog_examples", _FALLBACK_EXAMPLES)
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
                    tuning=tuning, catalog=catalog,
                )
            else:
                url = item
                label = url[:60]
                row = process_track(
                    source=url,
                    title=fallback_title,
                    artist=fallback_artist,
                    api_key=api_key, tmp_dir=tmp, vocab=vocab,
                    tuning=tuning, catalog=catalog,
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
        height=120,
    )

    col_t, col_a = st.columns(2)
    with col_t:
        url_title = st.text_input("Название трека", placeholder="Например: Sail On", key="url_title")
    with col_a:
        url_artist = st.text_input("Артист", placeholder="Например: T-Bone Walker", key="url_artist")
    st.caption("Если Яндекс.Музыка не отдаёт метаданные — введи название и артиста вручную")

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
            results = _run_batch(urls, fallback_title=url_title.strip(), fallback_artist=url_artist.strip())
            st.session_state["results_url"] = results

    if st.session_state.get("results_url"):
        results = st.session_state["results_url"]

        st.divider()
        st.subheader("Результаты")
        for i, r in enumerate(results):
            with st.expander(f"{r['status']} {r['title']} — {r['artist']}"):
                _render_track_card(r, idx=i, results_key="results_url")

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
        with st.expander("➕ Метаданные (автопарсинг из имени файла, можно изменить)"):
            for uf in uploaded_files:
                auto_title, auto_artist = _parse_filename(uf.name)
                c1, c2 = st.columns(2)
                with c1:
                    t = st.text_input("Название", key=f"t_{uf.name}", value=auto_title)
                with c2:
                    a = st.text_input("Артист", key=f"a_{uf.name}", value=auto_artist)
                meta_rows.append((uf, t, a))
        if not meta_rows:
            meta_rows = [(uf, *_parse_filename(uf.name)) for uf in uploaded_files]

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
        for i, r in enumerate(results):
            with st.expander(f"{r['status']} {r['title']} — {r['artist']}"):
                _render_track_card(r, idx=i, results_key="results_upload")

        csv_bytes = _to_csv_export(results)
        st.download_button(
            "⬇️ Скачать CSV (human_tags формат)",
            csv_bytes,
            "syncoteca_auto_tags.csv",
            "text/csv",
            key="dl_upload",
        )
