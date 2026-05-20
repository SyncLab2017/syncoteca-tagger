import csv
import json
import os
import re
import random
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ─── Supabase client ──────────────────────────────────────────────────────────
@st.cache_resource
def _supabase_client():
    from supabase import create_client
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        st.error("⚠️ SUPABASE_URL / SUPABASE_KEY не настроены в Streamlit Secrets. Теги не сохраняются!")
        st.stop()
    return create_client(url, key)

st.set_page_config(page_title="Tagger — Синкотека", page_icon="🎵", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2.8rem; padding-bottom: 2rem; }
[data-testid="stPills"] button {
    font-size: 12px !important; padding: 2px 8px !important;
    touch-action: manipulation;
}
[data-testid="stPills"] { margin-bottom: 2px !important; touch-action: pan-y; }
.group-label { font-size: 11px; font-weight: 600; color: #888;
               text-transform: uppercase; letter-spacing: .06em;
               margin: 6px 0 1px 0; }
hr { margin: 4px 0 !important; }
.stTabs [data-baseweb="tab"] { font-size: 12px; padding: 3px 8px; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
section[data-testid="stMain"] { overflow-y: auto !important; }
@media (max-width: 768px) {
    .block-container { padding-top: 1rem !important; padding-bottom: 4rem; }
    [data-testid="stPills"] button { font-size: 13px !important; padding: 4px 10px !important; }
}
</style>
""", unsafe_allow_html=True)

DAILY_BATCH_SIZE = 50
BASE = Path(__file__).parent

# ─── Load DISCO Tags ──────────────────────────────────────────────────────────

def _load_disco() -> dict[str, list[dict]]:
    cats: dict[str, list[dict]] = {}
    with open(BASE / "DISCO Tags.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            en = row.get("Name", "").strip()
            ru = row.get("", "").strip()
            cat = row.get("Category", "").strip()
            if en and cat:
                cats.setdefault(cat, []).append({"en": en, "ru": ru or en})
    return cats

_D = _load_disco()

def _by_ru(tags: list[dict]) -> list[dict]:
    """Sort tags list by Russian name."""
    return sorted(tags, key=lambda t: t["ru"].lower())

def _fmt(tags: list[dict]) -> list[str]:
    return [f"{t['ru']} · {t['en']}" for t in tags]

def _d(cat: str) -> list[dict]:
    return _D.get(cat, [])

# ─── Tag groups ───────────────────────────────────────────────────────────────

# ЖАНР — 4 семейства
GENRE_GROUPS = {
    "🎸 Рок / Метал / Панк": _by_ru([
        {"ru": "Альтернатива", "en": "Alternative"},
        {"ru": "Инди", "en": "Indie"},
        {"ru": "Метал", "en": "Metal"},
        {"ru": "Панк", "en": "Punk"},
        {"ru": "Рок", "en": "Rock"},
    ]),
    "🎤 Поп / Электронная / Хип-хоп": _by_ru([
        {"ru": "Дэнс", "en": "Dance"},
        {"ru": "Поп", "en": "Pop"},
        {"ru": "Поп-рок", "en": "Pop-Rock"},
        {"ru": "Танцевальная", "en": "Danceable"},
        {"ru": "Фанк", "en": "Funk"},
        {"ru": "Хип-хоп / Рэп", "en": "Hip-hop / Rap"},
        {"ru": "Электронная", "en": "Electronic"},
        {"ru": "R&B", "en": "R&B"},
    ]),
    "🎷 Классика / Джаз / Соул": _by_ru([
        {"ru": "Амбиент", "en": "Ambient"},
        {"ru": "Блюз", "en": "Blues"},
        {"ru": "Госпел", "en": "Gospel"},
        {"ru": "Джаз", "en": "Jazz"},
        {"ru": "Классика", "en": "Classical"},
        {"ru": "Нью-эйдж / Медитация", "en": "New Age / Meditation"},
        {"ru": "Регги", "en": "Reggae"},
        {"ru": "Соул", "en": "Soul"},
    ]),
    "🌍 Мировая / Русская / Ретро": _by_ru([
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
        {"ru": "Ретро / Винтаж", "en": "Vintage"},
        {"ru": "Романс", "en": "Romance"},
        {"ru": "Саундтрек", "en": "Soundtrack"},
        {"ru": "Фолк", "en": "Folk"},
        {"ru": "Шансон", "en": "Chanson / Russian bard"},
        {"ru": "Эстрада", "en": "Estrada (Soviet pop)"},
    ]),
}

# НАСТРОЕНИЕ — 6 смысловых групп, каждая по алфавиту RU
# Дедупликация по EN чтобы убрать дубли из Mood/feel + Mood/feel / Fun
_mood_raw = _d("Mood/feel") + _d("Mood/feel / Fun")
_mood_seen: set[str] = set()
_mood_all: list[dict] = []
for _t in _mood_raw:
    if _t["en"] not in _mood_seen:
        _mood_seen.add(_t["en"])
        _mood_all.append(_t)

# Распределяем по ключевым словам EN (каждый тег попадает только в одну группу)
_POSITIVE_EN   = {"Upbeat","Uplifting","Positive","Happy","Hopeful","Cheerfulness",
                  "Joy","Excitement","Happiness","Amusement","Merriment",
                  "Laughter","Pleasure","Enjoyment","Party","Light"}
_DARK_EN       = {"Dark","Tense","Tension","Gritty","Swagger","Mysterious","Moody","Gloomy"}
_SAD_EN        = {"Sad","Reflective","Dreamy","Retro","Romantic","Warm",
                  "Lyrical","Melancholic","Soulful","Calm"}
_ENERGY_EN     = {"Energetic","Epic","Powerful","Driving","Anthemic","Building",
                  "Dramatic","Percussive","Rhythmic","Intense"}
_FUN_EN        = {"Fun","Playful","Quirky","Catchy","Cool","Sexy"}
# остальные → Атмосферные

_EXTRA_MOODS = [
    {"ru": "Душевное",      "en": "Soulful"},
    {"ru": "Лирическое",    "en": "Lyrical"},
    {"ru": "Медитативное",  "en": "Meditative"},
    {"ru": "Меланхоличное", "en": "Melancholic"},
    {"ru": "Мрачное",       "en": "Gloomy"},
    {"ru": "Спокойное",     "en": "Calm"},
]
for _t in _EXTRA_MOODS:
    if _t["en"] not in _mood_seen:
        _mood_seen.add(_t["en"])
        _mood_all.append(_t)

def _group_mood_rest(tags: list[dict], already_used: set) -> list[dict]:
    return _by_ru([t for t in tags if t["en"] not in already_used])

_used_en: set[str] = set()
def _mg(en_set: set) -> list[dict]:
    result = _by_ru([t for t in _mood_all if t["en"] in en_set and t["en"] not in _used_en])
    _used_en.update(en_set)
    return result

MOOD_GROUPS: dict[str, list[dict]] = {}
MOOD_GROUPS["😊 Позитивные / Радостные"]  = _mg(_POSITIVE_EN)
MOOD_GROUPS["⚡ Энергичные / Мощные"]     = _mg(_ENERGY_EN)
MOOD_GROUPS["😌 Романтичные / Тёплые"]   = _mg(_SAD_EN)
MOOD_GROUPS["😄 Весёлые / Игривые"]      = _mg(_FUN_EN)
MOOD_GROUPS["🌙 Тёмные / Напряжённые"]   = _mg(_DARK_EN)
MOOD_GROUPS["🎭 Атмосферные / Особые"]   = _group_mood_rest(_mood_all, _used_en)

# ЭПОХА — хронологический порядок, не алфавитный
DECADES = [
    {"ru": "30-е", "en": "1930s"}, {"ru": "40-е", "en": "1940s"},
    {"ru": "50-е", "en": "1950s"}, {"ru": "60-е", "en": "1960s"},
    {"ru": "70-е", "en": "1970s"}, {"ru": "80-е", "en": "1980s"},
    {"ru": "90-е", "en": "1990s"}, {"ru": "2000-е", "en": "2000s"},
    {"ru": "2010-е", "en": "2010s"}, {"ru": "2020-е", "en": "2020s"},
]

# ВОКАЛ
VOCAL_TAGS = _by_ru([
    {"ru": "Акапелла", "en": "A cappella"},
    {"ru": "Бэк-вокал", "en": "Background vocals"},
    {"ru": "Вокализ", "en": "Vocalise"},
    {"ru": "Дуэт", "en": "Duet"},
    {"ru": "Женский вокал", "en": "Female vocal"},
    {"ru": "Инструментальная", "en": "Instrumental"},
    {"ru": "Мужской вокал", "en": "Male vocal"},
    {"ru": "Речитатив / Рэп", "en": "Rap / Spoken"},
    {"ru": "Хор", "en": "Choir"},
    {"ru": "Шёпотом", "en": "Whisper"},
])

# ИНСТРУМЕНТЫ
INSTR_TAGS = _by_ru([
    {"ru": "Акустическая гитара", "en": "Acoustic guitar"},
    {"ru": "Барабаны", "en": "Drums"},
    {"ru": "Бас-гитара", "en": "Bass guitar"},
    {"ru": "Виолончель", "en": "Cello"},
    {"ru": "Духовые", "en": "Brass / Horns"},
    {"ru": "Оркестр", "en": "Orchestra"},
    {"ru": "Перкуссия", "en": "Percussion"},
    {"ru": "Саксофон", "en": "Saxophone"},
    {"ru": "Синтезатор", "en": "Synth"},
    {"ru": "Скрипка", "en": "Violin"},
    {"ru": "Струнные", "en": "Strings"},
    {"ru": "Труба", "en": "Trumpet"},
    {"ru": "Флейта", "en": "Flute"},
    {"ru": "Фортепиано", "en": "Piano"},
    {"ru": "Электрогитара", "en": "Electric guitar"},
])

# ТЕМЫ ТЕКСТА — все теги отсортированы по RU и делятся на 3 колонки
_theme_all = _by_ru(
    _d("Lyric themes") + _d("Lyric themes / Adventure") +
    _d("Lyric themes / Ambition") + _d("Lyric themes / Love") +
    [{"ru": "Мат / Ненормативная лексика", "en": "Explicit / Profanity"},
     {"ru": "Новый год", "en": "New Year"},
     {"ru": "Ссора", "en": "Quarrel"}]
)
# Дедупликация по (ru, en) + исключения
_THEME_EXCLUDE_EN = {"Attachment"}
_seen: set = set()
_theme_dedup: list[dict] = []
for t in _theme_all:
    if t["en"] in _THEME_EXCLUDE_EN:
        continue
    k = (t["ru"], t["en"])
    if k not in _seen:
        _seen.add(k)
        _theme_dedup.append(t)
_theme_all = _theme_dedup

_chunk = len(_theme_all) // 3
THEME_GROUPS = {
    "А — " + _theme_all[_chunk - 1]["ru"][:2]:      _theme_all[:_chunk],
    _theme_all[_chunk]["ru"][:2] + " — " + _theme_all[2*_chunk - 1]["ru"][:2]: _theme_all[_chunk:2*_chunk],
    _theme_all[2*_chunk]["ru"][:2] + " — Я":        _theme_all[2*_chunk:],
}

# ─── Data helpers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Загружаю треки...")
def load_tracks() -> pd.DataFrame:
    df = pd.read_csv(BASE / "tracks_rows.csv", low_memory=False)
    return df[df["link"].str.startswith("https://music.yandex.ru/", na=False)].reset_index(drop=True)

def get_daily_batch(df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(int(date.today().strftime("%Y%m%d")))
    idx = list(range(len(df)))
    rng.shuffle(idx)
    return df.iloc[idx[:DAILY_BATCH_SIZE]].reset_index(drop=True)

def tagged_ids_for(employee: str) -> set:
    sb = _supabase_client()
    if sb:
        res = sb.table("human_tags").select("track_id").eq("employee", employee).execute()
        return {r["track_id"] for r in res.data}
    # CSV fallback (локальный запуск)
    csv_path = BASE / "human_tags.csv"
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set() if df.empty else set(df[df["employee"] == employee]["track_id"].astype(str))

def save_entry(track_id, url, title, artist, employee, tags_dict, notes):
    sb = _supabase_client()
    if sb:
        sb.table("human_tags").insert({
            "track_id": str(track_id), "yandex_url": url, "title": title,
            "artist": artist, "employee": employee,
            "tagged_at": datetime.now().isoformat(timespec="seconds"),
            "notes": notes,
            **tags_dict,
        }).execute()
    else:
        # CSV fallback (локальный запуск)
        row = {"track_id": str(track_id), "yandex_url": url, "title": title,
               "artist": artist, "employee": employee,
               "tagged_at": datetime.now().isoformat(timespec="seconds"),
               "notes": notes,
               **{k: json.dumps(v, ensure_ascii=False) for k, v in tags_dict.items()}}
        csv_path = BASE / "human_tags.csv"
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

def extract_ids(url: str):
    t = re.search(r"/track/(\d+)", url)
    a = re.search(r"/album/(\d+)", url)
    return (t.group(1) if t else None), (a.group(1) if a else None)

def yandex_player(url: str):
    tid, aid = extract_ids(url)
    if not tid:
        return
    src = f"https://music.yandex.ru/iframe/#track/{tid}" + (f"/{aid}" if aid else "")
    components.html(
        f'<iframe style="border:none;width:100%;height:88px;border-radius:8px" '
        f'src="{src}" allow="autoplay"></iframe>', height=92)

# ─── UI helpers ───────────────────────────────────────────────────────────────

def pills_group(label: str, tags: list[dict], key: str):
    """Render one labelled group of pills."""
    if not tags:
        return
    st.markdown(f'<div class="group-label">{label}</div>', unsafe_allow_html=True)
    st.pills("", _fmt(tags), selection_mode="multi",
             key=key, label_visibility="collapsed")

def collect(*keys) -> list[str]:
    result = []
    for k in keys:
        result.extend(st.session_state.get(k, []))
    return result

def clear_keys(*keys):
    for k in keys:
        if k in st.session_state:
            st.session_state[k] = []

# ─── Login ────────────────────────────────────────────────────────────────────

if "employee" not in st.session_state:
    st.title("🎵 Синкотека — Разметка треков")
    col, _ = st.columns([2, 3])
    with col:
        name = st.text_input("Твоё имя", placeholder="Например: Катя")
        if st.button("Начать", type="primary") and name.strip():
            st.session_state.update({"employee": name.strip(), "track_idx": 0})
            st.rerun()
    st.stop()

# ─── Main ─────────────────────────────────────────────────────────────────────

employee   = st.session_state["employee"]
all_tracks = load_tracks()
batch      = get_daily_batch(all_tracks)
total      = len(batch)
done_ids   = tagged_ids_for(employee)
today_ids  = set(str(t) for t in batch["id"])
today_done = len(done_ids & today_ids)

if "track_idx" not in st.session_state:
    st.session_state["track_idx"] = 0
idx = max(0, min(st.session_state["track_idx"], total - 1))

# ── Progress bar (topmost visible element) ─
today_str = date.today().strftime("%d.%m")
pct = today_done / DAILY_BATCH_SIZE
st.progress(pct, text=f"🎵 {employee}  ·  {today_str}  ·  сегодня: {today_done}/{DAILY_BATCH_SIZE}  ·  трек {idx+1}/{total}")

if today_done >= DAILY_BATCH_SIZE:
    st.success(f"✅ Все {DAILY_BATCH_SIZE} треков на сегодня готовы! Возвращайся завтра.")
    st.stop()

# ── Track ─
track    = batch.iloc[idx]
tid      = str(track["id"])
url      = str(track["link"])
title    = str(track.get("title", "—"))
artist   = str(track.get("artist", "—"))
album    = str(track.get("album", ""))
g        = [str(track.get(f"genre_{i}", "")) for i in range(1, 4)]
genre    = " / ".join(x for x in g if x not in ("", "nan", "None"))
done     = tid in done_ids
pk       = f"t{tid}"  # short key prefix

# pill state keys (one per group)
G_KEYS   = {g: f"{pk}_g{i}"  for i, g in enumerate(GENRE_GROUPS)}
M_KEYS   = {g: f"{pk}_m{i}"  for i, g in enumerate(MOOD_GROUPS)}
ERA_KEY  = f"{pk}_era"
TEMPO_KEY= f"{pk}_tempo"
VOC_KEY  = f"{pk}_voc"
INS_KEY  = f"{pk}_ins"
TH_KEYS  = {g: f"{pk}_th{i}" for i, g in enumerate(THEME_GROUPS)}

# ── Two columns ─
left, right = st.columns([1, 2])

with left:
    badge = "✅ " if done else ""
    st.markdown(
        f'<div style="background:#1a1a2e;border-radius:8px;padding:7px 12px;margin-bottom:3px">'
        f'<div style="font-size:15px;font-weight:700">{badge}{title}</div>'
        f'<div style="font-size:12px;color:#bbb">{artist}'
        f'{" · " + album if album not in ("", "nan") else ""}</div>'
        f'<div style="font-size:11px;color:#555">{genre}</div></div>',
        unsafe_allow_html=True)

    yandex_player(url)
    st.link_button("Открыть в Яндекс.Музыке ↗", url)

    notes = st.text_input("Заметка", placeholder="необязательно",
                          label_visibility="collapsed", key=f"notes_{pk}")

    # count selected
    all_sel = (collect(*G_KEYS.values()) + collect(*M_KEYS.values()) +
               collect(ERA_KEY, TEMPO_KEY, VOC_KEY, INS_KEY) + collect(*TH_KEYS.values()))
    n = len(all_sel)

    s1, s2, s3 = st.columns([2, 2, 1])
    with s1:
        nav_l = st.button("◀", use_container_width=True)
    with s2:
        nav_r = st.button("▶", use_container_width=True)
    with s3:
        if st.button("↩", use_container_width=True, help="Выйти"):
            del st.session_state["employee"]
            st.rerun()

    if nav_l and idx > 0:
        st.session_state["track_idx"] = idx - 1; st.rerun()
    if nav_r and idx < total - 1:
        st.session_state["track_idx"] = idx + 1; st.rerun()

    c1, c2 = st.columns([3, 2])
    with c1:
        label = f"💾 Сохранить ({n})" if n else "💾 Сохранить"
        if st.button(label, type="primary", use_container_width=True, disabled=(n == 0)):
            # Build flat list of selected tag dicts per category
            def _resolve(keys_dict: dict, groups: dict) -> list[dict]:
                result = []
                label_to_tag = {f"{t['ru']} · {t['en']}": t
                                 for tags in groups.values() for t in tags}
                for lbl in collect(*keys_dict.values()):
                    if lbl in label_to_tag:
                        result.append(label_to_tag[lbl])
                return result

            tempo_src = _d("Tempo")
            label_to_tempo = {f"{t['ru']} · {t['en']}": t for t in tempo_src}

            tags_dict = {
                "genre": _resolve(G_KEYS,  GENRE_GROUPS),
                "mood":  _resolve(M_KEYS,  MOOD_GROUPS),
                "era":   [{"ru": l.split(" · ")[0], "en": l.split(" · ")[1]}
                          for l in collect(ERA_KEY) if " · " in l],
                "tempo": [label_to_tempo[l] for l in collect(TEMPO_KEY) if l in label_to_tempo],
                "vocal": [{"ru": l.split(" · ")[0], "en": l.split(" · ")[1]}
                          for l in collect(VOC_KEY) if " · " in l],
                "instr": [{"ru": l.split(" · ")[0], "en": l.split(" · ")[1]}
                          for l in collect(INS_KEY) if " · " in l],
                "theme": _resolve(TH_KEYS, THEME_GROUPS),
            }
            save_entry(tid, url, title, artist, employee, tags_dict, notes)
            clear_keys(*G_KEYS.values(), *M_KEYS.values(),
                       ERA_KEY, TEMPO_KEY, VOC_KEY, INS_KEY, *TH_KEYS.values())
            if idx < total - 1:
                st.session_state["track_idx"] = idx + 1
            st.rerun()
    with c2:
        if st.button("⏭ Пропустить", use_container_width=True):
            save_entry(tid, url, title, artist, employee,
                       {k: [] for k in ("genre","mood","era","tempo","vocal","instr","theme")}, "SKIP")
            if idx < total - 1:
                st.session_state["track_idx"] = idx + 1
            st.rerun()

# ── Right: tabs with grouped pills ─
with right:
    tab_g, tab_m, tab_e, tab_v, tab_t = st.tabs(
        ["🎭 Жанр", "😊 Настроение", "🕰️ Эпоха + Темп", "🎤 Вокал + Инструменты", "📖 Темы текста"])

    with tab_g:
        for gname, gtags in GENRE_GROUPS.items():
            pills_group(gname, gtags, G_KEYS[gname])

    with tab_m:
        for gname, gtags in MOOD_GROUPS.items():
            pills_group(gname, gtags, M_KEYS[gname])

    with tab_e:
        pills_group("🕰️ Эпоха / Decade", DECADES, ERA_KEY)
        tempo_tags = _by_ru(_d("Tempo") + [{"ru": "Переменчивый", "en": "Variable tempo"}])
        pills_group("⏱️ Темп / Tempo", tempo_tags, TEMPO_KEY)

    with tab_v:
        pills_group("🎤 Вокал / Vocals", VOCAL_TAGS, VOC_KEY)
        pills_group("🎸 Инструменты / Instruments", INSTR_TAGS, INS_KEY)

    with tab_t:
        cols = st.columns(3)
        for i, (gname, gtags) in enumerate(THEME_GROUPS.items()):
            with cols[i]:
                st.markdown(f'<div class="group-label">{gname}</div>', unsafe_allow_html=True)
                st.pills("", _fmt(gtags), selection_mode="multi",
                         key=TH_KEYS[gname], label_visibility="collapsed")

# ── Stats ─
with st.expander("📊 Мой прогресс"):
    sb = _supabase_client()
    if sb:
        res = sb.table("human_tags").select("track_id,title,artist,tagged_at").eq("employee", employee).execute()
        if res.data:
            df_my = pd.DataFrame(res.data)
            st.write(f"Всего размечено: **{len(df_my)}** треков")
            st.dataframe(df_my[["title","artist","tagged_at"]], use_container_width=True)
    else:
        csv_path = BASE / "human_tags.csv"
        if csv_path.exists():
            df_my = pd.read_csv(csv_path)
            df_my = df_my[df_my["employee"] == employee]
            st.write(f"Всего размечено: **{len(df_my)}** треков")
            st.download_button("⬇️ Скачать CSV", df_my.to_csv(index=False).encode(),
                               f"tags_{employee}.csv", "text/csv")
