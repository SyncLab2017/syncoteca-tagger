# Syncoteca Tagger — контекст проекта

## Что это
Инструмент обогащения метаданных музыкального каталога для sync-лицензирования (Synclab Pro).

## Три компонента

| Файл | Что делает | Запуск |
|------|-----------|--------|
| `human_tagger.py` | Streamlit-форма ручной разметки треков | `streamlit run human_tagger.py` |
| `app.py` | Streamlit-приложение: URL/файл → аудио-анализ → Claude теги → CSV | `streamlit run app.py` |
| `syncoteca_bot.py` | Telegram-бот для просмотра каталога | `./start_bot.sh` |

## Деплой
- `human_tagger.py` и `app.py` — на Streamlit Cloud (автодеплой при push в `main`)
- URL: `syncoteca-tagger-j9qizecjhfjkrsaaqokhnu.streamlit.app`
- Remote: `https://github.com/SyncLab2017/syncoteca-tagger.git`
- Secrets (Streamlit Cloud): `SUPABASE_URL`, `SUPABASE_KEY` (в `.streamlit/secrets.toml` локально)

## База данных
- **Supabase**: таблица `human_tags` — сохраняет результаты ручной разметки
- **SQLite** (`syncoteca.db`): используется Telegram-ботом (hardcoded path — см. `syncoteca_bot.py:7`)
- **Треки**: `tracks_rows.csv` — источник для `human_tagger.py`

## Структура тегов (human_tagger.py)
Теги хранятся как JSON-массивы объектов `{"ru": "...", "en": "..."}` в колонках:
`genre`, `mood`, `era`, `tempo`, `vocal`, `instr`, `theme`

Источники тегов:
- `DISCO Tags.csv` — основной словарь тегов (EN + RU перевод)
- Кастомные теги прямо в коде (жанры, вокал, инструменты, и т.д.)

## Окружение
- Python 3.11, venv в `./venv` (основной) и `./venv_new`
- Ключевые зависимости: `streamlit`, `anthropic`, `supabase`, `essentia`, `yt-dlp`, `python-telegram-bot`
- `essentia` — опциональна (graceful fallback в `app.py`)

## Текущее состояние (обновлено 2026-05-30)

### app.py — рефакторинг под схему human_tags
- ✅ Словарь тегов встроен в app.py: genre (41), mood (из DISCO Tags.csv + экстра), era (10), tempo (5), vocal (11), instr (15), theme (из DISCO Tags.csv)
- ✅ `enrich_with_claude()` переписан: Claude выбирает EN-теги из словаря → Python маппит в `{en,ru}`; few-shot из 4 реальных примеров разметки
- ✅ Выходной формат совпадает со схемой `human_tags`: поля `genre/mood/era/tempo/vocal/instr/theme` как `[{en,ru}]`
- ✅ CSV-экспорт в формате human_tags (JSON-массивы, те же колонки)
- ✅ Сохранение в Supabase `human_tags` с `employee='auto'` (чекбокс в сайдбаре)
- ✅ UI: карточки треков с раскрытыми тегами по категориям

### human_tagger.py — правки 2026-05-22
- ✅ Добавлен жанр **Романс · Romance**
- ✅ Добавлены 6 тегов настроения
- ✅ Добавлены жанры Дроны, Звуковой пейзаж, Ретро
- ✅ Удалены 13 неиспользуемых тем
- ✅ Миграция: Танцевальная → Дэнс в БД

## Частые задачи
- **Добавить жанр**: в `GENRE_GROUPS` нужной группы добавить `{"ru": "...", "en": "..."}` — `_by_ru()` отсортирует автоматически
- **Добавить тег настроения**: добавить в `_EXTRA_MOODS` + EN-ключ в нужный set (`_SAD_EN`, `_DARK_EN`, `_ENERGY_EN`, и т.д.) если нужна конкретная группа
- **Деплой**: `git add human_tagger.py && git commit -m "..." && git push origin main`
