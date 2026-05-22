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

## Текущее состояние (обновлено 2026-05-22)
- ✅ Добавлен жанр **Романс · Romance** (группа Мировая / Русская / Ретро)
- ✅ Добавлены 6 тегов настроения: Душевное, Лирическое, Меланхоличное, Спокойное → Романтичные/Тёплые; Мрачное → Тёмные/Напряжённые; Медитативное → Атмосферные/Особые
- ✅ Добавлены жанры **Дроны · Drones**, **Звуковой пейзаж · Soundscape** (группа Классика / Джаз / Соул), **Ретро · Retro** (группа Мировая / Русская / Ретро, рядом с "Ретро / Винтаж")
- ✅ Удалены 13 тем текста никогда не использовавшихся: Believe, Change, Commitment, Empowerment, Escape, Expedition, Goal, Hustle, Loss, Rebellion, Success, Vision, Voyage → добавлены в `_THEME_EXCLUDE_EN`
- ✅ Миграция БД: 22 строки с legacy-тегом **Танцевальная · Danceable** → **Дэнс · Dance** (тег существовал до удаления из кода)
- Тег "Эпичная · Epic" уже существовал в группе Энергичные/Мощные

## Частые задачи
- **Добавить жанр**: в `GENRE_GROUPS` нужной группы добавить `{"ru": "...", "en": "..."}` — `_by_ru()` отсортирует автоматически
- **Добавить тег настроения**: добавить в `_EXTRA_MOODS` + EN-ключ в нужный set (`_SAD_EN`, `_DARK_EN`, `_ENERGY_EN`, и т.д.) если нужна конкретная группа
- **Деплой**: `git add human_tagger.py && git commit -m "..." && git push origin main`
