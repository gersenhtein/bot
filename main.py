import asyncio
import random
import textwrap
import logging
from datetime import datetime, timedelta
import aiosqlite
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode, ChatType
from aiogram.filters import Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    PollAnswer,
    FSInputFile,
)
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from functools import wraps
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from aiogram.fsm.storage.memory import MemoryStorage
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transliterate import translit
from setting import BOT_TOKEN, ADMIN_IDS, DB_PATH

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
TIMEZONE = "Europe/Moscow"

# Флаг для предотвращения одновременных подборов
pairing_in_progress = False

# Глобальный словарь для хранения данных опросов
poll_data = {}

# Инициализация SentenceTransformer и NLTK
model = SentenceTransformer('sentence-transformers/LaBSE')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian') + stopwords.words('english'))

# Кэш эмбеддингов
embedding_cache = {}

# Расширенный словарь синонимов (включая возможные опечатки)
SYNONYMS = {
    'программирование': 'programming',
    'programmirovanie': 'programming',
    'программирование и it': 'programming',
    'разработка по': 'programming',
    'разработка программного обеспечения': 'programming',
    'разработка программ': 'programming',
    'software development': 'programming',
    'it': 'programming',
    'кодирование': 'programming',
    'coding': 'programming',
    'программинг': 'programming',
    'pragraming': 'programming',  # Опечатка
    'progrmming': 'programming',  # Опечатка
    'programing': 'programming',  # Опечатка
    'backend': 'programming',
    'frontend': 'programming',
    'fullstack': 'programming',
    'разработка': 'programming',
    'дизайн': 'design',
    'ui/ux': 'design',
    'графический дизайн': 'design',
    'маркетинг': 'marketing',
    'digital marketing': 'marketing',
    'цифровой маркетинг': 'marketing',
    'реклама': 'marketing',
}

def preprocess_text(text: str) -> str:
    if not text:
        logging.warning("Empty text received for preprocessing")
        return text
    # Приведение к нижнему регистру и удаление лишних пробелов
    text = text.lower().strip()
    logging.debug(f"Original text: '{text}'")
    
    # Нормализация через словарь синонимов
    text = SYNONYMS.get(text, text)
    logging.debug(f"After synonym normalization: '{text}'")
    
    # Транслитерация русских слов
    try:
        text = translit(text, 'ru', reversed=True)
    except Exception as e:
        logging.warning(f"Transliteration failed for '{text}': {e}")
    logging.debug(f"After transliteration: '{text}'")
    
    # Токенизация и удаление стоп-слов
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    result = ' '.join(tokens)
    logging.debug(f"Final preprocessed text: '{result}'")
    return result

def is_similar(s1: str, s2: str, threshold: float = 0.7) -> bool:
    s1_clean = preprocess_text(s1)
    s2_clean = preprocess_text(s2)
    if not s1_clean or not s2_clean:
        logging.warning(f"Empty preprocessed text: s1='{s1_clean}', s2='{s2_clean}'")
        return False
    
    # Если строки идентичны после нормализации
    if s1_clean == s2_clean:
        logging.info(f"Exact match after preprocessing: '{s1_clean}' == '{s2_clean}'")
        return True
    
    # Динамический порог: 0.6 для синонимов, 0.7 для остальных
    effective_threshold = 0.6 if s1_clean in SYNONYMS.values() and s2_clean in SYNONYMS.values() else threshold
    
    # Используем кэш эмбеддингов
    if s1_clean not in embedding_cache:
        embedding_cache[s1_clean] = model.encode(s1_clean)
    if s2_clean not in embedding_cache:
        embedding_cache[s2_clean] = model.encode(s2_clean)
    
    sim = cosine_similarity([embedding_cache[s1_clean]], [embedding_cache[s2_clean]])[0][0]
    logging.info(f"Similarity between '{s1_clean}' and '{s2_clean}': {sim} (threshold={effective_threshold})")
    return sim >= effective_threshold

def clean_string(text: str) -> str:
    if not text:
        return text
    # Удаляем суррогатные символы и лишние пробелы
    text = ''.join(c for c in text.strip() if not (0xD800 <= ord(c) <= 0xDFFF))
    return text

CREATE_USERS_SQL = """
CREATE TABLE IF NOT EXISTS users (
    tg_id       INTEGER PRIMARY KEY,
    username    TEXT,
    name        TEXT,
    bio         TEXT,
    sphere      TEXT,
    photo_id    TEXT,
    match_mode  TEXT DEFAULT 'same',
    is_waiting  INTEGER DEFAULT 0,
    registered  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_paired TIMESTAMP
);
"""

CREATE_GROUPS_SQL = """
CREATE TABLE IF NOT EXISTS groups (
    chat_id     INTEGER PRIMARY KEY,
    registered  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_GROUP_USERS_SQL = """
CREATE TABLE IF NOT EXISTS group_users (
    chat_id     INTEGER,
    tg_id       INTEGER,
    PRIMARY KEY (chat_id, tg_id)
);
"""

class ProfileForm(StatesGroup):
    name = State()
    sphere = State()
    expertise = State()
    photo = State()
    video = State()

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_USERS_SQL)
        await db.execute(CREATE_GROUPS_SQL)
        await db.execute(CREATE_GROUP_USERS_SQL)
        await db.commit()

def admin_required(handler):
    @wraps(handler)
    async def wrapper(message: Message, *args, **kwargs):
        if message.from_user.id in ADMIN_IDS:
            return await handler(message, *args, **kwargs)
        member = await message.bot.get_chat_member(message.chat.id, message.from_user.id)
        if member.status in ["administrator", "creator"]:
            return await handler(message, *args, **kwargs)
        await message.reply("Эта команда доступна только администраторам.")
    return wrapper

async def upsert_user(tg_id: int, name: str, username: str | None, bio: str | None = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO users (tg_id, name, username, bio)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(tg_id) DO UPDATE SET
                name=excluded.name,
                username=excluded.username,
                bio=COALESCE(excluded.bio, users.bio)
            """,
            (tg_id, name, username, bio),
        )
        await db.commit()

async def save_user_photo(tg_id: int, photo_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE users SET photo_id=? WHERE tg_id=?", (photo_id, tg_id))
        await db.commit()

async def set_user_sphere_waiting(tg_id: int, sphere: str):
    # Очистка и валидация sphere
    sphere = clean_string(sphere.strip())
    if not sphere:
        logging.error(f"Empty sphere for tg_id={tg_id}")
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET sphere=?, match_mode='same', is_waiting=1 WHERE tg_id=?",
            (sphere.lower(), tg_id),
        )
        await db.commit()

async def set_user_waiting_status(tg_id: int, is_waiting: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET is_waiting=? WHERE tg_id=?",
            (is_waiting, tg_id),
        )
        await db.commit()

async def is_user_registered(tg_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT 1 FROM users WHERE tg_id=?", (tg_id,)) as cursor:
            return bool(await cursor.fetchone())

async def get_waiting_users():
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT tg_id, username, name, sphere, match_mode, photo_id, bio, last_paired "
            "FROM users WHERE is_waiting=1 AND match_mode='same'"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(zip([column[0] for column in cursor.description], row)) for row in rows]

async def mark_users_paired(user_ids: list[int]):
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE users SET is_waiting=0, last_paired=? WHERE tg_id IN ({','.join('?'*len(user_ids))})",
            (now, *user_ids),
        )
        await db.commit()

async def add_group(chat_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO groups (chat_id) VALUES (?)",
            (chat_id,),
        )
        await db.commit()

async def add_group_user(chat_id: int, tg_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO group_users (chat_id, tg_id) VALUES (?, ?)",
            (chat_id, tg_id),
        )
        await db.commit()

async def remove_group_user(chat_id: int, tg_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM group_users WHERE chat_id=? AND tg_id=?",
            (chat_id, tg_id),
        )
        await db.commit()

async def get_groups():
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT chat_id FROM groups") as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

async def get_group_users(chat_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT tg_id FROM group_users WHERE chat_id=?", (chat_id,)) as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

# ---------------------------------------------------------------------------
# Command & message handlers
# ---------------------------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    if message.chat.type != ChatType.PRIVATE:
        await add_group(message.chat.id)
        await add_group_user(message.chat.id, message.from_user.id)
        poll = await bot.send_poll(
            chat_id=message.chat.id,
            question="Хотите участвовать в подборе пар для нетворкинга?",
            options=["Да", "Нет"],
            is_anonymous=False,
            allows_multiple_answers=False,
        )
        poll_data[poll.poll.id] = {"chat_id": message.chat.id}
        if not await is_user_registered(message.from_user.id):
            try:
                await bot.send_message(
                    chat_id=message.from_user.id,
                    text="Привет! Ты выбрал участие в подборе пар в группе, но еще не зарегистрирован. Напиши /start в личных сообщениях, чтобы пройти регистрацию!"
                )
                await message.answer("👋 Бот добавлен в группу! Я отправил тебе личное сообщение для регистрации. Ответь на опрос в группе, чтобы подтвердить участие.")
            except Exception as exc:
                logging.error(f"Failed to send registration prompt to {message.from_user.id}: {exc}")
                await message.answer("👋 Бот добавлен в группу! Пожалуйста, напиши /start в личных сообщениях бота, чтобы зарегистрироваться, и ответь на опрос для участия.")
        else:
            await message.answer("👋 Бот добавлен в группу! Ответь на опрос, чтобы подтвердить участие в подборе пар в понедельник в 9:00 по Москве.")
        return

    await upsert_user(
        message.from_user.id,
        message.from_user.first_name,
        message.from_user.username,
        None,
    )
    welcome_message = textwrap.dedent(
        """
        Привет! Я Mactracger Bot ☕️
        Помогаю находить интересные знакомства каждую неделю.

        1️⃣ Расскажи коротко о себе и о том, что ищешь в нетворкинге.
        """
    ).strip()
    await message.answer(welcome_message)
    try:
        await bot.send_photo(
            chat_id=message.from_user.id,
            photo=FSInputFile(path="my_photo.jpg"),
            caption="вот пример!😊"
        )
    except Exception as e:
        logging.error(f"Failed to send welcome photo: {e}")
        await message.answer("Не удалось отправить фото, но давай продолжим! 😊")
    await state.update_data(state="awaiting_bio")

@dp.message(Command("pairnow"))
@admin_required
async def cmd_pairnow(message: Message):
    global pairing_in_progress
    if pairing_in_progress:
        await message.answer("Подбор пар уже выполняется, пожалуйста, подождите.")
        return

    users = await get_waiting_users()
    if len(users) < 2:
        await message.answer("Недостаточно пользователей для подбора пар (требуется минимум 2).")
        return

    pairing_in_progress = True
    try:
        await message.answer("Запускаю немедленный подбор пар...")
        await create_pairs_and_notify()
        await message.answer("Подбор пар завершён! Уведомления отправлены.")
    finally:
        pairing_in_progress = False

@dp.message(Command("profile"))
async def cmd_profile(message: Message):
    if not await is_user_registered(message.from_user.id):
        await message.answer("Вы ещё не зарегистрированы. Напишите /start для регистрации.")
        return
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT name, username, bio, sphere, photo_id FROM users WHERE tg_id=?", (message.from_user.id,)
        ) as cursor:
            user = await cursor.fetchone()
            if user:
                name, username, bio, sphere, photo_id = user
                text = (
                    f"📛 Имя: {name}\n"
                    f"👤 Username: @{username if username else 'не указан'}\n"
                    f"ℹ️ О себе: {bio or 'не указана'}\n"
                    f"💼 Сфера: {sphere or 'не указана'}"
                )
                if photo_id:
                    await bot.send_photo(chat_id=message.from_user.id, photo=photo_id, caption=text)
                else:
                    await message.answer(text)
                await message.answer(
                    "Хотите отредактировать профиль? Напишите /edit",
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                        [InlineKeyboardButton(text="Редактировать", callback_data="edit_profile")]
                    ])
                )

@dp.callback_query(F.data == "edit_profile")
async def edit_profile(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Введите новое имя:")
    await state.set_state(ProfileForm.name)
    await callback.answer()

@dp.message(ProfileForm.name)
async def process_name(message: Message, state: FSMContext):
    await state.update_data(name=message.text)
    await message.answer("Введите новую сферу деятельности:")
    await state.set_state(ProfileForm.sphere)

@dp.message(ProfileForm.sphere)
async def process_sphere(message: Message, state: FSMContext):
    await state.update_data(sphere=message.text)
    await message.answer("Отправьте новое фото (или напишите 'пропустить'):")
    await state.set_state(ProfileForm.photo)

@dp.message(ProfileForm.photo, F.photo)
async def process_photo(message: Message, state: FSMContext):
    data = await state.get_data()
    await upsert_user(
        message.from_user.id,
        data["name"],
        message.from_user.username,
        None,  # bio можно добавить в форму позже
    )
    await save_user_photo(message.from_user.id, message.photo[-1].file_id)
    await set_user_sphere_waiting(message.from_user.id, data["sphere"])
    await message.answer("Профиль обновлён!")
    await state.clear()

@dp.message(ProfileForm.photo, F.text.lower() == "пропустить")
async def skip_photo(message: Message, state: FSMContext):
    data = await state.get_data()
    await upsert_user(
        message.from_user.id,
        data["name"],
        message.from_user.username,
        None,
    )
    await set_user_sphere_waiting(message.from_user.id, data["sphere"])
    await message.answer("Профиль обновлён без фото!")
    await state.clear()

@dp.callback_query(F.data == "edit_profile")
async def edit_profile(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Введите новое имя:")
    await state.set_state(ProfileForm.name)
    await callback.answer()

@dp.message(ProfileForm.name)
async def process_name(message: Message, state: FSMContext):
    await state.update_data(name=message.text)
    await message.answer("Введите новую сферу деятельности:")
    await state.set_state(ProfileForm.sphere)

@dp.message(ProfileForm.sphere)
async def process_sphere(message: Message, state: FSMContext):
    await state.update_data(sphere=message.text)
    await message.answer("Отправьте новое фото (или напишите 'пропустить'):")
    await state.set_state(ProfileForm.photo)

@dp.message(ProfileForm.photo, F.photo)
async def process_photo(message: Message, state: FSMContext):
    data = await state.get_data()
    await upsert_user(
        message.from_user.id,
        data["name"],
        message.from_user.username,
        None,  # bio можно добавить в форму позже
    )
    await save_user_photo(message.from_user.id, message.photo[-1].file_id)
    await set_user_sphere_waiting(message.from_user.id, data["sphere"])
    await message.answer("Профиль обновлён!")
    await state.clear()

@dp.message(ProfileForm.photo, F.text.lower() == "пропустить")
async def skip_photo(message: Message, state: FSMContext):
    data = await state.get_data()
    await upsert_user(
        message.from_user.id,
        data["name"],
        message.from_user.username,
        None,
    )
    await set_user_sphere_waiting(message.from_user.id, data["sphere"])
    await message.answer("Профиль обновлён без фото!")
    await state.clear()

@dp.message(Command("stats"))
@admin_required
async def cmd_stats(message: Message):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM users") as cursor:
            total_users = (await cursor.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM users WHERE is_waiting=1") as cursor:
            waiting_users = (await cursor.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM groups") as cursor:
            total_groups = (await cursor.fetchone())[0]
        async with db.execute("SELECT sphere, COUNT(*) FROM users WHERE sphere IS NOT NULL GROUP BY sphere") as cursor:
            spheres = await cursor.fetchall()

    sphere_labels = [row[0] for row in spheres]
    sphere_counts = [row[1] for row in spheres]

    await message.answer(
        f"📊 Статистика:\n"
        f"👥 Всего пользователей: {total_users}\n"
        f"⏳ Ожидают подбора: {waiting_users}\n"
        f"👥 Групп: {total_groups}\n"
        f"📈 Распределение по сферам см. в графике:"
    )

@dp.message(F.text & F.chat.type == ChatType.PRIVATE)
async def handle_bio_or_sphere(message: Message, state: FSMContext):
    data = await state.get_data()
    current_state = data.get("state")

    if current_state == "awaiting_bio":
        await upsert_user(
            message.from_user.id,
            message.from_user.first_name,
            message.from_user.username,
            message.text,
        )
        await message.answer("Теперь отправьте свою фотографию (портрет или аватар):")
        await state.update_data(state="awaiting_photo")

    elif current_state == "awaiting_sphere":
        sphere = message.text.strip()
        await set_user_sphere_waiting(message.from_user.id, sphere)
        await message.answer(
            f"Вы зарегистрированы для участия! ☕️ Подбор пар будет выполнен в понедельник в 9:00 по Москве."
        )
        await state.clear()

@dp.message(F.photo & F.chat.type == ChatType.PRIVATE)
async def handle_photo(message: Message, state: FSMContext):
    data = await state.get_data()
    current_state = data.get("state")
    if current_state == "awaiting_photo":
        file_id = message.photo[-1].file_id
        await save_user_photo(message.from_user.id, file_id)
        await message.answer("Введите вашу сферу деятельности:")
        await state.update_data(state="awaiting_sphere")

@dp.poll_answer()
async def handle_poll_answer(poll_answer: PollAnswer):
    user_id = poll_answer.user.id
    poll_id = poll_answer.poll_id
    option = poll_answer.option_ids[0] if poll_answer.option_ids else None
    chat_data = poll_data.get(poll_id)

    if chat_data:
        chat_id = chat_data["chat_id"]
        if option == 0:
            await add_group_user(chat_id, user_id)
            if not await is_user_registered(user_id):
                try:
                    await bot.send_message(
                        chat_id=user_id,
                        text="Привет! Ты выбрал участие в поиске пар в группе, но еще не зарегистрирован. Напиши /start в личных сообщениях, чтобы пройти регистрацию и участвовать в подборе пар!"
                    )
                except Exception as exc:
                    logging.error(f"Failed to send registration prompt to {user_id}: {exc}")
            else:
                await set_user_waiting_status(user_id, 1)
        elif option == 1:
            await remove_group_user(chat_id, user_id)
            await set_user_waiting_status(user_id, 0)
        poll_data.pop(poll_id, None)
    else:
        if option == 0:
            await set_user_waiting_status(user_id, 1)
        elif option == 1:
            await set_user_waiting_status(user_id, 0)

# ---------------------------------------------------------------------------
# Pairing logic
# ---------------------------------------------------------------------------

async def create_pairs_and_notify():
    users = await get_waiting_users()
    if len(users) < 2:
        logging.info("Недостаточно пользователей для подбора пар (требуется минимум 2).")
        return

    pairs = []
    used_ids = set()

    # Группируем пользователей по схожим сферам
    sphere_groups = {}
    for u in users:
        sphere = u["sphere"].lower().strip()
        if not sphere:
            continue
        matched = False
        for key in sphere_groups:
            if is_similar(sphere, key):
                sphere_groups[key].append(u)
                matched = True
                break
        if not matched:
            sphere_groups[sphere] = [u]

    # Формируем пары внутри каждой группы
    for sphere, group in sphere_groups.items():
        random.shuffle(group)
        while len(group) >= 2:
            u1 = group.pop()
            u2 = group.pop()
            pairs.append((u1, u2))
            used_ids.update([u1["tg_id"], u2["tg_id"]])

    # Отправка уведомлений
    paired_ids = []
    quests = [
        "Обсудите свой самый провальный проект и чему он вас научил.",
        "Придумайте вместе идею стартапа за 15 минут.",
        "Расскажите друг другу о любимом фильме и почему он вас зацепил.",
        "Поделитесь смешной историей, связанной с вашей работой или учёбой.",
        "Угадайте, чем занимается ваш собеседник, не задавая прямых вопросов о профессии.",
    ]

    pair_info = []
    for u1, u2 in pairs:
        quest = random.choice(quests)
        paired_ids.extend([u1["tg_id"], u2["tg_id"]])
        pair_info.append((u1, u2, quest))

        try:
            u1_name = clean_string(u1["username"] or u1["name"])
            u2_name = clean_string(u2["username"] or u2["name"])
            u2_sphere = clean_string(u2["sphere"] or "Не указана")
            u2_bio = clean_string(u2["bio"] or "Не указана")
            u1_sphere = clean_string(u1["sphere"] or "Не указана")
            u1_bio = clean_string(u1["bio"] or "Не указана")
            await bot.send_photo(
                chat_id=u1["tg_id"],
                photo=u2.get("photo_id"),
                caption=(
                    f"🎉 Ваша пара: @{u2_name}\n"
                    f"📛 Имя: {u2_name}\n"
                    f"💼 Сфера: {u2_sphere}\n"
                    f"ℹ️ О себе: {u2_bio}\n"
                    f"🧹 Задание: {clean_string(quest)}"
                ),
            )
            u1_last_paired = u1.get("last_paired")
            if not u1_last_paired or datetime.fromisoformat(u1_last_paired) < datetime.utcnow() - timedelta(hours=24):
                await bot.send_poll(
                    chat_id=u1["tg_id"],
                    question="Участвуешь в следующем поиске пар?",
                    options=["Да", "Нет"],
                    is_anonymous=False,
                    allows_multiple_answers=False,
                )
            await bot.send_photo(
                chat_id=u2["tg_id"],
                photo=u1.get("photo_id"),
                caption=(
                    f"🎉 Ваша пара: @{u1_name}\n"
                    f"📛 Имя: {u1_name}\n"
                    f"💼 Сфера: {u1_sphere}\n"
                    f"ℹ️ О себе: {u1_bio}\n"
                    f"🧹 Задание: {clean_string(quest)}"
                ),
            )
            u2_last_paired = u2.get("last_paired")
            if not u2_last_paired or datetime.fromisoformat(u2_last_paired) < datetime.utcnow() - timedelta(hours=24):
                await bot.send_poll(
                    chat_id=u2["tg_id"],
                    question="Участвуешь в следующем поиске пар?",
                    options=["Да", "Нет"],
                    is_anonymous=False,
                    allows_multiple_answers=False,
                )
        except Exception as exc:
            logging.error(f"Send fail to user {u1['tg_id']} or {u2['tg_id']}: {exc}")

    group_chats = await get_groups()
    for chat_id in group_chats:
        group_users = await get_group_users(chat_id)
        if not group_users:
            continue

        relevant_pairs = [
            (u1, u2, quest)
            for u1, u2, quest in pair_info
            if u1["tg_id"] in group_users or u2["tg_id"] in group_users
        ]
        if not relevant_pairs:
            continue

        group_message = "🎉 Сформированы пары для нетворкинга:\n\n"
        for u1, u2, quest in relevant_pairs:
            u1_name = clean_string(u1["username"] or u1["name"])
            u2_name = clean_string(u2["username"] or u2["name"])
            pair_text = f"@{u1_name} + @{u2_name} — Задание: {clean_string(quest)}\n"
            group_message += pair_text

        try:
            await bot.send_message(chat_id=chat_id, text=group_message)
            logging.info(f"Sent group message to chat_id={chat_id}")
        except Exception as exc:
            logging.error(f"Failed to send to group {chat_id}: {exc}")

    await mark_users_paired(paired_ids)
    logging.info(f"Pairs created: {len(pairs)}")

scheduler = AsyncIOScheduler(timezone=TIMEZONE)
scheduler.add_job(
    create_pairs_and_notify,
    CronTrigger(day_of_week="mon", hour=9, minute=0),
    name="Weekly pairing",
)

async def main():
    await init_db()
    scheduler.start()
    logging.info("Mactracher Bot started…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")