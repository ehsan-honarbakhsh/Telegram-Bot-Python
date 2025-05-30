import telebot
import subprocess
import re
import requests
import os
import time
import logging
import json
import threading
import html
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from gtts import gTTS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.users import User
from models.tasks import Task
from models import Base
import asyncio
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import transformers and numpy
try:
    from transformers import pipeline
    import numpy
    TRANSFORMERS_AVAILABLE = True
    logger.info(f"NumPy version: {numpy.__version__}, Path: {numpy.__file__}")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    numpy = None
    logger.error(f"Failed to import transformers or numpy: {str(e)}")

# Import googletrans
try:
    from googletrans import Translator, LANGUAGES
    GOOGLETRANS_AVAILABLE = True
    logger.info("Googletrans imported successfully")
except ImportError as e:
    GOOGLETRANS_AVAILABLE = False
    Translator = None
    LANGUAGES = {}
    logger.error(f"Failed to import googletrans: {str(e)}")

# Initialize Telegram bot
API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise ValueError("API_TOKEN environment variable not set")
bot = telebot.TeleBot(API_TOKEN)

# Initialize PostgreSQL database
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
logger.info("Database initialized")

# Initialize GPT-2
gpt2_generator = None
if TRANSFORMERS_AVAILABLE:
    try:
        gpt2_generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")
        logger.info("GPT-2 model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load GPT-2 model: {str(e)}")
        gpt2_generator = None
else:
    logger.error("Transformers or numpy not available. GPT-2 feature disabled.")

# Initialize Google Translate
translator = None
if GOOGLETRANS_AVAILABLE:
    try:
        translator = Translator()
        logger.info("Google Translate initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Google Translate: {str(e)}")
        translator = None
else:
    logger.error("Googletrans not available. Translation feature disabled.")

# State management
user_states = {}
message_history = {}

def get_main_keyboard():
    """Create the main reply keyboard."""
    markup = ReplyKeyboardMarkup(resize_keyboard=True, input_field_placeholder="Choose a Service")
    markup.add(KeyboardButton('Convert Text to Sound'))
    markup.add(KeyboardButton('Cambridge Dictionary'))
    markup.add(KeyboardButton('To-Do List'))
    if gpt2_generator:
        markup.add(KeyboardButton('Ask a Question'))
    if translator:
        markup.add(KeyboardButton('Translate to Persian'))
        markup.add(KeyboardButton('Translate to English'))
    markup.add(KeyboardButton('Clear Chat'))
    return markup

def get_todo_keyboard():
    """Create the to-do list sub-menu keyboard."""
    markup = ReplyKeyboardMarkup(resize_keyboard=True, input_field_placeholder="Choose a To-Do Action")
    markup.add(KeyboardButton('Add Task'))
    markup.add(KeyboardButton('View Tasks'))
    markup.add(KeyboardButton('Delete Task'))
    markup.add(KeyboardButton('Back to Main Menu'))
    return markup

def clean_ansi_codes(text):
    """Remove ANSI escape codes."""
    ansi_regex = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_regex.sub('', text)
    cleaned = cleaned.replace('\t', ' ').replace('\r', '')
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = re.sub(r'\n+', '\n', cleaned).strip()
    logger.info(f"Raw text before ANSI cleaning:\n{repr(text)}")
    logger.info(f"Text after ANSI cleaning:\n{repr(cleaned)}")
    return cleaned

def parse_cambridge_output(raw_output):
    """Parse Cambridge CLI output."""
    lines = raw_output.splitlines()
    result = {"definitions": []}
    seen_definitions = set()
    current_definition = None
    current_pos = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        logger.info(f"Parsing line: {repr(line)}")
        if re.match(r'^\s*(verb|noun|adjective|adverb|pronoun|preposition|conjunction|interjection)\s*$', line, re.IGNORECASE):
            current_pos = line.strip().capitalize()
            logger.info(f"Detected part of speech: {current_pos}")
            continue
        if (line.startswith(':') or (':' in line and '|' not in line and not line.startswith('uk') and not line.startswith('us'))):
            meaning = line.split(':', 1)[-1].strip()
            meaning = re.sub(r'\s*\[.*?\]\s*$', '', meaning).strip()
            if (meaning and len(meaning) >= 10 and ' ' in meaning and not meaning.startswith('(') and not meaning.startswith('[')):
                normalized_meaning = re.sub(r'\s+', ' ', meaning.lower()).strip()
                if normalized_meaning not in seen_definitions:
                    current_definition = {"meaning": meaning, "examples": [], "pos": current_pos}
                    result["definitions"].append(current_definition)
                    seen_definitions.add(normalized_meaning)
                    logger.info(f"Added definition: {meaning} (POS: {current_pos})")
                else:
                    logger.info(f"Skipped duplicate definition: {meaning}")
            else:
                logger.info(f"Skipped invalid definition: {meaning}")
        if line.startswith('|') and current_definition is not None:
            example = line.split('|', 1)[-1].strip()
            example = re.sub(r'\s+', ' ', example).strip()
            if example and not example.startswith('[') and len(example) >= 10:
                current_definition["examples"].append(example)
                logger.info(f"Added example: {example}")
            else:
                logger.info(f"Skipped invalid example: {example}")

    result["definitions"] = [
        d for d in result["definitions"]
        if d["meaning"] and len(d["meaning"]) >= 10 and (d["examples"] or d["meaning"].lower() not in ["to cause to be perfect"])
    ]
    return result

def escape_markdown_v2(text):
    """Escape special characters for Telegram MarkdownV2."""
    if not text:
        return text
    special_chars = r'_*[]()~`>#+-=|{}.!'
    escaped = ''
    for char in text:
        if char in special_chars:
            escaped += '\\' + char
        else:
            escaped += char
    return escaped

def format_for_telegram(data, word, use_html=False):
    """Format parsed data for Telegram."""
    if not data["definitions"]:
        return f"No definitions found for '{word}'."
    
    if use_html:
        output = []
        current_pos = None
        for i, defn in enumerate(data["definitions"], 1):
            pos = defn.get("pos", "Unknown")
            if pos != current_pos:
                output.append(f"\n{pos}:")
                current_pos = pos
            output.append(f"{i}. {defn['meaning']}")
            for example in defn["examples"]:
                output.append(f"   - {example}")
        return "\n".join(output).strip()
    else:
        output = []
        current_pos = None
        for i, defn in enumerate(data["definitions"], 1):
            pos = defn.get("pos", "Unknown")
            if pos != current_pos:
                output.append(f"\n*{escape_markdown_v2(pos)}*:")
                current_pos = pos
            meaning = escape_markdown_v2(defn['meaning'])
            output.append(f"{i}\\. {meaning}")
            for example in defn["examples"]:
                output.append(f"   \\- {escape_markdown_v2(example)}")
        return "\n".join(output).strip()

def lookup_word_fallback(word):
    """Fallback to web scraping if CLI fails."""
    time.sleep(1)
    url = f"https://dictionary.cambridge.org/dictionary/english/{word.replace(' ', '-')}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        data = {"definitions": []}
        seen_definitions = set()

        for entry in soup.select(".entry-body__el"):
            pos_elem = entry.select_one(".pos-header .pos")
            pos = pos_elem.get_text(strip=True).capitalize() if pos_elem else "Unknown"
            logger.info(f"Processing POS: {pos}")
            for def_block in entry.select(".def-block"):
                meaning_elem = def_block.select_one(".def")
                if not meaning_elem:
                    logger.info("No meaning found in def-block")
                    continue
                meaning = meaning_elem.get_text(separator=" ", strip=True)
                meaning = html.unescape(meaning)
                meaning = re.sub(r'\s+', ' ', meaning).strip()
                meaning = re.sub(r'[^\w\s.,;?!-]', '', meaning)
                normalized_meaning = meaning.lower().strip()
                if (not meaning or len(meaning) < 10 or ' ' not in meaning or
                        normalized_meaning in seen_definitions or
                        meaning.startswith('(') or meaning.startswith('[')):
                    logger.info(f"Skipped invalid/duplicate meaning: {meaning}")
                    continue
                examples = []
                for ex_elem in def_block.select(".examp"):
                    example = ex_elem.get_text(separator=" ", strip=True)
                    example = html.unescape(example)
                    example = re.sub(r'\s+', ' ', example).strip()
                    example = re.sub(r'[^\w\s.,;?!-]', '', example)
                    if example and len(example) >= 10 and not example.startswith('['):
                        examples.append(example)
                        logger.info(f"Added example: {example}")
                    else:
                        logger.info(f"Skipped invalid example: {example}")
                data["definitions"].append({
                    "meaning": meaning,
                    "examples": examples,
                    "pos": pos
                })
                seen_definitions.add(normalized_meaning)
                logger.info(f"Added definition: {meaning} (POS: {pos})")

        logger.info(f"Web scraped data for '{word}':\n{data}")
        return (format_for_telegram(data, word, use_html=False)
                if data["definitions"]
                else f"No definitions found for '{word}' (web fallback).")
    except requests.RequestException as e:
        error_msg = f"Web scraping error: {str(e)}"
        logger.error(error_msg)
        return error_msg

def clear_cambridge_cache(word):
    """Clear the cambridge CLI cache."""
    try:
        subprocess.run(['camb', 'l', '-d', word], capture_output=True, text=True, check=True)
        logger.info(f"Cleared cache for '{word}'")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clear cache for '{word}': {e.stderr}")

def lookup_word(word):
    """Look up a word using the cambridge CLI, with web scraping fallback."""
    clear_cambridge_cache(word)
    try:
        result = subprocess.run(
            ['camb', word],
            capture_output=True,
            text=True,
            check=True
        )
        clean_output = clean_ansi_codes(result.stdout)
        logger.info(f"Raw camb output for '{word}':\n{repr(clean_output)}")
        if not clean_output.strip():
            logger.error(f"Empty output from camb for '{word}'")
            return lookup_word_fallback(word)
        parsed_data = parse_cambridge_output(clean_output)
        logger.info(f"Parsed data for '{word}':\n{parsed_data}")
        if not parsed_data["definitions"] or not any(defn["examples"] for defn in parsed_data["definitions"]):
            logger.info(f"No definitions or examples found in CLI output for '{word}', trying web fallback")
            return lookup_word_fallback(word)
        formatted_output = format_for_telegram(parsed_data, word, use_html=False)
        return formatted_output if formatted_output else f"No definitions found for '{word}'."
    except subprocess.CalledProcessError as e:
        error_msg = f"CLI error: {e.stderr}"
        logger.error(error_msg)
        return lookup_word_fallback(word)
    except Exception as e:
        error_msg = f"Error processing word: {str(e)}"
        logger.error(error_msg)
        return lookup_word_fallback(word)

def generate_gpt2_response(question):
    """Generate a response using GPT-2."""
    if not gpt2_generator:
        return "Error: GPT-2 feature is unavailable due to missing dependencies."
    try:
        prompt = question.strip()
        if not prompt.endswith('?'):
            prompt += '?'
        response = gpt2_generator(
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            truncation=True
        )[0]['generated_text']
        response = response.replace(prompt, '').strip()
        if not response:
            response = "Sorry, I couldn't generate a meaningful answer."
        logger.info(f"GPT-2 response for '{question}': {response}")
        return response
    except Exception as e:
        error_msg = f"Error generating GPT-2 response: {str(e)}"
        logger.error(error_msg)
        return error_msg

async def translate_to_persian(text):
    """Translate text from English to Persian (asynchronous)."""
    if not translator:
        return escape_markdown_v2("Error: Google Translate feature is unavailable due to missing dependencies.")
    if not text.strip():
        return escape_markdown_v2("Error: No text provided for translation.")
    try:
        src_lang = 'en'
        dest_lang = 'fa'
        if src_lang not in LANGUAGES or dest_lang not in LANGUAGES:
            return escape_markdown_v2("Error: English ('en') or Persian ('fa') not supported.")
        translation = await translator.translate(text, src=src_lang, dest=dest_lang)
        result = (
            f"English: {text}\n"
            f"Persian: {translation.text}"
        )
        logger.info(f"Translated '{text}' from English to Persian: {translation.text}")
        return escape_markdown_v2(result)
    except Exception as e:
        error_msg = f"Error translating text to Persian: {str(e)}"
        logger.error(error_msg)
        return escape_markdown_v2(error_msg)

async def translate_to_english(text):
    """Translate text from Persian to English (asynchronous)."""
    if not translator:
        return escape_markdown_v2("Error: Google Translate feature is unavailable due to missing dependencies.")
    if not text.strip():
        return escape_markdown_v2("Error: No text provided for translation.")
    try:
        src_lang = 'fa'
        dest_lang = 'en'
        if src_lang not in LANGUAGES or dest_lang not in LANGUAGES:
            return escape_markdown_v2("Error: Persian ('fa') or English ('en') not supported.")
        translation = await translator.translate(text, src=src_lang, dest=dest_lang)
        result = (
            f"Persian: {text}\n"
            f"English: {translation.text}"
        )
        logger.info(f"Translated '{text}' from Persian to English: {translation.text}")
        return escape_markdown_v2(result)
    except Exception as e:
        error_msg = f"Error translating text to English: {str(e)}"
        logger.error(error_msg)
        return escape_markdown_v2(error_msg)

def manage_user(chat_id, username, first_name, last_name):
    """Create or update a user in the database."""
    session = Session()
    try:
        user = session.query(User).filter_by(chat_id=chat_id).first()
        if user:
            # Update existing user
            user.username = username
            user.first_name = first_name
            user.last_name = last_name
            logger.info(f"Updated user: chat_id={chat_id}, username={username}")
        else:
            # Create new user
            user = User(
                chat_id=chat_id,
                username=username,
                first_name=first_name,
                last_name=last_name
            )
            session.add(user)
            logger.info(f"Created new user: chat_id={chat_id}, username={username}")
        session.commit()
        return user
    except Exception as e:
        logger.error(f"Error managing user (chat_id={chat_id}): {str(e)}")
        session.rollback()
        return None
    finally:
        session.close()

def add_task(chat_id, task_text):
    """Add a task with optional due date to the database."""
    session = Session()
    try:
        # Get or create user
        user = session.query(User).filter_by(chat_id=chat_id).first()
        if not user:
            return "Error: User not found. Please use /start first."

        # Parse task
        parts = task_text.split(',', 1)
        description = parts[0].strip()
        due_date = None
        now = datetime.now()
        if len(parts) > 1:
            due_str = parts[1].strip().lower()
            try:
                if re.match(r'in\s*\d+\s*minute(s)?', due_str):
                    minutes = int(re.search(r'\d+', due_str).group())
                    due_date = now + timedelta(minutes=minutes)
                elif re.match(r'in\s*\d+\s*hour(s)?', due_str):
                    hours = int(re.search(r'\d+', due_str).group())
                    due_date = now + timedelta(hours=hours)
                elif re.match(r'in\s*\d+\s*day(s)?', due_str):
                    days = int(re.search(r'\d+', due_str).group())
                    due_date = now + timedelta(days=days)
                elif re.match(r'today\s*\d{1,2}:\d{2}', due_str):
                    time_str = re.search(r'\d{1,2}:\d{2}', due_str).group()
                    due_date = datetime.strptime(f"{now.date()} {time_str}", "%Y-%m-%d %H:%M")
                elif re.match(r'tomorrow\s*\d{1,2}:\d{2}', due_str):
                    time_str = re.search(r'\d{1,2}:\d{2}', due_str).group()
                    due_date = datetime.strptime(f"{now.date() + timedelta(days=1)} {time_str}", "%Y-%m-%d %H:%M")
                else:
                    due_date = datetime.strptime(due_str, "%Y-%m-%d %H:%M")
                logger.info(f"Parsed due date: {due_date}")
            except ValueError:
                return "Error: Invalid due date. Use 'in 2 hours', 'tomorrow 14:00', 'today 18:00', or 'YYYY-MM-DD HH:MM'."
        
        if not description:
            return "Error: Task description cannot be empty."
        
        # Create task
        task = Task(
            user_id=user.id,
            description=description,
            due_date=due_date,
            notified=False
        )
        session.add(task)
        session.commit()
        logger.info(f"Added task for chat {chat_id}: {description}")
        response = f"Task added: {escape_markdown_v2(description)}" + (f" \\(Due: {escape_markdown_v2(due_date.isoformat())}\\)" if due_date else "")
        return response
    except Exception as e:
        logger.error(f"Error adding task for chat {chat_id}: {str(e)}")
        session.rollback()
        return f"Error adding task: {str(e)}"
    finally:
        session.close()

def view_tasks(chat_id):
    """View all tasks for a user from the database."""
    session = Session()
    try:
        user = session.query(User).filter_by(chat_id=chat_id).first()
        if not user:
            return "Error: User not found. Please use /start first."
        
        tasks = session.query(Task).filter_by(user_id=user.id).all()
        if not tasks:
            return "No tasks found."
        
        output = ["*Your Tasks*:"]
        for i, task in enumerate(tasks, 1):
            due = f" \\(Due: {escape_markdown_v2(task.due_date.isoformat())}\\)" if task.due_date else ""
            output.append(f"{i}\\. {escape_markdown_v2(task.description)}{due}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Error viewing tasks for chat {chat_id}: {str(e)}")
        return f"Error viewing tasks: {str(e)}"
    finally:
        session.close()

def delete_task(chat_id, index):
    """Delete a task by index from the database."""
    session = Session()
    try:
        user = session.query(User).filter_by(chat_id=chat_id).first()
        if not user:
            return "Error: User not found. Please use /start first."
        
        tasks = session.query(Task).filter_by(user_id=user.id).all()
        if not tasks:
            return "No tasks to delete."
        
        try:
            index = int(index) - 1
            if 0 <= index < len(tasks):
                task = tasks[index]
                session.delete(task)
                session.commit()
                logger.info(f"Deleted task for chat {chat_id}: {task.description}")
                return f"Deleted task: {escape_markdown_v2(task.description)}"
            else:
                return "Error: Invalid task index."
        except ValueError:
            return "Error: Please provide a valid number."
    except Exception as e:
        logger.error(f"Error deleting task for chat {chat_id}: {str(e)}")
        session.rollback()
        return f"Error deleting task: {str(e)}"
    finally:
        session.close()

async def task_reminder_loop():
    """Background task for reminders using the database."""
    logger.info("Starting task reminder loop")
    while True:
        try:
            now = datetime.now().replace(microsecond=0)
            logger.info(f"Checking tasks at {now}")
            session = Session()
            try:
                tasks = session.query(Task).filter(Task.due_date != None, Task.notified == False).all()
                for task in tasks:
                    if task.due_date and now >= task.due_date:
                        try:
                            user = session.query(User).filter_by(id=task.user_id).first()
                            if user:
                                msg = bot.send_message(
                                    chat_id=user.chat_id,
                                    text=f"Task due: {escape_markdown_v2(task.description)}",
                                    parse_mode="MarkdownV2"
                                )
                                if user.chat_id not in message_history:
                                    message_history[user.chat_id] = []
                                message_history[user.chat_id].append(msg.message_id)
                                message_history[user.chat_id] = message_history[user.chat_id][-10:]
                                task.notified = True
                                session.commit()
                                logger.info(f"Sent Telegram reminder for task: {task.description} (chat {user.chat_id})")
                        except telebot.apihelper.ApiTelegramException as e:
                            logger.error(f"Failed to send reminder to chat {user.chat_id}: {str(e)}")
                        except Exception as e:
                            logger.error(f"Unexpected error sending reminder for task {task.description}: {str(e)}")
            except Exception as e:
                logger.error(f"Error querying tasks in reminder loop: {str(e)}")
            finally:
                session.close()
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Error in task reminder loop: {str(e)}")
            await asyncio.sleep(10)

def clear_chat(chat_id):
    """Delete recent bot messages."""
    if chat_id in message_history:
        for msg_id in message_history[chat_id]:
            try:
                bot.delete_message(chat_id, msg_id)
                logger.info(f"Deleted message {msg_id} in chat {chat_id}")
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to delete message {msg_id} in chat {chat_id}: {str(e)}")
        message_history[chat_id].clear()

def send_long_message(bot, chat_id, reply_to_message_id, text, parse_mode):
    """Split and send long messages."""
    if not text:
        text = "No response available."
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    if chat_id not in message_history:
        message_history[chat_id] = []
    for chunk in chunks:
        try:
            msg = bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=reply_to_message_id,
                text=chunk,
                parse_mode=parse_mode,
                reply_markup=get_main_keyboard()
            )
            message_history[chat_id].append(msg.message_id)
            message_history[chat_id] = message_history[chat_id][-10:]
        except Exception as e:
            logger.error(f"Failed to send message with {parse_mode}: {str(e)}")
            if parse_mode == "MarkdownV2":
                msg = bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=reply_to_message_id,
                    text=chunk,
                    parse_mode="HTML",
                    reply_markup=get_main_keyboard()
                )
                message_history[chat_id].append(msg.message_id)
                message_history[chat_id] = message_history[chat_id][-10:]
            else:
                msg = bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=reply_to_message_id,
                    text=chunk,
                    reply_markup=get_main_keyboard()
                )
                message_history[chat_id].append(msg.message_id)
                message_history[chat_id] = message_history[chat_id][-10:]

@bot.message_handler(commands=['start'])
def welcome(message):
    """Handle /start command."""
    chat_id = message.chat.id
    user = message.from_user
    user_states[chat_id] = None
    if chat_id not in message_history:
        message_history[chat_id] = []
    
    # Manage user in database
    manage_user(
        chat_id=chat_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name
    )
    
    # Escape the entire message for MarkdownV2
    message_text = escape_markdown_v2(
        f"Hi {user.first_name or 'User'}! Welcome to WordWave Bot. Choose a service:"
    )
    msg = bot.send_message(
        chat_id,
        message_text,
        parse_mode="MarkdownV2",
        reply_markup=get_main_keyboard()
    )
    message_history[chat_id].append(msg.message_id)
    message_history[chat_id] = message_history[chat_id][-10:]
    logger.info(f"Welcome message sent to chat_id={chat_id}")

@bot.message_handler(commands=['test_reminder'])
def test_reminder(message):
    """Send a test reminder."""
    try:
        if message.chat.id not in message_history:
            message_history[message.chat.id] = []
        msg = bot.send_message(
            message.chat.id,
            escape_markdown_v2("Test reminder!"),
            parse_mode="MarkdownV2"
        )
        message_history[message.chat.id].append(msg.message_id)
        message_history[message.chat.id] = message_history[message.chat.id][-10:]
        logger.info(f"Sent test reminder to chat {message.chat.id}")
    except Exception as e:
        logger.error(f"Failed to send test reminder to chat {message.chat.id}: {str(e)}")
        msg = bot.send_message(
            message.chat.id,
            f"Error sending test reminder: {str(e)}"
        )
        message_history[message.chat.id].append(msg.message_id)
        message_history[message.chat.id] = message_history[message.chat.id][-10:]

@bot.message_handler(func=lambda message: message.text == "Cambridge Dictionary")
def prompt_dictionary(message):
    """Prompt for dictionary lookup."""
    user_states[message.chat.id] = "dictionary"
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        "Please enter a word to look up in the Cambridge Dictionary.",
        reply_markup=get_main_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} selected dictionary mode")

@bot.message_handler(func=lambda message: message.text == "Convert Text to Sound")
def prompt_text_to_speech(message):
    """Prompt for text-to-speech."""
    user_states[message.chat.id] = "text_to_speech"
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        "Please enter the text you want to convert to speech.",
        reply_markup=get_main_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} selected text-to-speech mode")

@bot.message_handler(func=lambda message: message.text == "Ask a Question")
def prompt_gpt2(message):
    """Prompt for GPT-2 question."""
    user_states[message.chat.id] = "gpt2"
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        escape_markdown_v2("Please ask a question, and I'll answer using GPT-2!"),
        parse_mode="MarkdownV2",
        reply_markup=get_main_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} selected GPT-2 question mode")

@bot.message_handler(func=lambda message: message.text == "Translate to Persian")
def prompt_translate_to_persian(message):
    """Prompt for Persian translation."""
    user_states[message.chat.id] = "translate_to_persian"
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        "Please enter English text to translate to Persian.",
        reply_markup=get_main_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} selected translate to Persian mode")

@bot.message_handler(func=lambda message: message.text == "Translate to English")
def prompt_translate_to_english(message):
    """Prompt for English translation."""
    user_states[message.chat.id] = "translate_to_english"
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        "Please enter Persian text to translate to English.",
        reply_markup=get_main_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} selected translate to English mode")

@bot.message_handler(func=lambda message: message.text == "To-Do List")
def prompt_todo(message):
    """Prompt for to-do list action."""
    user_states[message.chat.id] = "todo"
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        "Choose a to-do list action:",
        reply_markup=get_todo_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} selected to-do list mode")

@bot.message_handler(func=lambda message: message.text == "Add Task")
def prompt_add_task(message):
    """Prompt to add a task."""
    user_states[message.chat.id] = "add_task"
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        "Enter a task (e.g., 'Buy groceries', 'Buy groceries, in 2 hours', 'Buy groceries, tomorrow 14:00', or 'Buy groceries, 2025-04-25 14:00'):",
        reply_markup=get_todo_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} selected add task mode")

@bot.message_handler(func=lambda message: message.text == "View Tasks")
def handle_view_tasks(message):
    """Handle view tasks action."""
    response = view_tasks(message.chat.id)
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        response,
        parse_mode="MarkdownV2",
        reply_markup=get_todo_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} viewed tasks")

@bot.message_handler(func=lambda message: message.text == "Delete Task")
def prompt_delete_task(message):
    """Prompt to delete a task."""
    user_states[message.chat.id] = "delete_task"
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        "Enter the task number to delete (use 'View Tasks' to see numbers):",
        reply_markup=get_todo_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} selected delete task mode")

@bot.message_handler(func=lambda message: message.text == "Back to Main Menu")
def back_to_main_menu(message):
    """Return to main menu."""
    user_states[message.chat.id] = None
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        "Choose a service:",
        reply_markup=get_main_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} returned to main menu")

@bot.message_handler(func=lambda message: message.text == "Clear Chat")
def handle_clear_chat(message):
    """Clear recent bot messages."""
    clear_chat(message.chat.id)
    if message.chat.id not in message_history:
        message_history[message.chat.id] = []
    msg = bot.send_message(
        message.chat.id,
        escape_markdown_v2("Chat cleared!"),
        parse_mode="MarkdownV2",
        reply_markup=get_main_keyboard()
    )
    message_history[message.chat.id].append(msg.message_id)
    message_history[message.chat.id] = message_history[message.chat.id][-10:]
    logger.info(f"User {message.chat.id} cleared chat")

@bot.message_handler(func=lambda message: True)
def handle_input(message):
    """Handle user input based on mode."""
    chat_id = message.chat.id
    state = user_states.get(chat_id, None)
    text = message.text.strip()

    if not state:
        if chat_id not in message_history:
            message_history[chat_id] = []
        msg = bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=message.id,
            text="Please choose a service first using the buttons.",
            reply_markup=get_main_keyboard()
        )
        message_history[chat_id].append(msg.message_id)
        message_history[chat_id] = message_history[chat_id][-10:]
        return

    if state == "dictionary":
        if not text:
            if chat_id not in message_history:
                message_history[chat_id] = []
            msg = bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide a word to look up.",
                reply_markup=get_main_keyboard()
            )
            message_history[chat_id].append(msg.message_id)
            message_history[chat_id] = message_history[chat_id][-10:]
            return
        response = lookup_word(text)
        try:
            logger.info(f"Sending dictionary response for '{text}':\n{response}")
            send_long_message(bot, chat_id, message.id, response, "MarkdownV2")
        except Exception as e:
            logger.error(f"Markdown failed: {str(e)}")
            try:
                parsed_data = parse_cambridge_output(clean_ansi_codes(subprocess.run(['camb', text], capture_output=True, text=True, check=True).stdout)) if not response.startswith("Web scraping error") else {"definitions": []}
                response_html = format_for_telegram(parsed_data or {"definitions": []}, text, use_html=True)
                logger.info(f"Sending HTML response for '{text}':\n{response_html}")
                send_long_message(bot, chat_id, message.id, response_html, "HTML")
            except Exception as e2:
                if chat_id not in message_history:
                    message_history[chat_id] = []
                error_msg = f"Error sending message: {str(e2)}"
                logger.error(error_msg)
                msg = bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=message.id,
                    text=error_msg,
                    reply_markup=get_main_keyboard()
                )
                message_history[chat_id].append(msg.message_id)
                message_history[chat_id] = message_history[chat_id][-10:]

    elif state == "text_to_speech":
        if not text:
            if chat_id not in message_history:
                message_history[chat_id] = []
            msg = bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide text to convert to speech.",
                reply_markup=get_main_keyboard()
            )
            message_history[chat_id].append(msg.message_id)
            message_history[chat_id] = message_history[chat_id][-10:]
            return
        try:
            if chat_id not in message_history:
                message_history[chat_id] = []
            file_name = f"voices/output_{chat_id}_{int(time.time())}.mp3"
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            output = gTTS(text=text, lang="en", tld='com.au')
            output.save(file_name)
            with open(file_name, "rb") as voice_file:
                msg1 = bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=message.id,
                    text="Generating voice...",
                    reply_markup=get_main_keyboard()
                )
                message_history[chat_id].append(msg1.message_id)
                msg2 = bot.send_voice(
                    chat_id=chat_id,
                    reply_to_message_id=message.id,
                    voice=voice_file,
                    reply_markup=get_main_keyboard()
                )
                message_history[chat_id].append(msg2.message_id)
                message_history[chat_id] = message_history[chat_id][-10:]
            os.remove(file_name)
            logger.info(f"Sent voice message for '{text}' to {chat_id}")
        except Exception as e:
            if chat_id not in message_history:
                message_history[chat_id] = []
            error_msg = f"Error converting text to speech: {str(e)}"
            logger.error(error_msg)
            msg = bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text=error_msg,
                reply_markup=get_main_keyboard()
            )
            message_history[chat_id].append(msg.message_id)
            message_history[chat_id] = message_history[chat_id][-10:]

    elif state == "gpt2":
        if not text:
            if chat_id not in message_history:
                message_history[chat_id] = []
            msg = bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please ask a question.",
                reply_markup=get_main_keyboard()
            )
            message_history[chat_id].append(msg.message_id)
            message_history[chat_id] = message_history[chat_id][-10:]
            return
        response = generate_gpt2_response(text)
        send_long_message(bot, chat_id, message.id, response, "MarkdownV2")

    elif state == "translate_to_persian":
        if not text:
            if chat_id not in message_history:
                message_history[chat_id] = []
            msg = bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide English text to translate to Persian.",
                reply_markup=get_main_keyboard()
            )
            message_history[chat_id].append(msg.message_id)
            message_history[chat_id] = message_history[chat_id][-10:]
            return
        response = asyncio.run(translate_to_persian(text))
        send_long_message(bot, chat_id, message.id, response, "MarkdownV2")

    elif state == "translate_to_english":
        if not text:
            if chat_id not in message_history:
                message_history[chat_id] = []
            msg = bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide Persian text to translate to English.",
                reply_markup=get_main_keyboard()
            )
            message_history[chat_id].append(msg.message_id)
            message_history[chat_id] = message_history[chat_id][-10:]
            return
        response = asyncio.run(translate_to_english(text))
        send_long_message(bot, chat_id, message.id, response, "MarkdownV2")

    elif state == "add_task":
        if not text:
            if chat_id not in message_history:
                message_history[chat_id] = []
            msg = bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide a task description.",
                reply_markup=get_todo_keyboard()
            )
            message_history[chat_id].append(msg.message_id)
            message_history[chat_id] = message_history[chat_id][-10:]
            return
        response = add_task(chat_id, text)
        if chat_id not in message_history:
            message_history[chat_id] = []
        msg = bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=message.id,
            text=response,
            parse_mode="MarkdownV2",
            reply_markup=get_todo_keyboard()
        )
        message_history[chat_id].append(msg.message_id)
        message_history[chat_id] = message_history[chat_id][-10:]

    elif state == "delete_task":
        if not text:
            if chat_id not in message_history:
                message_history[chat_id] = []
            msg = bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide the task number to delete.",
                reply_markup=get_todo_keyboard()
            )
            message_history[chat_id].append(msg.message_id)
            message_history[chat_id] = message_history[chat_id][-10:]
            return
        response = delete_task(chat_id, text)
        if chat_id not in message_history:
            message_history[chat_id] = []
        msg = bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=message.id,
            text=response,
            parse_mode="MarkdownV2",
            reply_markup=get_todo_keyboard()
        )
        message_history[chat_id].append(msg.message_id)
        message_history[chat_id] = message_history[chat_id][-10:]

# Start the bot and task reminder loop
if __name__ == "__main__":
    logger.info("Bot is running...")
    try:
        # Start the task reminder loop in a separate thread
        def run_reminder_loop():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(task_reminder_loop())
            finally:
                loop.close()
        threading.Thread(target=run_reminder_loop, daemon=True).start()
        bot.infinity_polling(none_stop=True)
    except Exception as e:
        logger.error(f"Error in bot polling: {str(e)}")