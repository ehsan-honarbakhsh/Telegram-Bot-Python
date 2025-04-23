import telebot
import subprocess
import re
import requests
import os
import time
import logging
from bs4 import BeautifulSoup
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from gtts import gTTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Telegram bot using environment variable
API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise ValueError("API_TOKEN environment variable not set")
bot = telebot.TeleBot(API_TOKEN)

# State management to track user mode
user_states = {}

def get_keyboard():
    """Create the reply keyboard with service buttons."""
    markup = ReplyKeyboardMarkup(resize_keyboard=True, input_field_placeholder="Choose a Service")
    markup.add(KeyboardButton('Convert Text to Sound'))
    markup.add(KeyboardButton('Cambridge Dictionary'))
    return markup

def clean_ansi_codes(text):
    """Remove all ANSI escape codes from the text."""
    ansi_regex = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_regex.sub('', text)
    logger.info(f"Before ANSI cleaning:\n{text}")
    logger.info(f"After ANSI cleaning:\n{cleaned}")
    return cleaned

def parse_cambridge_output(raw_output):
    """Parse the raw Cambridge CLI output to extract definitions and examples."""
    lines = raw_output.splitlines()
    result = {
        "definitions": []
    }
    
    current_definition = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        logger.info(f"Parsing line: {line}")
        
        # Extract definitions
        if line.startswith(':') or (':' in line and '|' not in line and not line.startswith('uk') and not line.startswith('us')):
            meaning = line.split(':', 1)[-1].strip()
            meaning = re.sub(r'\s*\[.*?\]\s*$', '', meaning).strip()  # Remove [informal], etc.
            if meaning and not meaning.startswith('('):
                current_definition = {"meaning": meaning, "examples": []}
                result["definitions"].append(current_definition)
                logger.info(f"Added definition: {meaning}")
        
        # Extract examples
        if line.startswith('|') and current_definition is not None:
            example = line.split('|')[-1].strip()
            if example and not example.startswith('['):
                current_definition["examples"].append(example)
                logger.info(f"Added example: {example}")

    return result

def escape_markdown_v2(text):
    """Escape special characters for Telegram MarkdownV2."""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

def format_for_telegram(data, word, use_html=False):
    """Format the parsed data as a Telegram-friendly message with only definitions and examples."""
    if not data["definitions"]:
        return f"No definitions found for '{word}'."
    
    if use_html:
        output = []
        for i, defn in enumerate(data["definitions"], 1):
            output.append(f"{i}. {defn['meaning']}")
            for example in defn["examples"]:
                output.append(f"   - {example}")
        return "\n".join(output)
    else:
        output = []
        for i, defn in enumerate(data["definitions"], 1):
            output.append(f"{i}\\. {escape_markdown_v2(defn['meaning'])}")
            for example in defn["examples"]:
                output.append(f"   \\- {escape_markdown_v2(example)}")
        return "\n".join(output)

def lookup_word_fallback(word):
    """Fallback to web scraping if CLI fails."""
    time.sleep(1)
    url = f"https://dictionary.cambridge.org/dictionary/english/{word.replace(' ', '-')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        data = {"definitions": []}
        for def_block in soup.select(".def-block"):
            meaning = def_block.select_one(".def.ddef_d").get_text(strip=True) if def_block.select_one(".def.ddef_d") else ""
            examples = [ex.get_text(strip=True) for ex in def_block.select(".examp.dexamp") if ex.get_text(strip=True)]
            if meaning:
                data["definitions"].append({"meaning": meaning, "examples": examples})
        logger.info(f"Web scraped data for '{word}':\n{data}")
        return format_for_telegram(data, word, use_html=False) if data["definitions"] else f"No definitions found for '{word}' (web fallback)."
    except requests.RequestException as e:
        error_msg = f"Web scraping error: {str(e)}"
        logger.error(error_msg)
        return error_msg

def clear_cambridge_cache(word):
    """Clear the cambridge CLI cache for a word."""
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
        logger.info(f"Raw camb output for '{word}':\n{clean_output}")
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

def send_long_message(bot, chat_id, reply_to_message_id, text, parse_mode):
    """Split and send long messages to avoid Telegram's character limit."""
    if not text:
        text = "No definitions found."
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
            text=chunk,
            parse_mode=parse_mode,
            reply_markup=get_keyboard()
        )

@bot.message_handler(commands=['start'])
def welcome(message):
    """Handle the /start command and show the service menu."""
    user_states[message.chat.id] = None
    bot.send_message(
        message.chat.id,
        "Hi! Welcome to my Telegram Bot. Choose a service:",
        reply_markup=get_keyboard()
    )
    logger.info("Welcome message sent")

@bot.message_handler(func=lambda message: message.text == "Cambridge Dictionary")
def prompt_dictionary(message):
    """Prompt the user to enter a word for dictionary lookup."""
    user_states[message.chat.id] = "dictionary"
    bot.send_message(
        message.chat.id,
        "Please enter a word to look up in the Cambridge Dictionary.",
        reply_markup=get_keyboard()
    )
    logger.info(f"User {message.chat.id} selected dictionary mode")

@bot.message_handler(func=lambda message: message.text == "Convert Text to Sound")
def prompt_text_to_speech(message):
    """Prompt the user to enter text for speech conversion."""
    user_states[message.chat.id] = "text_to_speech"
    bot.send_message(
        message.chat.id,
        "Please enter the text you want to convert to speech.",
        reply_markup=get_keyboard()
    )
    logger.info(f"User {message.chat.id} selected text-to-speech mode")

@bot.message_handler(func=lambda message: True)
def handle_input(message):
    """Handle user input based on their selected mode."""
    chat_id = message.chat.id
    state = user_states.get(chat_id, None)
    text = message.text.strip()

    if not state:
        bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=message.id,
            text="Please choose a service first using the buttons.",
            reply_markup=get_keyboard()
        )
        return

    if state == "dictionary":
        if not text:
            bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide a word to look up.",
                reply_markup=get_keyboard()
            )
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
                error_msg = f"Error sending message: {str(e2)}"
                logger.error(error_msg)
                bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=message.id,
                    text=error_msg,
                    reply_markup=get_keyboard()
                )

    elif state == "text_to_speech":
        if not text:
            bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide text to convert to speech.",
                reply_markup=get_keyboard()
            )
            return
        try:
            file_name = f"voices/output_{chat_id}_{int(time.time())}.mp3"
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            output = gTTS(text=text, lang="en", tld='com.au')
            output.save(file_name)
            with open(file_name, "rb") as voice_file:
                bot.send_voice(
                    chat_id=chat_id,
                    reply_to_message_id=message.id,
                    voice=voice_file,
                    reply_markup=get_keyboard()
                )
            os.remove(file_name)
            logger.info(f"Sent voice message for '{text}' to {chat_id}")
        except Exception as e:
            error_msg = f"Error converting text to speech: {str(e)}"
            logger.error(error_msg)
            bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text=error_msg,
                reply_markup=get_keyboard()
            )


bot.infinity_polling(none_stop=True)