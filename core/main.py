import telebot
import subprocess
import re
import requests
import os
import time
import logging
import asyncio
from bs4 import BeautifulSoup
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from gtts import gTTS

# Setup logging before any logger usage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import transformers and numpy after logger setup
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

# Import googletrans after logger setup
try:
    from googletrans import Translator, LANGUAGES
    GOOGLETRANS_AVAILABLE = True
    logger.info("Googletrans imported successfully")
except ImportError as e:
    GOOGLETRANS_AVAILABLE = False
    Translator = None
    LANGUAGES = {}
    logger.error(f"Failed to import googletrans: {str(e)}")

# Initialize the Telegram bot using environment variable
API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise ValueError("API_TOKEN environment variable not set")
bot = telebot.TeleBot(API_TOKEN)

# Initialize GPT-2 pipeline (loaded once to save memory)
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

# Initialize a single event loop for async translations
loop = None
if GOOGLETRANS_AVAILABLE:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        logger.info("Event loop initialized for async translations")
    except Exception as e:
        logger.error(f"Failed to initialize event loop: {str(e)}")
        loop = None

# State management to track user mode
user_states = {}

def get_keyboard():
    """Create the reply keyboard with service buttons."""
    markup = ReplyKeyboardMarkup(resize_keyboard=True, input_field_placeholder="Choose a Service")
    markup.add(KeyboardButton('Convert Text to Sound'))
    markup.add(KeyboardButton('Cambridge Dictionary'))
    if gpt2_generator:
        markup.add(KeyboardButton('Ask a Question'))
    if translator:
        markup.add(KeyboardButton('Translate to Persian'))
        markup.add(KeyboardButton('Translate to English'))
    return markup

def clean_ansi_codes(text):
    """Remove ANSI escape codes while preserving spaces and punctuation."""
    ansi_regex = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_regex.sub('', text)
    # Preserve single spaces and line breaks, avoid collapsing multiple spaces
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n+', '\n', cleaned).strip()
    logger.info(f"Raw text before ANSI cleaning:\n{repr(text)}")
    logger.info(f"Text after ANSI cleaning:\n{repr(cleaned)}")
    return cleaned

def parse_cambridge_output(raw_output):
    """Parse the raw Cambridge CLI output to extract definitions and examples."""
    lines = raw_output.splitlines()
    result = {
        "definitions": []
    }
    seen_definitions = set()  # Track unique definitions to avoid duplicates
    current_definition = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        logger.info(f"Parsing line: {repr(line)}")
        
        # Extract definitions (require meaningful content)
        if (line.startswith(':') or (':' in line and '|' not in line and not line.startswith('uk') and not line.startswith('us'))):
            meaning = line.split(':', 1)[-1].strip()
            meaning = re.sub(r'\s*\[.*?\]\s*$', '', meaning).strip()
            # Ensure definition is meaningful (at least 5 characters and contains a space or punctuation)
            if (meaning and len(meaning) >= 5 and (' ' in meaning or any(c in meaning for c in '.!?:')) and not meaning.startswith('(')):
                normalized_meaning = re.sub(r'\s+', ' ', meaning.lower()).strip()
                if normalized_meaning not in seen_definitions:
                    current_definition = {"meaning": meaning, "examples": []}
                    result["definitions"].append(current_definition)
                    seen_definitions.add(normalized_meaning)
                    logger.info(f"Added definition: {meaning}")
                else:
                    logger.info(f"Skipped duplicate definition: {meaning}")
            else:
                logger.info(f"Skipped invalid definition: {meaning}")
        
        # Extract examples
        if line.startswith('|') and current_definition is not None:
            example = line.split('|')[-1].strip()
            example = re.sub(r'\s+', ' ', example).strip()
            if example and not example.startswith('[') and len(example) >= 5:
                current_definition["examples"].append(example)
                logger.info(f"Added example: {example}")
            else:
                logger.info(f"Skipped invalid example: {example}")

    return result

def escape_markdown_v2(text):
    """Escape special characters for Telegram MarkdownV2, preserving spaces."""
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
    """Format the parsed data as a Telegram-friendly message with proper spacing."""
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
            meaning = escape_markdown_v2(defn['meaning'])
            output.append(f"{i}\\. {meaning}")
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
        seen_definitions = set()
        for def_block in soup.select(".def-block"):
            meaning_elem = def_block.select_one(".def.ddef_d")
            meaning = meaning_elem.get_text(strip=True) if meaning_elem else ""
            meaning = re.sub(r'\s+', ' ', meaning).strip()
            normalized_meaning = meaning.lower()
            examples = [
                re.sub(r'\s+', ' ', ex.get_text(strip=True)).strip()
                for ex in def_block.select(".examp.dexamp")
                if ex.get_text(strip=True) and len(ex.get_text(strip=True)) >= 5
            ]
            if meaning and len(meaning) >= 5 and normalized_meaning not in seen_definitions:
                data["definitions"].append({"meaning": meaning, "examples": examples})
                seen_definitions.add(normalized_meaning)
                logger.info(f"Web scraped definition: {meaning}")
            else:
                logger.info(f"Skipped invalid or duplicate web definition: {meaning}")
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
    """Generate a response to a question using GPT-2."""
    if not gpt2_generator:
        return "Error: GPT-2 feature is unavailable due to missing dependencies (e.g., numpy or transformers). Please contact the bot administrator."
    
    try:
        # Ensure the question ends with a question mark for better context
        prompt = question.strip()
        if not prompt.endswith('?'):
            prompt += '?'
        # Generate response with explicit truncation
        response = gpt2_generator(
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            truncation=True
        )[0]['generated_text']
        # Clean up response (remove prompt and extra whitespace)
        response = response.replace(prompt, '').strip()
        if not response:
            response = "Sorry, I couldn't generate a meaningful answer."
        logger.info(f"GPT-2 response for '{question}': {response}")
        return response
    except Exception as e:
        error_msg = f"Error generating GPT-2 response: {str(e)}"
        logger.error(error_msg)
        return error_msg

async def async_translate(text, src_lang, dest_lang):
    """Helper function to perform asynchronous translation."""
    try:
        translation = await translator.translate(text, src=src_lang, dest=dest_lang)
        return translation
    except Exception as e:
        raise Exception(f"Async translation failed: {str(e)}")

def translate_to_persian(text):
    """Translate text from English to Persian using Google Translate."""
    if not translator:
        return "Error: Google Translate feature is unavailable due to missing dependencies (e.g., googletrans)."
    
    if not text.strip():
        return "Error: No text provided for translation."
    
    if not loop:
        return "Error: Event loop unavailable for async translation."
    
    try:
        # Validate language codes
        src_lang = 'en'
        dest_lang = 'fa'
        if src_lang not in LANGUAGES or dest_lang not in LANGUAGES:
            return "Error: English ('en') or Persian ('fa') not supported by Google Translate."
        # Perform translation using the event loop
        translation = loop.run_until_complete(async_translate(text, src_lang, dest_lang))
        result = (
            f"English: {text}\n"
            f"Persian: {translation.text}"
        )
        logger.info(f"Translated '{text}' from English to Persian: {translation.text}")
        return result
    except Exception as e:
        error_msg = f"Error translating text to Persian: {str(e)}"
        logger.error(error_msg)
        return error_msg

def translate_to_english(text):
    """Translate text from Persian to English using Google Translate."""
    if not translator:
        return "Error: Google Translate feature is unavailable due to missing dependencies (e.g., googletrans)."
    
    if not text.strip():
        return "Error: No text provided for translation."
    
    if not loop:
        return "Error: Event loop unavailable for async translation."
    
    try:
        # Validate language codes
        src_lang = 'fa'
        dest_lang = 'en'
        if src_lang not in LANGUAGES or dest_lang not in LANGUAGES:
            return "Error: Persian ('fa') or English ('en') not supported by Google Translate."
        # Perform translation using the event loop
        translation = loop.run_until_complete(async_translate(text, src_lang, dest_lang))
        result = (
            f"Persian: {text}\n"
            f"English: {translation.text}"
        )
        logger.info(f"Translated '{text}' from Persian to English: {translation.text}")
        return result
    except Exception as e:
        error_msg = f"Error translating text to English: {str(e)}"
        logger.error(error_msg)
        return error_msg

def send_long_message(bot, chat_id, reply_to_message_id, text, parse_mode):
    """Split and send long messages to avoid Telegram's character limit."""
    if not text:
        text = "No response available."
    if parse_mode == "MarkdownV2":
        text = escape_markdown_v2(text)
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=reply_to_message_id,
                text=chunk,
                parse_mode=parse_mode,
                reply_markup=get_keyboard()
            )
        except Exception as e:
            logger.error(f"Failed to send message with {parse_mode}: {str(e)}")
            if parse_mode == "MarkdownV2":
                bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=reply_to_message_id,
                    text=chunk,
                    parse_mode="HTML",
                    reply_markup=get_keyboard()
                )
            else:
                bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=reply_to_message_id,
                    text=chunk,
                    reply_markup=get_keyboard()
                )

@bot.message_handler(commands=['start'])
def welcome(message):
    """Handle the /start command and show the service menu."""
    user_states[message.chat.id] = None
    bot.send_message(
        message.chat.id,
        "Hi! Welcome to WordWave Bot. Choose a service:",
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

@bot.message_handler(func=lambda message: message.text == "Ask a Question")
def prompt_gpt2(message):
    """Prompt the user to enter a question for GPT-2."""
    user_states[message.chat.id] = "gpt2"
    bot.send_message(
        message.chat.id,
        "Please ask a question, and I'll answer using GPT-2!",
        reply_markup=get_keyboard()
    )
    logger.info(f"User {message.chat.id} selected GPT-2 question mode")

@bot.message_handler(func=lambda message: message.text == "Translate to Persian")
def prompt_translate_to_persian(message):
    """Prompt the user to enter English text for translation to Persian."""
    user_states[message.chat.id] = "translate_to_persian"
    bot.send_message(
        message.chat.id,
        "Please enter English text to translate to Persian.",
        reply_markup=get_keyboard()
    )
    logger.info(f"User {message.chat.id} selected translate to Persian mode")

@bot.message_handler(func=lambda message: message.text == "Translate to English")
def prompt_translate_to_english(message):
    """Prompt the user to enter Persian text for translation to English."""
    user_states[message.chat.id] = "translate_to_english"
    bot.send_message(
        message.chat.id,
        "Please enter Persian text to translate to English.",
        reply_markup=get_keyboard()
    )
    logger.info(f"User {message.chat.id} selected translate to English mode")

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
                bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=message.id,
                    text="Generating voice...",
                    reply_markup=get_keyboard()
                )
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

    elif state == "gpt2":
        if not text:
            bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please ask a question.",
                reply_markup=get_keyboard()
            )
            return
        response = generate_gpt2_response(text)
        send_long_message(bot, chat_id, message.id, response, "MarkdownV2")

    elif state == "translate_to_persian":
        if not text:
            bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide English text to translate to Persian.",
                reply_markup=get_keyboard()
            )
            return
        response = translate_to_persian(text)
        send_long_message(bot, chat_id, message.id, response, "MarkdownV2")

    elif state == "translate_to_english":
        if not text:
            bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=message.id,
                text="Please provide Persian text to translate to English.",
                reply_markup=get_keyboard()
            )
            return
        response = translate_to_english(text)
        send_long_message(bot, chat_id, message.id, response, "MarkdownV2")

# Start the bot
if __name__ == "__main__":
    logger.info("Bot is running...")
    try:
        bot.infinity_polling(none_stop=True)
    finally:
        # Clean up: close the event loop if it exists
        if loop and not loop.is_closed():
            loop.close()
            logger.info("Event loop closed")