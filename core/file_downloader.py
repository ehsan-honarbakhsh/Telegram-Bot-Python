import telebot
import os
import pprint
import json
import logging
import requests
#Logging produce a separated thread , instead of printing everything,use logger
logger=telebot.logging
telebot.logger.setLevel(logging.DEBUG)

API_TOKEN =os.environ.get("API_TOKEN") 
bot = telebot.TeleBot(API_TOKEN)


if not os.path.exists("downloads"):
    os.makedirs("downloads")
DOWNLOAD_DIR = "downloads/"

@bot.message_handler(commands=['start'])
def send_welcome(message):
    logger.info("The game is started!")
    bot.send_message(message.chat.id,"Welcome to my bot downloader,Please provide the url link:")

def download_file(url):
    local_filename = url.split('/')[-1]
    file_path = DOWNLOAD_DIR + local_filename
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return file_path


@bot.message_handler(func= lambda message:True)
def download_file_url(message):
    logger.info(message.text)
    url = message.text
    try:
        file_path = download_file(url)
        bot.send_document(chat_id=message.chat.id, reply_to_message_id=message.id, document=open(
            file_path, "rb"), caption="file downloaded successfully, ENJOY!")
        os.remove(file_path)
    except:
        bot.reply_to(message, text="problem downloading the requested file")


bot.infinity_polling()