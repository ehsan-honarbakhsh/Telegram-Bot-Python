from cambridge import camb
import cambridge.camb
import telebot
import requests
import logging
import os

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

API_TOKEN = os.environ.get("API_TOKEN")
bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(
        message.chat.id, "Welcome to dictionary  Bot")

@bot.message_handler(func= lambda message:True)
def look_up(message):
    word=message.text
    response=camb(word)
    bot.send_message(chat_id=message.chat.id,reply_to_message_id=message.id,text=response)


bot.infinity_polling()