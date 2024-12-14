import telebot
import threading
import time

class TelegramBot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token, parse_mode=None)
        self.setup_handlers()

    def setup_handlers(self):
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            self.bot.reply_to(message, "Howdy, how are you doing?")

    def send_message(self, user_id, message):
        try:
            self.bot.send_message(user_id, message)
        except Exception as e:
            print(f"Error sending message: {e}")

    def run(self):
        polling_thread = threading.Thread(target=self.bot.infinity_polling)
        polling_thread.daemon = True
        polling_thread.start()