import warnings
warnings.filterwarnings('ignore')

import json
import random
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load intents dataset
def load_json_file(filename):
    with open(filename) as f:
        file = json.load(f)
    return file

filename = 'intents_exp.json'
intents = load_json_file(filename)

# Load mappings
label2id = {
    'greeting': 0, 'goodbye': 1, 'creator': 2, 'name': 3, 'hours': 4, 
    'number': 5, 'course': 6, 'fees': 7, 'location': 8, 'hostel': 9, 
    'event': 10, 'document': 11, 'floors': 12, 'syllabus': 13, 'library': 14, 
    'infrastructure': 15, 'canteen': 16, 'menu': 17, 'placement': 18, 
    'ithod': 19, 'computerhod': 20, 'extchod': 21, 'principal': 22, 'sem': 23, 
    'admission': 24, 'scholarship': 25, 'facilities': 26, 'college intake': 27, 
    'uniform': 28, 'committee': 29, 'random': 30, 'swear': 31, 'vacation': 32, 
    'sports': 33, 'salutation': 34, 'task': 35, 'ragging': 36, 'hod': 37
}
id2label = {v: k for k, v in label2id.items()}

# Load trained model and tokenizer
model_path = "chatbot_model11.h5"
model = load_model(model_path)

tokenizer_path = "tokenizer11.pkl"
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Preprocess user message
def preprocess_message(message):
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=model.input_shape[1], padding='post')
    return padded

# Predict function
def predict_label(message):
    padded_message = preprocess_message(message)
    prediction = model.predict(padded_message).argmax(axis=1)
    return id2label[prediction[0]]

# Define command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hi! I'm your chatbot. Send me a message, and I'll respond.")

async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    pred_tag = predict_label(user_message)
    
    # Get the prediction score
    padded_message = preprocess_message(user_message)
    prediction_scores = model.predict(padded_message)
    score = prediction_scores.max()

    if score > 0.7:
        responses = [intent['responses'] for intent in intents['intents'] if intent['tag'] == pred_tag]
        response = random.choice(responses[0]) if responses else "I'm sorry, I didn't understand that."
        await update.message.reply_text(f"{response}\n {pred_tag} - {score}")
    else:
        await update.message.reply_text("I'm sorry, I didn't understand that.\n")

    # if score < 0.7:
    #     response = f"I'm sorry, I didn't understand that. (Score: {score:.2f})"
    # else:
    #     # Find response for the predicted tag
    #     responses = [intent['responses'] for intent in intents['intents'] if intent['tag'] == pred_tag]
    #     response = random.choice(responses[0]) if responses else "I'm sorry, I didn't understand that."

    # Send response with label and score
    # await update.message.reply_text(f"Label: {pred_tag}, Score: {score:.2f}\nResponse: {response}")

# Main function to set up the bot
def main():
    application = Application.builder().token("").build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))

    application.run_polling()

if __name__ == '__main__':
    main()
