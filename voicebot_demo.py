import speech_recognition as sr
import whisper
import numpy as np
import tempfile
import os
import torch
import openai
import argparse


#read the txt file contains OpenAI API key
with open('openai_api_key.txt') as f:
    api_key = f.readline()
openai.api_key = api_key


def get_voice_command(timeout):
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        # adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        print("starts listening")
        try:
            # listen for speech for up to `timeout` seconds
            audio = r.listen(source, timeout)
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            return

        # Save audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(audio.get_wav_data())

        # Transcribe the temporary WAV file using Whisper
        result = audio_model.transcribe(temp_wav.name, fp16=torch.cuda.is_available())
        text = result['text'].strip()
        # Delete the temporary WAV file
        os.remove(temp_wav.name)

        return text
    


def ask_to_continue():
    while True:
        # Ask the user if they want to continue conversation
        continue_flag = input("Do you want to continue to edit? Please enter 'y' or 'n': ")
        if continue_flag == "y":
            return True # Return True to indicate the user wants to continue editing
            break
        elif continue_flag == "n":
            return False # Return False to indicate the user does not want to continue editing
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def get_SQL_query(timeout):
    # Define the system role to set the behavior of the chat assistant
    messages = [
        {"role": "system", "content" : "You're a data scientist helping me with writing SQL queries. I will ask for simple queries, and you will provide only the SQL query, no explanations."}
    ]
    while True:
        # Get the user's voice command
        command = get_voice_command(timeout)  
        # Add the user's command to the message history
        messages.append({"role": "user", "content": command})  
        # Generate a response from the chat assistant
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )  
        
        chat_response = completion.choices[0].message.content  # Extract the response from the completion
        print(f'ChatGPT: {chat_response}')  # Print the assistant's response
        messages.append({"role": "assistant", "content": chat_response})  # Add the assistant's response to the message history

        continue_flag = ask_to_continue()  # Prompt the user to continue or not
        if not continue_flag:
            return chat_response  # Return the final chat response and exit the loop
            break

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--timeout", default=3, type=float, help="Timeout for stopping transcription")
    args = parser.parse_args()
    model = args.model + ".en"
    audio_model = whisper.load_model(model)
    sql_query = get_SQL_query(args.timeout)
    print('GPT response:')
    print(f'{sql_query}')









