from model_loader import load_model
from chat_memory import ChatMemory

def main():
    generator = load_model("distilgpt2")  # or "gpt2" for slightly bigger
    memory = ChatMemory(max_turns=5)

    print("🤖 Local CLI Chatbot (type '/exit' to quit)\n")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "/exit":
            print("Exiting chatbot. Goodbye!")
            break

        # Build prompt with memory context
        context = memory.get_context()
        prompt = f"{context}\nUser: {user_input}\nBot:"

        # Generate response
        response = generator(prompt, max_length=200, do_sample=True, top_p=0.95, temperature=0.7)[0]['generated_text']

        # Extract latest reply after prompt
        bot_reply = response[len(prompt):].split("\n")[0].strip()

        print(f"Bot: {bot_reply}\n")

        memory.add_message(user_input, bot_reply)

if __name__ == "__main__":
    main()
