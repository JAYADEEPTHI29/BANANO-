class ChatMemory:
    def __init__(self, max_turns=3):
        self.max_turns = max_turns
        self.history = []

    def add_message(self, user_input, bot_response):
        self.history.append(f"User: {user_input}\nBot: {bot_response}")
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self):
        return "\n".join(self.history)
