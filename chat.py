from web_reader import read_website
from ai_brain import WebAI

ai = WebAI()
memory = []

print("🌐 Web AI Ready (type exit to quit)")

while True:
    user = input("\nYou: ")
    if user.lower() == "exit":
        break

    if user.startswith("read "):
        url = user.replace("read ", "")
        print("🔍 Reading website...")
        text = read_website(url)
        ai.add_text(text)
        print("✅ Website learned!")
    else:
        response = ai.ask(user)
        memory.append((user, response))
        print("\nAI:", response[:1000])  # limit output
