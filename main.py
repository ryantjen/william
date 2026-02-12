from agent import ask_agent

print("William initialized. Hello sire!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = ask_agent(user_input)
    print("\nCopilot:", response, "\n")
