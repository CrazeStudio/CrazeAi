from __future__ import annotations

import argparse

from crazeai.prompt import SYSTEM_PROMPT
from crazeai.runtime import CrazeAIRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with CrazeAI in the terminal.")
    parser.add_argument("--model", default=None, help="Local checkpoint directory or Hugging Face model ID")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    runtime = CrazeAIRuntime(args.model)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"CrazeAI loaded from {runtime.model_name} on {runtime.device}.")
    print("Type /exit to quit or /clear to reset the conversation.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            print("Goodbye.")
            break
        if user_text.lower() == "/clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation cleared.")
            continue

        history.append({"role": "user", "content": user_text})
        try:
            answer, _, _ = runtime.generate(
                history,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
        except Exception as exc:
            history.pop()
            print(f"CrazeAI error: {exc}")
            continue

        print(f"\nCrazeAI: {answer}")
        history.append({"role": "assistant", "content": answer})
        # Keep the system prompt and the most recent turns to bound memory.
        if len(history) > 17:
            history = [history[0]] + history[-16:]


if __name__ == "__main__":
    main()
