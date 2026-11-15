from __future__ import annotations

import argparse

from crazeai.runtime import CrazeAIRuntime


def main():
    parser = argparse.ArgumentParser(description="Chat with CrazeAI's own pretrained base model.")
    parser.add_argument("--model", default=None, help="Local scratch-trained checkpoint directory")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    runtime = CrazeAIRuntime(args.model)
    print(f"CrazeAI scratch model loaded from {runtime.model_name} on {runtime.device}.")
    print("Base pretraining teaches text patterns; instruction-following requires a later stage.")
    print("Type /exit to quit.")
    while True:
        try:
            prompt = input("\\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if prompt:
            answer, _, _ = runtime.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=args.max_new_tokens,
            )
            print("\\nCrazeAI:", answer)


if __name__ == "__main__":
    main()
