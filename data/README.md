# CrazeAI training data

Put the existing database at \`data/database.jsonl\` to make the preparation script use it automatically.

One JSON object per line is required. Preferred schema:

    {"messages":[{"role":"user","content":"..."},
                 {"role":"assistant","content":"..."}]}

The legacy instruction schema is also accepted:

    {"instruction":"...","input":"","output":"..."}

Only text-only system/user/assistant messages are retained. Empty, malformed, oversized and duplicate records are rejected. The script adds the CrazeAI system identity if a record does not already contain a system message.

The current repository history did not contain a 50K database; the old committed JSON file had only 22 entries. If no local database is present, the builder streams 49,988 examples from the configured SmolTalk subsets and adds the 12 CrazeAI seed examples. The resulting total is exactly 50,000 records. It writes separate train/validation JSONL files and a manifest with source counts and SHA-256 hashes.

Do not commit private, copyrighted or personally identifying training data. Check the license and provenance of any local database before training or publishing model weights.
