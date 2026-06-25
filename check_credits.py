import os
import csv
from datetime import datetime
from dotenv import load_dotenv
import openai

# 1. Load the environment variables from your .env file
load_dotenv()

# Ensure the key is present
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")

# 2. Initialize the OpenAI client
# It automatically picks up the 'OPENAI_API_KEY' environment variable
client = openai.OpenAI()


def log_transaction_cost(model_name, prompt_tok, completion_tok, estimated_cost):
    """Appends transaction data and cost to a local CSV file."""
    log_file = "api_spend_log.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Create header if file is new
            writer.writerow(["Timestamp", "Model", "Prompt Tokens", "Completion Tokens", "Cost ($)"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            prompt_tok,
            completion_tok,
            f"{estimated_cost:.6f}"
        ])


try:
    current_model = "gpt-4o"

    # 3. Execute your API call
    response = client.chat.completions.create(
        model=current_model,
        messages=[
            {"role": "user", "content": "Write a quick Python script to parse a CSV file."}
        ]
    )

    # 4. Extract token data
    usage = response.usage
    p_tokens = usage.prompt_tokens
    c_tokens = usage.completion_tokens

    # 5. Define current model pricing (Per 1 Million Tokens)
    # Adjust these if you switch to gpt-4o-mini, o1, etc.
    INPUT_COST_PER_1M = 5.00
    OUTPUT_COST_PER_1M = 15.00

    # Calculate exact cost
    call_cost = ((p_tokens / 1_000_000) * INPUT_COST_PER_1M) + \
                ((c_tokens / 1_000_000) * OUTPUT_COST_PER_1M)

    # 6. Log it locally
    log_transaction_cost(current_model, p_tokens, c_tokens, call_cost)

    print(f"Success! Response received. Estimated cost: ${call_cost:.5f} (Logged to api_spend_log.csv)")
    print(f"Result: {response.choices[0].message.content[:100]}...")

except openai.RateLimitError:
    print("🚨 Out of funds or hit rate limits! Check your OpenAI developer dashboard.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")