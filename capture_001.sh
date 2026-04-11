curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": "Return only one word: EVEN or ODD. Question: A box contains 7 red balls and 8 blue balls. Two balls are removed without replacement. Is the probability that both removed balls are blue greater than 1/4?"
      }
    ],
    "logprobs": true,
    "top_logprobs": 20,
    "temperature": 0,
    "seed": 42,
    "max_tokens": 3
  }' > data/openai_raw_capture_001.json