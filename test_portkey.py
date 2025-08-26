from portkey_ai import Portkey

portkey = Portkey(
  base_url = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1",
  api_key = "NIbqgpk/0LIXtczDecd1UJYt5Tob"
)

response = portkey.chat.completions.create(
    model = "@o3-mini-156ab6/o3-mini",
    messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Portkey"}
    ],
    max_tokens = 512
)

print(response.choices[0].message.content)