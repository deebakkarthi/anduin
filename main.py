#!/usr/bin/env python3
import sys
import os
import json
import requests

HEADERS = {
    "Authorization": f"Bearer {os.environ.get('UVARC_GenAI_API')}",
    "Content-Type": "application/json",
}
UVARC_GENAI_ENDPOINT = "https://open-webui.rc.virginia.edu/api/chat/completions"


# API return schema for reference
# https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events?site_locale=en
def resp_to_chat(r: requests.Response):
    chat = {"reasoning": "", "content": ""}
    for line in r.iter_lines(decode_unicode=True):
        # Skip empty lines
        if not line:
            continue
        if line.startswith("data: "):
            data = line.removeprefix("data: ").strip()

            # End marker
            if data == "[DONE]":
                break

            # Convert to json
            data = json.loads(data)
            # choices is an array of CompletionChoice
            # Usually this is a singleton but for the sake of completeness
            # I'm iterating over it
            for choice in data.get("choices", []):
                # Delta is a originally a str|None
                # If it is indeed a str, then the content will be one of the
                # following:
                #   - Role token { "role": "assistant", "content" : "" }
                #   - Reasoning token { "reasoning": "..." }
                #   - Content token { "content": "..." }
                # json.loads(data) converts this str to a dict for us
                delta = choice.get("delta", {})
                if "role" in delta:
                    # don't do anything of the Role token
                    continue
                elif "reasoning" in delta:
                    chat["reasoning"] += delta.get("reasoning", "")
                elif "content" in delta and "role" not in delta:
                    chat["content"] += delta.get("content", "")
                    print(delta.get("content", ""), end="", flush=True)
    return chat


def main() -> None:
    messages=[]
    while True:
        user_prompt=input()
        if user_prompt == "/exit":
            break
        messages.append({ "role": "user",
            "content": user_prompt
            })
        with requests.post( UVARC_GENAI_ENDPOINT, headers=HEADERS,
            # Passing a dict through the json parameter automatically calls
            # json.dumps() on it and encodes it.
            json={
                "model": "Kimi K2.5",
                "messages": messages,
                # Send stream as json too
                "stream": True,
                "stream_options": {"include_usage": True}
            },
            # Stream in chunks
            stream=True,
            ) as r:
            # Raise exception incase of a bad request (4xxs)
            r.raise_for_status()
            chat = resp_to_chat(r)
            print()
            messages.append({"role":"assistant", "content": chat["content"]})
    sys.exit(0)


if __name__ == "__main__":
    main()
