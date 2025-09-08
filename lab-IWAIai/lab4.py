from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader, PageObject
import gradio as gr

load_dotenv(override=True)
openai = OpenAI()

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

if pushover_user:
    print(f"Pushover user found and starts with {pushover_user[0]}")
else:
    print("Pushover user not found")

if pushover_token:
    print(f"Pushover token found and starts with {pushover_token[0]}")
else:
    print("Pushover token not found")

def push(message):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)

#push("Hello There! This is AI Agent Calling You.....")

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results

reader = PdfReader("IWAI_annual_report_2023_24.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if page:
        linkedin += text

with open("summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

name = "Inland Waterways Authority Of India"
website = "https://www.iwai.nic.in"

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s background, achievements, careers, projects and contribution. \
Your responsibility is to represent {name} for interactions on the {website} and {linkedin} as faithfully as possible. \
You are given a summary of {name}'s background, LinkedIn profile and website which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n## Website:\n{website}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

BASE_URL1 = "https://openrouter.ai/api/v1"
api_key1 = os.getenv("ROUTER_API_KEY")
router = OpenAI(base_url=BASE_URL1, api_key=api_key1)

def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    done = False
    while not done:

        # This is the call to the LLM - see that we pass in the tools json

        #response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        response = router.chat.completions.create(model="deepseek/deepseek-chat-v3.1:free", messages=messages, tools=tools)
        finish_reason = response.choices[0].finish_reason

        # If the LLM wants to call a tool, we do that!

        if finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True
    return response.choices[0].message.content

gr.ChatInterface(chat, type="messages").launch()


