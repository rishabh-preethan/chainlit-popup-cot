import chainlit as cl
import groq
import logging
import json
import time
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the GROQ client with your API key
client = groq.Groq(api_key="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")  # Replace with your actual API key

# Set the model name
MODEL_NAME = "llama-3.1-70b-versatile"

async def make_api_call(messages, max_tokens, is_final_answer=False, custom_client=None):
    # Use the provided custom_client if available, otherwise fallback to the global client
    client_to_use = custom_client if custom_client is not None else client

    loop = asyncio.get_event_loop()  # Get the current event loop

    for attempt in range(3):
        try:
            if is_final_answer:
                # Run the API call in an executor to prevent blocking the event loop
                response = await loop.run_in_executor(
                    None,
                    lambda: client_to_use.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.2
                    )
                )
                return response.choices[0].message.content
            else:
                # Run the API call in an executor to prevent blocking the event loop
                response = await loop.run_in_executor(
                    None,
                    lambda: client_to_use.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {
                        "title": "Error",
                        "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"
                    }
                else:
                    return {
                        "title": "Error",
                        "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                        "next_action": "final_answer"
                    }
            await asyncio.sleep(1)  # Wait for 1 second before retrying

async def generate_response(prompt, custom_client=None):
    messages = [
        {
            "role": "system",
            "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue"
}```
"""
        },
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."
        }
    ]
    
    step_count = 1
    total_thinking_time = 0

    while True:
        start_time = time.time()
        step_data = await make_api_call(messages, 300, custom_client=custom_client)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        # Yield each step as it is generated
        yield (f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time)

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data.get('next_action') == 'final_answer' or step_count > 25:
            break

        step_count += 1

    # Generate final answer
    messages.append({
        "role": "user",
        "content": "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. Only provide the text response without any titles or preambles. Retain any formatting as instructed by the original prompt, such as exact formatting for free response or multiple choice."
    })

    start_time = time.time()
    final_data = await make_api_call(messages, 1200, is_final_answer=True, custom_client=custom_client)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    yield ("Final Answer", final_data, thinking_time)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome to the AI Assistant! Please type your question below to get started.").send()

@cl.on_message
async def on_message(message):
    async for step_title, step_content, thinking_time in generate_response(message.content):
        # Send each reasoning step to the user
        await cl.Message(content=f"**{step_title}**\n\n{step_content}").send()

if __name__ == "__main__":
    cl.run()
