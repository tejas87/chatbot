import os

import streamlit as st


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


st.set_page_config(page_title="Hugging Face Chatbot")


@st.cache_resource(show_spinner="Loading Hugging Face model...")
def get_llm(model_id):
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_generator = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
    )

    return text_generator, tokenizer


def build_prompt(messages, tokenizer):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt_parts = []
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        prompt_parts.append(f"{role}: {message['content']}")
    prompt_parts.append("Assistant:")

    return "\n".join(prompt_parts)


def generate_assistant_response(messages, model_id, max_new_tokens, temperature):
    llm, tokenizer = get_llm(model_id)
    prompt = build_prompt(messages, tokenizer)

    response_output = llm(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    assistant_new_text = response_output[0]["generated_text"].strip()
    full_conversation_list = messages + [
        {"role": "assistant", "content": assistant_new_text}
    ]

    return assistant_new_text, full_conversation_list


if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("Hugging Face Chatbot")

with st.sidebar:
    st.header("Model")
    model_id = st.text_input(
        "Model ID",
        value=os.getenv("HF_MODEL_ID", DEFAULT_MODEL),
        help="Use any Hugging Face text-generation or instruct model.",
    )
    max_new_tokens = st.slider("Max new tokens", 32, 512, 160, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Type a message")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                assistant_text, updated_messages = generate_assistant_response(
                    st.session_state.messages,
                    model_id,
                    max_new_tokens,
                    temperature,
                )
            except Exception as exc:
                assistant_text = f"Model error: {exc}"
                updated_messages = st.session_state.messages + [
                    {"role": "assistant", "content": assistant_text}
                ]
        st.markdown(assistant_text)

    st.session_state.messages = updated_messages
