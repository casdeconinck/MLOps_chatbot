import chainlit as cl
import os
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

model_id = "openai-community/gpt2-medium"
conv_model = HuggingFaceHub(huggingfacehub_api_token=os.environ.get('HUGGINGFACE_API_TOKEN'),
                            repo_id=model_id,
                            model_kwargs={"max_new_tokens": 150})
#"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"
add_llm_provider(
    LangchainGenericProvider(
        id=conv_model._llm_type,
        name="HuggingFaceHub",
        llm=conv_model,
        is_chat=False
    )
)

template = """{query} is delicious and you can prepare it like this:"""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=['query'])
    conv_chain = LLMChain(llm=conv_model,
                          prompt=prompt,
                          verbose=True)
    cl.user_session.set("llm_chain", conv_chain)

@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    if isinstance(message, cl.message.Message):
        message_text = message.content
    else:
        message_text = message
    res = await llm_chain.acall(message_text, callbacks=[cl.AsyncLangchainCallbackHandler()])

    response_text = res.get("text", "Error processing your request")
    await cl.Message(content=response_text).send()