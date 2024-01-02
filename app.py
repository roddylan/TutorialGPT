import os 
from dotenv import load_dotenv


import streamlit as st
# from langchain.llms import openai
# from langchain.llms import llamacpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
# from langchain.utilities import wikipedia
from langchain.llms import huggingface_hub, ctransformers

load_dotenv()
key = os.getenv("HF_APIKEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = key

# create instance of llm
# llm = openai.OpenAI(temperature=0.9)
# llm = huggingface_hub.HuggingFaceHub(repo_id="TheBloke/dolphin-2.5-mixtral-8x7b-GPTQ",
#                                      model_kwargs={"temperature":1, "trust_remote_code":True})

# llm = ctransformers.CTransformers(model="dolphin-2.0-mistral-7b.Q4_K_M.gguf")
# llm = huggingface_hub.HuggingFaceHub(repo_id="cognitivecomputations/dolphin-2.2.1-mistral-7b",
#                                      model_kwargs={"temperature":1.2, "trust_remote_code":True})

llm = huggingface_hub.HuggingFaceHub(repo_id="cognitivecomputations/dolphin-2.2.1-mistral-7b",
                                     model_kwargs={"temperature":1.2, "trust_remote_code":True})

# llm = huggingface_hub.HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-v0.1",
#                                      model_kwargs={"temperature":0.9})


# App
st.title("Tutorial GPT")

prompt = st.text_input("Enter prompt here")

tutorial_template = PromptTemplate(
    input_variables=['title'],
    template='write a tutorial with the following title "{title}"'
)

# tutorial_template = PromptTemplate(
#     input_variables=['title'],
#     template='{title}'
# )

# memory
tutorial_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")




# llm chains
script_chain = LLMChain(llm=llm, prompt=tutorial_template, verbose=True, output_key="script", memory=tutorial_memory)

# wiki = wikipedia.WikipediaAPIWrapper()


if prompt:
    script = script_chain.run(prompt)
    st.write(script)
    
    with st.expander("History"):
        st.info(tutorial_memory.buffer)