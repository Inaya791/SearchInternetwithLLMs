import gradio as gr
from langchain.adapters.openai import convert_openai_messages
from langchain_community.chat_models import ChatOpenAI
from tavily import TavilyClient

# Set up the TavilyClient and OpenAI API keys
TAVILY_API_KEY = "xxx"
OPENAI_API_KEY = "xxx"

client = TavilyClient(api_key=TAVILY_API_KEY)

def generate_report(query):
    # Step 2: Executing the search query and getting the results
    content = client.search(query, search_depth="advanced")["results"]

    # Step 3: Setting up the OpenAI prompts
    prompt = [{
        "role": "system",
        "content": f'You are an AI critical thinker research assistant. '
                   f'Your sole purpose is to write well written, critically acclaimed,'
                   f'objective and structured reports on given text.'
    }, {
        "role": "user",
        "content": f'Information: """{content}"""\n\n'
                   f'Using the above information, answer the following'
                   f'query: "{query}" in a detailed report --'
                   f'Please use MLA format and markdown syntax.'
    }]

    # Step 4: Running OpenAI through Langchain
    lc_messages = convert_openai_messages(prompt)
    report = ChatOpenAI(model='gpt-4', openai_api_key=OPENAI_API_KEY).invoke(lc_messages).content

    return report

# Set up Gradio interface
iface = gr.Interface(
    fn=generate_report,
    inputs="text",
    outputs="text",
    title="Tavily With OpenAI LLM",
    description="Enter your query to get a detailed research report using AI."
)

if __name__ == "__main__":
    iface.launch()
