import json
import os
from typing import Dict

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.agent import ReActAgent
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from pydantic import BaseModel

from code_reader import code_reader

load_dotenv()

parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_dir="./data", file_extractor=file_extractor).load_data()
embed_model = resolve_embed_model(embed_model="local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model)

llm = Ollama(model="mistral", request_timeout=60 * 60)
query_engine = vector_index.as_query_engine(llm=llm)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives documentation about code for an API. Use this for reading docs for the API",
        ),
    ),
    code_reader,
]

code_llm = Ollama(model="codellama", request_timeout=60 * 60)

context = """Purpose: The primary role of this agent is to assist users by analyzing code. It should
            be able to generate code and answer questions about code provided. """
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)


class CodeOutput(BaseModel):
    """Role of this class is to provide the json format for the code prompt template"""
    code: str
    description: str
    filename: str


parser = PydanticOutputParser(output_cls=CodeOutput)
code_parser_template = """Parse the response from a previous LLM into a description and a string of valid code, 
                            also come up with a valid filename this could be saved as that doesnt contain special characters. 
                            Here is the response: {response}. You should parse this in the following JSON Format: """
json_prompt_str: str = parser.format(code_parser_template)
json_prompt_tmpl: PromptTemplate = PromptTemplate(json_prompt_str)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(str_or_query_bundle=prompt)
            # Format the prompt, call the LLM
            code_prompt: str = json_prompt_tmpl.format(response=str(result))
            # Use the code itself as the new prompt
            next_result = llm.complete(prompt=code_prompt)

            # Convert the string expression of the dictionary to a dictionary.
            cleaned_json: Dict = json.loads(s=str(next_result))
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}:", e)

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    print("Code generated")
    print(cleaned_json["code"])
    print("\n\nDesciption:", cleaned_json["description"])

    filename = cleaned_json["filename"]

    try:
        out_path: str = 'output'
        os.makedirs(name=out_path, exist_ok=True)
        with open(os.path.join(out_path, filename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except OSError:
        print("Error saving file...")
