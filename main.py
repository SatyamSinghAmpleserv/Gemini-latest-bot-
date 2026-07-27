from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage
)
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema import Document
from typing_extensions import TypedDict
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import Optional
from datetime import date
from pathlib import Path
import pandas as pd
import traceback
import requests
import pyodbc
import json
import uuid
import os

MAX_RETRIES = 2


BASE_DIR = Path(__file__).resolve().parent   #App directory
EXPORT_BASE = BASE_DIR / "exports"

embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


table_embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")



table_vector_store = FAISS.load_local(
            r"C:\Users\nitish.jha\Desktop\Gemini_chatbot\table_faiss_index",
            table_embedding_model,
            allow_dangerous_deserialization=True
        )


# API constants for HelpDesk knowledgebase
API_URL = "https://strapi.dcctz.com/helpdesk/api/chatbot-records/prepare?is_required_response=true"
HEADERS = {
    'Authorization':"Bearer a2abde4114364d2a5e05d4f7d9f9b1833bf8df21b9dbef2edf86547aad59fce65924e3f6e77aff97e4101a07231447f7733b5f12a681210e942525400d4468c2f8acd9cb0fd966ad1f574eac826f8d991a04e2fbdf85ab84bdbf0c1a57eb37005fe9f542ac67daad6a311f1a3292929c7cadb0fe41a1413733b407422ea9a2c2"
}



system_prompt = """
You're an intelligent assistant and official representative of the helpdesk portal named as "Aiva". Your role is to answer user queries accurately, clearly, helpfully and politely as if *you're helpdesk system itself*. Speak in friendly, professional tone and take ownership of the information you provide.

Whenever possible, refer users to the correct official resource or page. If the query is about process, policy, or status, answer directly and include the specific **Helpdesk URL** (if known or matched with the context).

Do not say *I am an AI* or *I was trained on..* - instead say *This portal provides..* or *You can find this under..*

Examples:
- Instead of: *"As an AI language model, I.."*
- Say: *"This helpdesk system allows you to.."*

If relevant content or links are fetched from the knowledge base, include them naturally in your response only once.
Do not modify, alter, innovate or create any new *URL* or *LINK*. 
If no information is available, politely inform the user this information could not be found. You may share **sapsupport@doubleclick.co.tz** contact information for further assistance only. 

Always aim to be clear, concise, polite and helpful.
""".strip()
    

prompt_template = PromptTemplate(
    input_variables=['context', 'question'],
    template="""
    {system_prompt}

    Context:
    {context}

    Question:
    {question}
    """.strip()
)

formatted_prompt = prompt_template.partial(system_prompt=system_prompt)



AGENT_PROMPT = """
# Agent System Prompt

You are **AIVA**, an AI assistant for DCC Logistics Suite.

Your responsibility is to determine whether a user's request requires access to the enterprise database.

## General Conversation

If the user's message is a greeting, farewell, thanks, introduction, small talk, or a general knowledge question that does not require company data, respond directly in a friendly, concise, and professional manner.

Examples include:

* Hello
* Hi
* Good morning
* Thank you
* Bye
* Who are you?
* What can you do?
* Tell me a joke.
* Explain Artificial Intelligence.
* What is RAG?
* What is LangGraph?

Do NOT call any tools for these requests.

---

## Database Queries

Call the **database_query_tool** whenever answering requires retrieving or analyzing information stored in the enterprise database.

Examples include requests related to:

* Vehicles
* Trucks
* Trailers
* Drivers
* Tyres
* Fleet
* Shipments
* Customers
* Orders
* Routes
* Workshops
* Files
* Invoices
* KPIs
* Reports
* Inventory
* Transport operations

Also call the tool when the user's request depends on previous database-related conversation.

Example:

User:
Show available trucks.

Assistant:
...

User:
Available drivers please.

Although the second request is short, it is a continuation of the database conversation and therefore requires the tool.

---

## Conversation Context

Always consider the complete conversation history before deciding.

If the current message depends on previous messages, use that context to determine whether database access is required.

---

## Tool Usage

If database information is required:

* Call database_query_tool.
* Do not answer from memory.
* Do not generate SQL yourself.

If database information is not required:

* Answer naturally without calling any tools.

Always be concise, accurate and professional.

"""



# Helpdesk query answering class
class HelpdeskResponse:

    def __init__(self, user_question, base_url, api_key):
        self.user_question = user_question
        self.base_url = base_url
        self.api_key = api_key

    def qa_chain_generation(self):
        try:

            response = requests.get(API_URL, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            # print(data)

        except Exception as e:
            return f"Didn't get the response from API (json). The error is {e}."
        print('Received the json data of helpdesk successfully.')
        

        actual_data = data['data']
        df = pd.DataFrame(actual_data)
        df['slugs'] = df['slugs'].apply(lambda x: f"{self.base_url}{x}")
        
        # Combine slugs + data
        combined_rows = [
            f"{slug}  \n\n{text}" for slug, text in zip(df['slugs'], df['data'])
        ]
        full_combined_text = ' ##### '.join(combined_rows)

        documents = Document(page_content=full_combined_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators="#####"
        )
        split_docs = text_splitter.split_documents([documents])

        retriever = FAISS.from_documents(split_docs, embedding_model).as_retriever(search_kwargs={'k': 3})


        qa_chain = ConversationalRetrievalChain.from_llm(
            llm = ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest",
         
            google_api_key=self.api_key
        ),
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": formatted_prompt}
        )
        print('QA Chain started successfully.')

        return qa_chain




# Table Retrieval function 
class TableRetrieval:
    def __init__(self, query, table_vector_store, schema_file):
        self.query = query
        self.table_vector_store = table_vector_store
        self.schema_file = schema_file


    def retrieve_table(self):
        print('Fetching tables')
        print(type(self.query))
        print(self.query)
        results = self.table_vector_store.similarity_search_with_score(query=self.query, k=10)
        print(type(results))
        print(type(results[0]))
        print(type(results[0][0]))
        print(type(results[0][1]))
                # return results

        planner_data = []
        print('Retrieval in process!')

        for doc, score in results:
                score = float(score)
                confidence = 1/(1+score)
                planner_data.append({
                    "type":doc.metadata.get('type'),
                    "table":doc.metadata.get('table'),
                    "confidence":round(float(confidence),3),
                    "decsription":doc.metadata.get('description')
                }
                )
        print('Retrieval Done!')
        return planner_data


    def selected_table(self):

        actual_schema = []
        data = self.retrieve_table()
        
        retrieved_table = [d['table'] for d in data]

        # return retrieved_table

        with open(self.schema_file, "r", encoding="utf-8") as f:
            self.schema_text = f.read()

        self.schema_text = self.schema_text.split('\n\n')

        for t in retrieved_table:
            for sch in self.schema_text:
                if t in sch:
                    actual_schema.append(sch)
        
        return "\n\n".join(actual_schema)


@tool
async def database_query_tool():
    """
    Use this tool whenever answering the user's question requires
    retrieving or analyzing information from the enterprise SQL database.

    This includes requests about operational data, fleet management,
    vehicles, drivers, tyres, shipments, routes, customers, workshops,
    transport orders, KPIs, reports, or any information stored in the
    database.

    Do not use this tool for greetings, introductions, general knowledge,
    casual conversation, or questions that can be answered without
    accessing enterprise data.
    """

    return {'requires_database':True}




# Defining the different conversational state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    SQL: str
    dbname: str
    Response: str
    Chart: str
    schema_context: str
    coding_prompt: str
    error: Optional[str]
    retry_count: int

    db_result: Optional[dict]

# Prompt for the LLM to generate code
def code_prompt()->PromptTemplate:
    
    
    prompt_template = PromptTemplate(
        input_variables = [ 'question', 'conversation_history'],
        template=""""
        You are Aiva, a senior Microsoft SQL Server (MSSQL) Data Analyst and SQL Engineer. Your primary goal is to help users understand their data.

        For query related to file management, clearence, tyre management, inventory management politely inform the user, we don't have information on this domain. We'll upgarde on this shortly. Do not invent or modify the table or column names.

    Given a conversation history: 
    "{conversation_history}"

    Generate a valid MS SQL query (SELECT only) that answers the current question:
    "{question}"

    Return ONLY the SQL query.
    Only return the SQL query. Do not include explanations.
            """.strip()
        )
    
    return prompt_template



# Chatbot Class
class GeminiChatBot:

    def __init__(
        self,
        gemini_api_key: str,
        api_token: str,
        domain_url: str,
        model_name: str = "gemini-flash-lite-latest",
        max_history: int = 20
    ):
        self.api_token = api_token
        self.max_history = max_history
        self.domain_url = domain_url

        self.agent = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key
        ).bind_tools(
            [
                database_query_tool
            ]
        )

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key
        )

        self.graph = self._build_graph()
    

# Building the stategraph
    def _build_graph(self):

        builder = StateGraph(ChatState)

        builder.add_node( 
            "agent", 
            self._agent_node 
            )

        builder.add_node( 
            "retrieve_tables", 
            self._retrieve_tables_node 
            )

        builder.add_node(
            "chatbot",
            self._chatbot_node
        )

        builder.add_node(
            "execute_sql",
            self._execute_sql_node
        )

        builder.add_node(
            "retry_sql",
            self._retry_sql_node
        )

        builder.add_edge( 
            START, 
            "agent" 
            )


        builder.add_conditional_edges( 
            "agent", 
            self._route_after_agent, 
            { "database": "retrieve_tables", 
             "general": END } 
            )

        # builder.add_edge(
        #     START,
        #     "chatbot"
        # )

        builder.add_edge( 
            "retrieve_tables", 
            "chatbot" 
            )

        builder.add_edge(
            "chatbot",
            "execute_sql"
        )

        builder.add_conditional_edges(
            "execute_sql",
            self._route_after_execution,
            {
                "success": END,
                "retry": "retry_sql",
                "failed": END
            }
        )

        builder.add_edge(
            "retry_sql",
            "execute_sql"
        )


        memory = MemorySaver()

        return builder.compile(
            checkpointer=memory
        )


# Agent Node to decide direct answer or Tool calling

    async def _agent_node(self,
                          state: ChatState):
        
        messages = [SystemMessage(content=AGENT_PROMPT),
            *state['messages']]
        response = await self.agent.ainvoke(messages)

        print(response)
        print(response.tool_calls)
        return {
            "messages": [response]
        }
    
# Routing node after tool selection
    async def _route_after_agent( self, state: ChatState ):
        last_message = state["messages"][-1] 
        if getattr(last_message, "tool_calls", None): 
            return "database" 
        return "general"

# Table retrieval node
    async def _retrieve_tables_node( self, 
                                    state: ChatState ): 
        # last_message = state["messages"][-1]

        # question = last_message.tool_calls[0]["args"]["question"]
        question = next(
        msg.content
        for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )
        # question = state["messages"][-1].content 
        print(question)
        table_retrieval = TableRetrieval( query=question, table_vector_store=table_vector_store, schema_file=r"C:\Users\nitish.jha\Desktop\Gemini_chatbot\table_schema.txt" ) 
        selected_tables = table_retrieval.selected_table() 
        print('selected_tables done!')
        return { "schema_context": selected_tables }


# It is the LLM node for SQL generation
    async def _chatbot_node(
        self,
        state: ChatState
    ):
        schema_context = state["schema_context"]
        history = state["messages"][
            -self.max_history:
        ]

        conversation = []

        for msg in state["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None):
                conversation.append(msg)

        messages = [
            SystemMessage(
                # content=self.system_prompt
                content = f"""
You are 'Aiva' virtual assistant for 'DCC Logistics Suite NG'. You can help in managing transport operations, shipment order, vehicles, workshop, fleet activities etc. 

Use only information available in the schema. Do not invent or modify the table or column names.

Schema:
{schema_context}

Your job is to:

1. write MSSQL SQL for the user query whenever needed.
2. Write the response as the user asked.
3. Check whether the result is suitable for visualization.

Return ONLY valid JSON.

Rules:
- Do NOT wrap the response in markdown.
- Do NOT use ```json or ``` fences.
- Do NOT include explanations.
- Do NOT include any text before or after the JSON.
- The response must be parseable by json.loads().

Output format:

{{
    "SQL": "generated SQL or NONE",
    "Response": "Generate response as per the user query",
    "Chart": "chart object"

}}

Rules:

1. If SQL is required:
   - Generate MSSQL SQL only.
   - Do not invent or modify the table or column names. 
   - Use schema (table or columns) that is available.
   - If SQL is generated, set "answer" to a short natural language statement describing what the query will return and don't guess any  numbers in your answer. Simply say the 'Answer of your query is as follows:', 'The requested information is provided below:' etc. Keep the answer concise and user-friendly.

2. If SQL is NOT required:
   - Set SQL to NONE.
   - Put the answer in Response.

3. If chart is needed:
   - Chart MUST be an object.
   - Each chart object must have "type":"Bar|Pie|Line", "label":(name of the metric), "labels":"array of strings", and "data":"array of numbers".

4. Chart object format MUST be:

  "type": "Bar|Pie|Line|NONE",
  "label": "metric name",
  "labels": "array of strings",
  "data": "array of numbers"

5. If no chart is appropriate:
   - Chart = NONE


Only return the JSON. Do not include explanations.
            """.strip()
            ),
            *conversation
        ] 
    # +history
        response = await self.llm.ainvoke(
            messages
        )
        # print(messages)
        print("-"*100)
        print(response.content[0]['text'])
        print("-"*100)
        result = json.loads(response.content[0]['text'])
        # print(response)
        # return {
        #     "messages": [response]
        # }
        print(result)
        print("RETURNING STATE:")
        print({
            "SQL": result["SQL"],
            # "Response": result["Response"],
            "Chart": result["Chart"]
        })
        return {
        "messages": [response],
        "coding_prompt": messages,
        "SQL": result["SQL"],
        "Response": result["Response"],
        "Chart": result["Chart"],
        "error": None,
        "retry_count": 0
    }

# Generated SQL execution node from API calling
    async def _execute_sql_node(
            self,
            state: ChatState
        ):
        print("execute_sql entered")
        # prompt = state['coding_prompt']
        # print(state.keys())
        # sql = state['SQL']
        # print(sql)
        # print(state['messages'][1]['text']["SQL"])
        if state["SQL"] == "NONE":

            return {
                "error": None
            }
        try:
            print('SQL execution intiated...')
            executor = Code_Execution(
                sql=state["SQL"],
                dbname=state["dbname"],
                response=state['Response'],
                prompt=state['coding_prompt'],
                token=self.api_token,
                id=0,
                domain_url=self.domain_url
            )
            print('Executor intantiated!')
            result = executor.SQL_Query_Execution()
            print('SQL execution completed...')
            # print(result['data'])
            # return {
            #     "db_result": result,
            #     "error": None
            # }
            return {
                    "db_result": result,
                    "error": result["error"]
                }

        except Exception as e:
            print(traceback.print_exc())
            return {
                "error": str(e)
            }

# If failed or received error from API, retrying again with the node
    async def _retry_sql_node(
    self,
    state: ChatState
):
        prompt = f"""
        You are an expert SQL debugging and correction agent.

        Your task is to FIX a failed SQL query so that it:
        1. Executes successfully
        2. Still answers the original user query correctly

User Question:
{state['question']}

Given the table schema:
{state["schema_context"]}

Generated SQL:
{state['sql']}

Database Error:
{state['error']}

Your job is to:

1. Fix the wrong SQL generated as per the error received.
2. Write the response as the user asked.
3. Check whether the result is suitable for visualization.

Return ONLY valid JSON.

Output format:

{{
    "SQL": "generated SQL or NONE",
    "Response": "Generate response as per the user query",
    "Chart": "Bar | Line | Pie | NONE"

}}
"""
        response = await self.llm.ainvoke(prompt)

        fixed = json.loads(
    response.content
)
        return {
    "SQL": fixed["SQL"],
    "retry_count": state["retry_count"] + 1
}

# Decider node to re-try or send the output
    def _route_after_execution(
        self,
        state: ChatState
    ):
        
        if not state.get("error"):
            return "success"
        
        if state["retry_count"] >= MAX_RETRIES:
            return "failed"
        
        return "retry"


# Ask Method
    async def ask(
        self,
        session_id: str,
        question: str,
        # selected_table: str,
        dbname: str
    ):

        config = {
            "configurable": {
                "thread_id": session_id
            }
        }

        result = await self.graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=question
                    )
                ],
                # "schema_context": selected_table,
                "dbname": dbname
            },
            config=config
        )
        print(result["messages"][-1].usage_metadata)
        return result
        return result["messages"][-1].content




# # Node for generated SQL execution
class Code_Execution():

    def __init__(self, sql, dbname, response, prompt, id, token, domain_url):
        self.sql = sql
        self.dbname = dbname
        self.response = response
        self.prompt = prompt
        self.id = id
        self.token = token
        self.domain_url = domain_url



    def SQL_Query_Execution(self):
        try:
            guid = str(uuid.uuid1(4))
            print("GUID created successfully!")
            print(guid)
            serialized_prompt = [
                {
                    "type": msg.type,
                    "content": msg.content
                }
                for msg in self.prompt
            ]
            print(self.sql)
            print(self.response)
            sql_execution_payload = json.dumps({
                "id": 0, 
                "log_inst": 1, 
                "guid": guid, 
                "user_id": self.id, 
                "prompt": json.dumps(serialized_prompt), 
                "sql_query": self.sql, 
                "response": self.response, 
                "rating": 1
            })

            # API_URL = "https://logsuiteblapi_dev.dcctz.com/DCCLogisticsSuite/BLv2_demo/api/AivaUserPromptHistory/Add_v1"
            API_URL = f"{self.domain_url}/api/AivaUserPromptHistory/Add_v1"
            print(API_URL)
            HEADERS = {
                'Content-Type': 'application/json',
                'Authorization':f"Bearer {self.token}"
            }
            print(self.token)
            print('Hitting the DB API to get the response.')
            # Hitting the API for getting response from API of the generated SQL
            db_api_response = requests.request("POST", url=API_URL, headers=HEADERS, data=sql_execution_payload, verify=False)
            print('Received response from Db execution API!')
            print("="*100)
            print("Status:", db_api_response.status_code)
            # print("Response:", db_api_response.text)
            # Fetching the actual data of the user query
            result = db_api_response.json()
            print("="*100)
            print(result)
            # response = db_api_response.       
            # print(response)
            # return result['tblData']
            return {'data':result['tblData'],'error':None}

        except pyodbc.Error as e:
        # except Exception as e:
            print("SQL Error:", str(e))

            return {
                "data": None,
                "error": str(e)
            }

#         except Exception as e:
#             # return f'An error occured while executing the generated response. The error is {e}.'
#             return {'data':None, 'error': f"Generated SQL code execution error from DB. Received error {e}."}





# # Converting the all output to MarkDown formula
def dict_list_to_markdown(data: list[dict], text: str = "", chart= "") -> str:
    """
    Converts a list of dictionaries into a markdown table.
    Includes an optional descriptive text and chart image.
    """

    # Start with text (if any)
    markdown = ""
    if text:
        markdown += text.strip() + "\n\n"

    # Handle empty data
    if not data:
        markdown += "No data available."
        return markdown

    # Build table headers & rows
    headers = list(data[0].keys())
    markdown += "| " + " | ".join(headers) + " |\n"
    markdown += "|" + " --- |" * len(headers) + "\n"

    for row in data:
        markdown += "| " + " | ".join(map(str, row.values())) + " |\n"

    # Add chart (if provided)
    if chart:
        if chart.endswith(".html"):
            # Embed HTML chart in iframe
            markdown += f'\n<iframe src="{chart}" width="100%" height="500px" style="border:none;"></iframe>\n'
        else:
            # Static image
            markdown += f'\n![chart]({chart})\n'
        # markdown += "\n" + f"![chart]({chart})\n"

    return markdown




# # Converting the  longer responde to excel
def excel_conversion(data):
    today = date.today().isoformat()          # YYYY-MM-DD
    uid = str(uuid.uuid4())

    export_dir = EXPORT_BASE / today
    export_dir.mkdir(parents=True, exist_ok=True)

    file_path = export_dir / f"{uid}.xlsx"
    print(file_path)

    df = pd.DataFrame(data)

    df.to_excel(file_path, index=False)
    domain_url = os.getenv("DOMAIN_URL")
    # return 1
    return {"downloadable_url":f"{domain_url}/download/excel/{today}/{Path(file_path).name}"}

