
from main import GeminiChatBot,  TableRetrieval, table_vector_store,  HelpdeskResponse, excel_conversion
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.templating import Jinja2Templates
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
import shutil
import requests 
import json
import jwt
import sys

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

app = FastAPI(root_path='/chat')
templates = Jinja2Templates(directory="templates")



BASE_DIR = Path(__file__).resolve().parent   #App directory
EXPORT_BASE = BASE_DIR / "exports"


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)






def get_numeric_columns(rows):
    if not rows:
        return []

    numeric_columns = []

    # sample_row = rows[0]

    # for col, value in sample_row.items():
    #     if isinstance(value, (int, float)):
    #         numeric_columns.append(col)



    for col in rows[0].keys():

        is_numeric = True

        for row in rows:
            value = row.get(col)

            if value is None:
                continue

            try:
                float(value)
            except (ValueError, TypeError):
                is_numeric = False
                break

        if is_numeric:
            numeric_columns.append(col)

    return numeric_columns


def get_categorical_columns(rows):
    if not rows:
        return []

    categorical_columns = []

    sample_row = rows[0]

    for col, value in sample_row.items():
        if isinstance(value, str):
            categorical_columns.append(col)
            

    return categorical_columns

# chatbot = GeminiChatBot(
#     gemini_api_key=GEMINI_API_KEY,
#     schema_file="table_schema.txt",
#     model_name="gemini-flash-lite-latest"
# )

# chatbot = GeminiChatBot(
#     gemini_api_key=GEMINI_API_KEY,
#     model_name="gemini-flash-lite-latest"
# )

# Request receiving model.
class ChatRequest(BaseModel):
    # session_id: str
    question: str
    api_token: str
    db_api:Optional[str] = None
    category:str
    domain_name:str


api_key_url = "https://logsuitedomainverify.dcctz.com/api/domain/get_dynamic_data?access_token=46|dBslX9hktLYr3XfeD0uaoh3hd5ejfz6sPbQ6Midra9f22742"





@app.get("/download/excel/{file_path:path}")
def download_file(file_path: str):
    # return 2
    full_path = (EXPORT_BASE / file_path.replace('..','')).resolve()
    print(file_path)
    print(full_path)

    # return full_path
    if not full_path.exists():
        # return 2
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=full_path,
        # filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/destroy/excel-files")
def download_file():
    # Today’s folder (keep it)
    today_folder = date.today().isoformat() 

    delete_folder= BASE_DIR / "exports"

    # Ensure the base directory exists
    if not delete_folder.exists():
        print(f"Base directory {delete_folder} does not exist.")
        exit(1)

    # Loop through all subfolders in exports
    for folder in delete_folder.iterdir():
        try:
            if folder.is_dir() and folder.name != today_folder:
                print(f"Deleting folder: {folder}")
                shutil.rmtree(folder)  # permanently delete folder and all files
            else:
                print(f"Skipping folder: {folder}")
        except Exception as e:
            print(f"Exception: during deleteing excel file: {str(e)}")    

    return "Success:  files have been deleted successfully."



@app.post("/ask")
async def process_query(
    request: ChatRequest
):
    api_tkn = request.api_token
    print(jwt.decode(request.api_token, options={"verify_signature":False}, algorithms=["HS256"]))
    db_json = jwt.decode(request.api_token, options={"verify_signature":False}, algorithms=["HS256"])
    db_name = db_json['dbname']
    
    # Payload for token verification
    api_key_payload = {
"columns": ["DBName", "SubDomain", "aiva_api_key", "BLApiUrl"], 
"conditions": [ { "column": "DBName", "operator": "=", "value": db_name } ]
}
    api_key_data = requests.post(api_key_url,json=api_key_payload)
    print(api_key_data.json()['data'][0])
    api_key = api_key_data.json()['data'][0]['aiva_api_key'] # Got the actua API token
    domain_url = api_key_data.json()['data'][0]["BLApiUrl"]
    # return api_key

    try:
            query = request.question
            user_domain_name = f"https://{request.domain_name}.dcclogsuite.com/helpdesk/public"
            # user_domain_name = domain_name
            print(f"Received Query: {query}")
            print(user_domain_name)
    except Exception as e:
        return f"Error related to getting 'user question' and 'domain url'. The error is {e}."
    print('Question and domain URL received successfully.')

    if request.category=='Help Desk':
        # sys.exit()
        # query = request.question
        
        chat_history = []

        helpdesk = HelpdeskResponse(user_question=query,base_url=user_domain_name, api_key=api_key)

        try:

                results = helpdesk.qa_chain_generation()
                print(results)
                result = results({"question": query, "chat_history": chat_history})
                chat_history.append((query, result["answer"]))
                print(result['source_documents'])
                data = {
                    "answer": result["answer"],
                    "source_chunks": [doc.page_content for doc in result["source_documents"]]
                }
                return ORJSONResponse(content=data)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif request.category=='NG Analytics':

        # Poicking the User ID
        session_id = db_json['userid']
        print(db_name)

        #  Initiating the chatbot
        chatbot = GeminiChatBot(
                gemini_api_key=api_key,
                api_token=api_tkn,
                domain_url=domain_url,
                model_name="gemini-flash-lite-latest"
            )
        print('Graph instance created!')
        # return selected_table

        try:
            # Calling the model for response (SQL) generation
            response = await chatbot.ask(
                session_id=session_id,
                question=request.question,
                # selected_table=selected_table,
                dbname=db_name
            )
            # print("="*100)
            # print(response)
            print("="*100)
            print(f"Result fetched successfully!")
            print("="*100)
            # print(response['messages'][1]['content'])
            print("-"*100)
            print(request.question)
            print("-"*100)
            print(response.get("SQL"))
            print("="*100)
            # print(response.get("Response"))
            # print(response['messages'][1]['content']['text'])
            # print("="*100)
            # print(response["messages"][1].content[0]["text"])
            # print(response.get("db_result"))
            # return response
            # print(response.get("SQL"))
            if response.get("SQL") == "NONE" or response.get("SQL") == None:
                print('No SQL response!')
                return {
                    "chart": None,
                    "sql": None,
                    # "answer": response.get("Response"),
                    "answer":response["messages"][1].content[0]["text"],
                    "data": None
                }

            else:
                print('Entering the data expected output')
                raw_data = response.get("db_result", {}).get("data", [])
                print("="*100)
                print("Data found!")


                if isinstance(raw_data, str):
                    data = json.loads(raw_data)
                else:
                    data = raw_data

            # data = json.loads(response.get("db_result", {}).get("data", []))
            # data = json.loads(response.get("db_result", {}))
            # print(len(data['data']))

                if len(data)>50:
                    # data = json.loads(response.get("db_result", {}).get("data", []))
                    print('Excel Output!')
                    data = excel_conversion(data)
                    return {
                        "chart": None,
                        "sql": response.get("SQL"),
                        "answer": f"{response.get("Response")}\n{data['downloadable_url']}",
                        "data": data
                    }

                elif len(data)==0:
                    # data = json.loads(response.get("db_result", {}).get("data", []))
                    print('Excel Output!')
                    data = excel_conversion(data)
                    return {
                        "chart": None,
                        "sql": response.get("SQL"),
                        "answer": f"No data available. There are zero values available for the requested query.",
                        "data": data
                    }

                elif len(data)<=50 and response.get("Chart") =="NONE":
                    # data = json.loads(response.get("db_result", {}).get("data", []))
                    # data = excel_conversion(data)
                    print('Non Chart response!')
                    return {
                        "chart": None,
                        "sql": response.get("SQL"),
                        "answer": response.get("Response"),
                        "data": data
                    }

                elif len(data)<=50 and response.get("Chart") !="NONE":
                    # data = json.loads(response.get("db_result", {}).get("data", []))
                    print('Chart response!')
                    try:
                        catge_col = get_categorical_columns(data)[0]
                        numeric_col = get_numeric_columns(data)[0]
                        values = [
                            row[numeric_col]
                            for row in data
                        ]
                        catgeroy = [
                            row[catge_col]
                            for row in data
                        ]
                        response.get("Chart")['labels']=catgeroy
                        response.get("Chart")['data']=values
                        return {
                            "chart": response.get("Chart"),
                            "sql": response.get("SQL"),
                            "answer": response.get("Response"),
                            "data": json.loads(response.get("db_result", {}).get("data", []))
                        }

                    # category_col = get_categorical_columns(data)[0]
                    # numeric_cols = get_numeric_columns(data)

                    # categories = [
                    #     row[category_col]
                    #     for row in data
                    # ]

                    # chart_data = []

                    # for col in numeric_cols:
                    #     chart_data.append({
                    #         "label": col,
                    #         "values": [
                    #             float(row.get(col, 0))
                    #             if row.get(col) not in (None, "")
                    #             else 0
                    #             for row in data
                    #         ]
                    #     })

                    # response.get("Chart")["labels"] = categories
                    # response.get("Chart")["data"] = chart_data

                    # return {
                    #     "chart": response.get("Chart"),
                    #     "sql": response.get("SQL"),
                    #     "answer": response.get("Response"),
                    #     "data": json.loads(
                    #         response.get("db_result", {}).get("data", "[]")
                    #     )
                    # }



                    except Exception as e:
                        print(e)
                        return {
                            "chart": None,
                            "sql": response.get("SQL"),
                            "answer": response.get("Response"),
                            # "data": json.loads(response.get("db_result", {}).get("data", []))
                            "data": data
                        }
            

        except Exception as e:
            print(f"An error occured. The error is {e}")
            return {
                    "chart": None,
                    "sql": None,
                    "answer":"We're unable to process this request at the moment. Please try after sometime",
                    "data": None
                }
            # return e

    
