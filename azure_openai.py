import os, json
from openai import AzureOpenAI
from dotenv import load_dotenv
from schema import signature_analysis_schema

load_dotenv()


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),            
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_version="2024-02-01"
)

MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")   


async def get_model_response(prompt_text: str, reference_image: str, extracted_image: str):

    messages = [
        {
            "role": "system",
            "content": "You are an AI that compares original images with extracted images."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},

                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{reference_image}"
                    }
                },


                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{extracted_image}"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content


async def get_signature_model_response(prompt_text: str, reference_image: str, extracted_image: str):
    messages = [
        {
            "role": "system",
            "content": "You are an AI that compares original images with extracted images."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{reference_image}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{extracted_image}"
                    }
                }
            ]
        }
    ]

    # Force the model to return JSON that fits the schema
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        functions=[signature_analysis_schema],
        function_call={"name": "return_signature_analysis"}
    )

    fn_call = response.choices[0].message.function_call

    # Parse guaranteed JSON
    data = json.loads(fn_call.arguments)

    return data
