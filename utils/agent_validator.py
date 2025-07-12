# utils/agent_validator.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into environment

# Now this will work as expected
openai_key = os.getenv("OPENAI_API_KEY")

def validate_code(training_file: str):
    with open(training_file, "r") as f:
        code = f.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a senior MLOps engineer tasked with code review for ML training pipelines. 
Your job is to:
- Review Python training code for a tabular ML use case.
- Determine if it's ready for production use.

A good production-ready training script should:
- Handle missing/null values and categorical encoding
- Use proper train-test split with seed
- Select model type based on task (regression/classification)
- Choose proper evaluation metric
- Use sklearn pipelines if possible
- Save model to disk
- Avoid unnecessary global state or bad practices

Your output MUST follow this format:

[RESULT]
Valid: True or False

[REASON]
- Reason 1
- Reason 2
...
"""),
        ("human", "Here is the code to review:\n\n{code}")
    ])

    llm = ChatOpenAI(temperature=0, model="gpt-4",
                     openai_api_key=openai_key)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"code": code})

    # Parse result block from LLM response
    valid_flag = "valid: true" in response.lower()
    return valid_flag, response
