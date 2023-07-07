import os
import PyPDF2
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(pdf_path):
    text = ""

    directory = pdf_path
    for file in os.listdir(directory):
        if not file.endswith(".pdf"):
            continue
        with open(os.path.join(directory,file), 'rb') as pdfFileObj:  # Changes here
            reader = PyPDF2.PdfReader(pdfFileObj)

            for page in reader.pages:
                text += " " + page.extract_text()
    return text


def extract_details_chat(extracted_text):

    chat = ChatOpenAI(temperature=0.0)

    response_schemas = [
        ResponseSchema(name="place (from)", description="place where flight starts/takes-off"),
        ResponseSchema(name="date (from)", description="date on which flight starts/takes-off (DD/MM/YYYY)"),
        ResponseSchema(name="time (from)", description="time at which flight starts/takes-off"),
        ResponseSchema(name="place (to)", description="place where flight end/lands"),
        ResponseSchema(name="date (to)", description="date on which flight end/lands (DD/MM/YYYY)"),
        ResponseSchema(name="time (to)", description="time at which flight end/lands"),
        ResponseSchema(name="PNR Number", description ="PNR Number of flight"),
        ResponseSchema(name="amount", description="cost of flight ticket")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("Parse through and find the following details from the text extracted from a travel bill\n{format_instructions}\n{extracted_text}")  
    ],
    input_variables=["extracted_text"],
    partial_variables={"format_instructions": format_instructions}
)

    _input = prompt.format_prompt(extracted_text = extracted_text)
    output = chat(_input.to_messages())
    # print(output)

    return output_parser.parse(output.content)

def extract_details_model(extracted_text):
    model = OpenAI(temperature=0)

    response_schemas = [
        ResponseSchema(name="place (from)", description="place where flight starts/takes-off"),
        ResponseSchema(name="date (from)", description="date on which flight starts/takes-off (DD/MM/YYYY)"),
        ResponseSchema(name="time (from)", description="time at which flight starts/takes-off"),
        ResponseSchema(name="place (to)", description="place where flight end/lands"),
        ResponseSchema(name="date (to)", description="date on which flight end/lands (DD/MM/YYYY)"),
        ResponseSchema(name="time (to)", description="time at which flight end/lands"),
        ResponseSchema(name="PNR Number", description ="PNR Number of flight"),
        ResponseSchema(name="amount", description="cost of flight ticket (in INR)")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate( 
    template=
    '''
    Your task is to parse through and find the following details from the text extracted from a travel bill present in triple backticks and return the them in json format.\
    If detail is not found, return None. (if there are multiple tickets in the provided text so return a single JSON object containing multiple JSON objects where each object represents a different travel expense).\
    Incase of round trip tickets, count the travel expense just once and make the amount of the return trip to be 0. return JSON objects in the follwing format:\

    {
        from: {
            place,
            date: (Day, DD/MM/YYYY), 
            time
        }, 
        to: {
            place,
            date: (Day, DD/MM/YYYY),
            time
        },
        mode of travel: (flight, cab, train, other),
        pnr no,
        amount: (in INR)
    }
    {format_instructions}
    {extracted_text}
    ''',
        # template="Parse through and find the following details from the text extracted from a flight ticket\n{format_instructions}\n{extracted_text}",
        input_variables=["extracted_text"],
        partial_variables={"format_instructions": format_instructions}
    )

    _input = prompt.format_prompt(extracted_text = extracted_text)
    output = model(_input.to_string())

    return output_parser.parse(output)
    # return output

def main():
    pdf_path = '.'

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)

    # print(extracted_text)
    print("--------------------")
    

    # Extract PAN number using GPT-3
    details_json_1 = extract_details_chat(extracted_text)
    # details_json_2 = extract_details_model(extracted_text)

    print(details_json_1.get('place (from)'))
    # Print the extracted PAN number
    print(details_json_1)
    # print(details_json_2)


if __name__ == '__main__':
    main()
