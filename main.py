import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("Retrieving...")

    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)

    # query = "Explain all the migrations along with their file names that have been done in this project"
    query = "Read the schema and tell guess what this project is about? What else can be developed using this project?"
    # in service account -> console -> run this command = gcloud projects add-iam-policy-binding our-forest-445010-h3     --member="serviceAccount:arshan@our-forest-445010-h3.iam.gserviceaccount.com"     --role="roles/aiplatform.admin"
    embeddings = VertexAIEmbeddings(model="text-embedding-005")

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding = embeddings
    )

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrival_chain = create_retrieval_chain(
    #     retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    # )
    #
    # result = retrival_chain.invoke(input={"input": query})
    #
    # print(result)

    # test_case_generation_template = """
    # Use the following pieces of context to generate all the possible functional test cases to the question at the end.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # Give answer in a tabular manner with the column headings as: Name | Precondition | Objective | Test Script (Step-by-Step) - Step | Test Script (Step-by-Step) - Test Data | Test Script (Step-by-Step) - Expected Result
    #
    #
    # {context}
    #
    # Question: {question}
    #
    # Helpful Answer:
    # """

    # template = """You are an expert Rails developer assisting with personalized Active Record queries for a Rails 7.0 project. The project includes models like Doctor, Receptionist, Patient, and Appointment, and the schema, migration files, and dependencies are embedded in the system for reference.
    #
    # ### Task:
    # Generate an optimized Active Record query for the given project requirements, using context retrieved from the project's schema, migrations. Ensure the query is accurate, efficient, and adheres to Rails conventions.
    #
    # ### Input Format:
    # 1. **Query Description**: [Provide a detailed description of the data needed. Example: "Fetch all appointments for a doctor within the last month."]
    # 2. **Additional Constraints** (Optional): [Specify any additional conditions or filters. Example: "Doctor's name is 'Dr. John Doe'."]
    # 3. **Expected Output Structure** (Optional): [Define the structure of the output data if needed. Example: "Include appointment date and patient name."]
    #
    # ### Output Format:
    # - **Active Record Query**: [Provide the Rails query.]
    # - **Explanation**: [Explain the purpose of the query, model relationships used, and how it satisfies the input requirements.]
    # - **Optimization Tips**: [Offer suggestions to improve query performance if applicable.]
    #
    # ### Notes:
    # 1. Use the vector store to retrieve project details like schema, model relationships, and migration changes for accurate query generation.
    # 2. Always explain the logic behind the query, especially if it involves joins, conditions, or aggregations.
    # 3. Highlight performance considerations, such as indexing or eager loading.
    #
    # ### Example Input:
    # Query Description: "List all patients assigned to a specific doctor within the last week."
    # Additional Constraints: "Doctor's name is 'Dr. Alice Smith'."
    #
    # ### Example Output:
    # **Active Record Query**:
    # ```ruby
    # Patient.joins(:doctor)
    #        .where('patients.created_at >= ?', 1.week.ago)
    #
    # Question: {question}
    #
    # """

    template = """
    You are a Rails schema expert. Analyze the database schema and migrations. Provide details or answers based on the user's query.
    
    ### Input:
    1. **Query**: [Ask about model structure, relationships, or fields. Example: "What fields and relationships does the Patient model have?"]
    
    ### Output:
    - **Model**: [Name of the model]
    - **Fields**: [List fields and their data types]
    - **Relationships**: [Describe associations, e.g., has_many, belongs_to]
    - **Indexes**: [List indexes if relevant]
    - **Explanation**: [Briefly explain how the model fits into the schema]
    - **Example Query** (Optional): [Provide a sample Active Record query]
    
    ### Example Input:
    Query: "Describe the Doctor model and its relationships."
    
    ### Example Output:
    - **Model**: Doctor
    - **Fields**: id (integer), name (string), specialization (string), created_at (datetime), updated_at (datetime)
    - **Relationships**: has_many :appointments, has_many :patients through: :appointments
    - **Indexes**: Index on `name`
    - **Explanation**: Doctors manage appointments and have many patients via the Appointment model.
    - **Example Query**:
    ```ruby
    Doctor.includes(:appointments, :patients).find_by(name: 'Dr. Smith')
    
    Question: {question}
    
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)