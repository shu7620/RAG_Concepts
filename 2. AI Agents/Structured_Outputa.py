from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 1. Define expected output schema using Pydantic
class CustomerTicketAnalysis(BaseModel):
    category: str = Field(description="Issue category e.g. Billing, Technical, Account")
    urgency_score: int = Field(description="Priority score from 1 (low) to 5 (critical)")
    summary: str = Field(description="A concise 1-sentence summary of the ticket")
    action_items: list[str] = Field(description="List of immediate steps support should take")
    
# 2. Bind Model to schema
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
structured_llm=llm.with_structured_output(CustomerTicketAnalysis)


# 3. Invoke
user_email = "I've been charged twice for my subscription this month ($49.99 x 2). Please fix this immediately!"
result: CustomerTicketAnalysis=structured_llm.invoke(user_email)

print("Category:", result.category)
print("Urgency:", result.urgency_score)
print("Action Items:", result.action_items)