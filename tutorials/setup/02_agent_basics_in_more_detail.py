# Source of this code is the following article published in towardsdatascience https://medium.com/towards-data-science/genai-with-python-build-agents-from-scratch-complete-tutorial-4fc1e084e2ec
import ollama
ollama.pull("llama3.2") 
llm = "llama3.2"
q='''who died on Sep 9, 2024?'''
res = ollama.chat(model=llm,
                  messages= [{"role":"system","content":""},
                             {"role":"user","content":q}])
res["message"]["content"]

print(res["message"])

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool("tool_browser")
def tool_browser(q: str) -> str:
    """Search on DuckDuckGo browser by passing the input 'q'"""
    return DuckDuckGoSearchRun().run(q)
#test
print(tool_browser(q))

from semantic_router.utils.function_call import FunctionSchema

# Note: the function schema fomratted the tool in a way which was not 
# readable in the code snippet str_tools
# def browser(q:str) -> str:
#     """Search on DuckduckGo browser by passing the  input 'q'"""
#     return DuckDuckGoSearchRun().run(q)
# tool_browser=FunctionSchema(browser).to_ollama()
# tool_browser

@tool("final_answer")
def final_answer(text:str)-> str:
    """Returns a natural language response to the user by passing the input 'text'.
    You should provide as much context as possible and provide the source of the information.
    """
    return text

prompt = """
You know everything, you must answer each questions from the user. 
You can use the tools provided to you. The goal is to provide the user 
the best possible answer, including key information about the sources 
and tools used.
Note, when using as tool, you provide the tool name with and the arguments
to use JSON format. For each call, you MUST ONLY use one tool and the response
format must ALWAYS be in the pattern:
'''json
{"name":"<tool_name>","parameters":{"<tool_input_key>":<tool_input_value>}}
'''
Remember, do not use any tool with the same query more than once
Remember, if the user do not ask a specific question then you must use 
the tool "final answer" directly.
Everytime a user asks a question, you take a note of some key words in the memory.
Everytime you find some information related to the user's question, you 
take note of some key words in the memory.
You should aim to collect information from a diverse range of sources before
providing the answer to the user.
Once you have collected plenty of information to answer the user's question 
use the "final answer" tool.  
"""

dic_tools ={"tool_browser":tool_browser,
            "final_answer":final_answer}


str_tools="\n".join([str(n+1)+".`"+str(v.name)+"` :"+str(v.description) for n,v in enumerate(dic_tools.values())])

prompt_tools = f"You can use the following tools:\n{str_tools}"
print(prompt_tools)

# LLM deciding what tools to use

from pprint import pprint

llm_res = ollama.chat(
    model = llm,
    messages = [{"role":"system", "content":prompt+"\n"+prompt_tools},
                {"role":"user","content":"hello"
                 }],
    format = "json"   
)
pprint(llm_res)

# LLM deciding which tool to use (output format = json)

llm_res = ollama.chat(
    model = llm,
    messages = [{"role":"system","content":prompt+"\n"+prompt_tools},
                {"role":"user","content":q}],
    format ="json")
llm_res["message"]["content"]

#LLm with context

import json

tool_input = json.loads(llm_res["message"]["content"])["parameters"]["q"]
print(tool_input)
context = tool_browser(tool_input)
print("tool output:\n",context)

llm_output = ollama.chat(
    model=llm,
    messages=[{"role":"system", "content":"Give the most accurate answer using the following information:\n"+context},
              {"role":"user","content":q}
              ])
print("\nllm output:\n", llm_output["message"]["content"])


from pydantic import BaseModel #this is the standard class

# taking for example the last LLM response, I want this structure:
# {tool_name: 'tool_browser',
#  tool_input: {'q':'September 9 2024 deaths'},
#  tool_output: str(tool_browser({'q':'September 9 2024 deaths'})) }

class AgentRes(BaseModel):
    tool_name: str #<-- must be a string = 'tool_browser'
    tool_input: dict #<-- must be a dictionary = {'q': 'September 9 2024 deaths'}
    tool_output: str | None = None #can be a string or None, default = None. The | symbol is a shorthand for "or" 

    @classmethod
    def from_llm(cls,res:dict): #<--return the class itself
        try:
            out = json.loads(res["message"]["content"])
            return cls(tool_name=out["name"], tool_input=out["parameters"])
        except exception as e:
            print(f"Error from Ollama:\n{res}\n")
            raise e
        
#test
agent_res=AgentRes.from_llm(llm_res)
print("from\n",llm_res["message"]["content"],"\nto")
agent_res

#test the tool output
AgentRes(tool_name="tool_browser",
         tool_input = {'q':'September 9 2024 deaths'},
         tool_output = str(tool_browser({'q':'September 9 2024 deaths'})))

'''
Messages in memory will have this structure:
[{'role':'assistant','content':'{"name":"final_answer,"parameters":{"text":"How can I assist you today?"}}'},
 {'role':'user','content':None}]
'''

def save_memory(lst_res:list[AgentRes], user_q:str) -> list:
    ##create
    memory =[]
    for res in [res for res in lst_res if res.tool_output is not None]:
        memory.extend([
            ### assistant message
            {"role":"assistant", "content":json.dumps({"name":res.tool_name,"parameters":res.tool_input})},
            ### user message
            {"role":"user","content":res.tool_output}
        ])

    ## add a reminder of the original goal
    if memory:
        memory +=[{"role":"user","content":(f'''
                This is just a reminder that my original query was `{user_q}`.
                Only answer to the original query, and nothing else, but use the information I gave you.
                Provide as much information as possible when you use the `final_answer` tool.
                ''')}]
    return memory
    
history = [{"role":"user","content":"hi there,how are you?"},
           {"role":"assistant","content":"I'm good, thanks!"},
           {"role":"user","content":"I have a question"},
           {"role":"assistant","content":"tell me"}]

def run_agent(user_q:str, chat_history:list[dict], lst_res:list[AgentRes], lst_tools:list) -> AgentRes:
    ## start memory
    memory = save_memory(lst_res=lst_res, user_q=user_q)
    ## track used tools
    if memory:
        tools_used =[res.tool_name for res in lst_res]
        if len(tools_used) >= len(lst_tools):
            memory[-1]["content"]="You must now use the `final_answer` tool."
    ## messages
    messages = [{"role":"system","content":prompt+"\n"+prompt_tools},
                *chat_history,
                {"role":"user","content":user_q},
                *memory]
    pprint(messages) #<--print to see prompt + tools + chat_history

    ## output
    llm_res = ollama.chat(model=llm,messages=messages,format="json")
    return AgentRes.from_llm(llm_res)

# test
agent_res = run_agent(user_q=q, chat_history=history, lst_res=[], lst_tools=dic_tools.keys())
print("\nagent_res:",agent_res)


import typing

class State(typing.TypedDict):
    user_q:str
    chat_history:list
    lst_res:list[AgentRes]
    output: dict

# test
state = State({"user_q":q,"chat_history":history, "lst_res":[agent_res],"output":{}})
state

# Agent
def node_agent(state):
    print("--- node_agent ---")
    agent_res = run_agent(
                        #   prompt=prompt,
                          lst_tools={k:v for k,v in dic_tools.items() if k in ["tool_browser","final_answer"]},
                          user_q=state["user_q"],
                          chat_history=state["chat_history"],
                          lst_res=state["lst_res"])
    print(agent_res)
    return{"lst_res":[agent_res]} #<--must return a list of agent_res

# test
node_agent(state)

def node_tool(state):
    print("--- node tool ---")
    res = state["lst_res"][-1]
    print(f"{res.tool_name}(input = {res.tool_input})")

    agent_res=AgentRes(tool_name=res.tool_name,
                       tool_input=res.tool_input,
                       tool_output=str(dic_tools[res.tool_name](res.tool_input)))
    return {"output":agent_res} if res.tool_name =="final_answer" else {"lst_res":[agent_res]}
#test
node_tool(state)

def conditional_edges(state):
    print("--- conditional_edges ---")
    last_res = state["lst_res"][-1]
    next_node = last_res.tool_name if isinstance(state["lst_res"], list) else "final_answer"
    print("next_node:", next_node)
    return next_node #<--must return the next node to go

# test
conditional_edges(state)


from langgraph.graph import StateGraph, END

## start the graph
workflow = StateGraph(State)

## add agent Node
workflow.add_node(node="Agent", action = node_agent)
workflow.set_entry_point(key="Agent") #<--user query

## add Tools nodes
for k in dic_tools.keys():
    workflow.add_node(node=k,action=node_tool)

## conditional edges from Agent
workflow.add_conditional_edges(source="Agent",path=conditional_edges)

## normal edges to Agent
for k in dic_tools.keys():
    if k != "final_answer":
        workflow.add_edge(start_key=k, end_key="Agent")

## end the graph
workflow.add_edge(start_key="final_answer", end_key=END)
g = workflow.compile()

## plot 
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

display(Image(
    g.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
))



    
    