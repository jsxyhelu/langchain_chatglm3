from langchain import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma,FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_wenxin.llms import Wenxin
from langchain.chains import LLMChain
from langchain_chatglm3 import ChatGLM3

#读取领域信息
loader = TextLoader("./观潮.txt")
documents = loader.load()
#文本分割
text_splitter = CharacterTextSplitter(chunk_size=28, chunk_overlap=0)
documents = text_splitter.split_documents(documents)
# 向量化 embedding model: m3e-base
model_name = "./m3e-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="为文本生成向量表示用于文本检索"
            )
 
# load data to Chroma db
db = FAISS.from_documents(documents, embedding)
# LLM选型
#llm = Wenxin(model="ernie-bot", baidu_api_key="", baidu_secret_key="")

llm = ChatGLM3(endpoint_url = "http://192.168.196.211:6006/")

#output = llm("1+3=?")
#print(output)

#template = """{question}"""
#prompt = PromptTemplate(template=template, input_variables=["question"])
#llm_chain = LLMChain(prompt=prompt, llm=llm)               
#question = "2+3=？"
#llm_chain.run(question)


retriever = db.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
response = qa({"question": "大潮分为那几个阶段？"})
answer = response["answer"]
print("RAG回答:", answer)


#llm = ChatGLM3()
#output = llm("1+3=?")
#print(output)