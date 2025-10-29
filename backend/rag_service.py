import os
import tempfile
import re
import time
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pypdf
from openai import OpenAI

RETRIEVAL_K = 4
LINES_PER_PAGE_ESTIMATE = 50
CHUNK_PREVIEW_LENGTH = 50
SECTION_SEARCH_LINES = 5
MAX_CONFIDENCE = 100
MIN_CONFIDENCE = 10

class RAGService:
    def __init__(self):

        self.config = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'temperature': 0.0,
            'model_name': 'gpt-3.5-turbo'
        }
        
        # Initialize clients as None - will be created when API key is set
        self.embeddings = None
        self.llm = None
        self.openai_client = None
        self.vector_store: Optional[FAISS] = None
        
        # Initialize text splitter with configured chunking params
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            length_function=len,
        )
        
        prompt_text = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
            
Answer: """
        self.prompt_template = PromptTemplate(
            template=prompt_text,
            input_variables=["context", "question"]
        )

        self.qa_chain = None

    def _estimate_page_number(self, chunk_text: str, full_text_lines: list, current_page: int) -> int:
        # search for chunk start in full text and estimate page based on line position
        chunk_words = chunk_text[:CHUNK_PREVIEW_LENGTH].split()
        if not chunk_words:
            return current_page
        
        search_text = ' '.join(chunk_words[:SECTION_SEARCH_LINES])
        best_match = 0
        
        for i, line in enumerate(full_text_lines):
            if search_text.lower() in line.lower():
                best_match = i
                break
        
        estimated_page = max(1, (best_match // LINES_PER_PAGE_ESTIMATE) + 1)
        return estimated_page if estimated_page > 0 else current_page

    def _detect_section(self, chunk_text: str) -> str:
        # Look for section headers in first few lines - uppercase or title case with colon
        lines = chunk_text.split('\n')
        for line in lines[:SECTION_SEARCH_LINES]:
            line_stripped = line.strip()
            if len(line_stripped) > 0 and len(line_stripped) < 100:  # reasonable section header length
                if line_stripped.isupper() or (line_stripped[0].isupper() and ':' in line_stripped):
                    section_keywords = ['section', 'chapter', 'part', 'introduction', 'summary', 'conclusion']
                    if any(keyword in line_stripped.lower() for keyword in section_keywords):
                        return line_stripped[:50]
        return 'N/A'

    def _clean_extracted_text(self, text: str) -> str:
        if not text:
            return ''
        
        cleaned = text
        patterns = [
            r'^I will provide the complete text[:\s]*',
            r'^Here is the extracted text[:\s]*',
            r'^I\'ll extract.*?[\n\r]+',
            r'^I can.*?extract.*?[\n\r]+',
            r'^Here.*?extracted.*?[\n\r]+',
            r'^The.*?content.*?[\n\r]+',
            r'^Below.*?extracted.*?[\n\r]+',
        ]
        
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        cleaned = cleaned.strip()
        
        lines = cleaned.split('\n')
        
        if lines and lines[0].lower().startswith('microsoft word'):
            cleaned = '\n'.join(lines[1:]).strip()
        
        if lines and (lines[0].lower().startswith('here') or lines[0].lower().startswith('the following')):
            cleaned = '\n'.join(lines[1:]).strip()
        
        return cleaned

    def update_api_key(self, api_key: str):
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Initialize clients now that we have the API key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=self.config['temperature'], model_name=self.config['model_name'])
        self.openai_client = OpenAI()
    
    def _ensure_clients_initialized(self):
        """Ensure OpenAI clients are initialized, raise error if not"""
        if self.embeddings is None or self.llm is None or self.openai_client is None:
            raise RuntimeError("OpenAI API key not set. Please set API key first.")

    def process_pdf(self, pdf_file_path: str) -> tuple[List[Document], str]:
        self._ensure_clients_initialized()
        
        try:
            with open(pdf_file_path, 'rb') as file:
                pdf_content = file.read()
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name
            
            assistant = None
            thread = None
            uploaded_file = None
            vector_store = None
            
            try:
                max_wait_time = 300
                sleep_interval = 0.5
                poll_interval = 1
                with open(temp_file_path, 'rb') as upload_file:
                    uploaded_file = self.openai_client.files.create(
                        file=upload_file,
                        purpose='assistants'
                    )
                
                while uploaded_file.status != 'processed':
                    if uploaded_file.status == 'failed':
                        raise Exception("File upload failed")
                    time.sleep(sleep_interval)
                    uploaded_file = self.openai_client.files.retrieve(uploaded_file.id)

                assistant = self.openai_client.beta.assistants.create(
                    name="PDF Extractor",
                    instructions="Extract all text from the PDF file. Return only the text content, no formatting, explanations, or metadata. Do not include any introductory text or disclaimers.",
                    model="gpt-4o",
                    tools=[{"type": "file_search"}],
                    tool_resources={
                        "file_search": {
                            "vector_store_ids": []
                        }
                    }
                )

                vector_store = self.openai_client.beta.vector_stores.create(
                    name="PDF Vector Store"
                )
                
                self.openai_client.beta.vector_stores.files.create(
                    vector_store_id=vector_store.id,
                    file_id=uploaded_file.id
                )

                while True:
                    status = self.openai_client.beta.vector_stores.files.list(vector_store_id=vector_store.id)
                    if status.data and len(status.data) > 0:
                        file_status = status.data[0].status
                        if file_status == 'completed':
                            break
                        if file_status == 'failed':
                            raise Exception("Vector store file processing failed")
                    time.sleep(sleep_interval)

                assistant = self.openai_client.beta.assistants.update(
                    assistant_id=assistant.id,
                    tool_resources={
                        "file_search": {
                            "vector_store_ids": [vector_store.id]
                        }
                    }
                )

                thread = self.openai_client.beta.threads.create()

                self.openai_client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content="Extract all text from the PDF file. Return only the text content."
                )

                run = self.openai_client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id
                )
                
                elapsed_time = 0
                poll_interval = 1
                
                while run.status in ['queued', 'in_progress', 'cancelling']:
                    if elapsed_time >= max_wait_time:
                        self.openai_client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
                        raise Exception("PDF processing timed out after 5 minutes")
                    
                    time.sleep(poll_interval)
                    elapsed_time += 1
                    run = self.openai_client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id
                    )
                
                if run.status != 'completed':
                    error_msg = f"OpenAI extraction failed: {run.status}"
                    if run.status == 'failed':
                        if run.last_error:
                            error_msg += f" - {run.last_error.message}"
                    raise Exception(error_msg)
                
                messages = self.openai_client.beta.threads.messages.list(
                    thread_id=thread.id,
                    order='asc'
                )
                
                if not messages.data or not messages.data[-1].content or not messages.data[-1].content[0].text:
                    raise Exception("No text extracted from PDF")
                
                extracted_text = messages.data[-1].content[0].text.value
                extracted_text = self._clean_extracted_text(extracted_text)
                
                try:
                    self.openai_client.beta.assistants.delete(assistant.id)
                except:
                    pass
                try:
                    self.openai_client.beta.threads.delete(thread.id)
                except:
                    pass
                try:
                    self.openai_client.beta.vector_stores.delete(vector_store.id)
                except:
                    pass
                try:
                    self.openai_client.files.delete(uploaded_file.id)
                except:
                    pass

            except Exception as e:
                if assistant:
                    try:
                        self.openai_client.beta.assistants.delete(assistant.id)
                    except:
                        pass
                if thread:
                    try:
                        self.openai_client.beta.threads.delete(thread.id)
                    except:
                        pass
                if vector_store:
                    try:
                        self.openai_client.beta.vector_stores.delete(vector_store.id)
                    except:
                        pass
                if uploaded_file:
                    try:
                        self.openai_client.files.delete(uploaded_file.id)
                    except:
                        pass
                raise e
            finally:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

            documents = self.text_splitter.create_documents([extracted_text])
            
            lines = extracted_text.split('\n')
            current_page = 1
            
            for i, doc in enumerate(documents):
                chunk_text = doc.page_content
                
                estimated_page = self._estimate_page_number(chunk_text, lines, current_page)
                section = self._detect_section(chunk_text)
                
                doc.metadata['page'] = estimated_page
                doc.metadata['section'] = section
                doc.metadata['chunk_id'] = i + 1
                doc.metadata['source'] = 'pdf'
                
                current_page = estimated_page
            
            return documents, extracted_text
                
        except Exception as e:
            try:
                with open(pdf_file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    extracted_text = ""
                    for page in pdf_reader.pages:
                        extracted_text += page.extract_text() + "\n"
                    
                    documents = self.text_splitter.create_documents([extracted_text])
                    
                    lines = extracted_text.split('\n')
                    current_page = 1
                    
                    for i, doc in enumerate(documents):
                        chunk_text = doc.page_content
                        estimated_page = self._estimate_page_number(chunk_text, lines, current_page)
                        section = self._detect_section(chunk_text)
                        
                        doc.metadata['page'] = estimated_page
                        doc.metadata['section'] = section
                        doc.metadata['chunk_id'] = i + 1
                        doc.metadata['source'] = 'pdf'
                        
                        current_page = estimated_page
                    
                    return documents, extracted_text
                    
            except Exception as fallback_error:
                raise Exception(f"Failed to extract PDF: {str(fallback_error)}")

    def add_documents(self, documents: List[Document]) -> None:
        self._ensure_clients_initialized()
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Use top-k retrieval for context building
        # NOTE: Current approach recreates vector store on each upload, which is fine for single-doc use case
        retriever = self.vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def calculate_confidence_score(self, question: str, answer: str, retrieved_docs: List[Document]) -> int:
        if not retrieved_docs:
            return MIN_CONFIDENCE
        
        doc_count = len(retrieved_docs)
        answer_length = len(answer.strip())
        
        # Base score from document matches, capped at 60
        base_score = min(60, doc_count * 15)
        
        # Bonus for answer completeness
        if answer_length < 50:
            quality_bonus = 0
        elif answer_length < 200:
            quality_bonus = 10
        else:
            quality_bonus = 20
        
        score = base_score + quality_bonus
        
        return min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, score))

    def query(self, question: str) -> tuple[str, int, list]:
        if self.vector_store is None or self.qa_chain is None:
            return "No documents have been uploaded yet. Please upload a PDF first.", MIN_CONFIDENCE, []
        
        self._ensure_clients_initialized()
        
        try:
            similar_docs = self.vector_store.similarity_search(question, k=RETRIEVAL_K)
            result = self.qa_chain.invoke(question)
            confidence_score = self.calculate_confidence_score(question, result, similar_docs)
            
            sources = []
            preview_length = 100
            # print(f"Retrieved {len(similar_docs)} similar documents")  # debugging
            for doc in similar_docs:
                page_num = doc.metadata.get('page', 'N/A')
                section = doc.metadata.get('section', 'N/A')
                preview = doc.page_content[:preview_length] + '...' if len(doc.page_content) > preview_length else doc.page_content
                sources.append({
                    'page': page_num,
                    'section': section,
                    'preview': preview
                })
            
            return result, confidence_score, sources
        except Exception as e:
            return f"Error generating answer: {str(e)}", MIN_CONFIDENCE, []

    def has_documents(self) -> bool:
        return self.vector_store is not None

    def get_all_documents(self) -> List[Document]:
        if self.vector_store is None:
            return []
        
        try:
            return self.vector_store.similarity_search("", k=self.vector_store.index.ntotal)
        except Exception:
            return self.vector_store.similarity_search("", k=10000)

    def get_document_count(self) -> int:
        if self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal
    
    def get_configuration(self) -> dict:
        return self.config.copy()

    def update_configuration(self, new_config: dict) -> dict:
        validated_config = {}
        
        if 'chunk_size' in new_config:
            chunk_size = int(new_config['chunk_size'])
            if 100 <= chunk_size <= 5000:
                validated_config['chunk_size'] = chunk_size
        
        if 'chunk_overlap' in new_config:
            chunk_overlap = int(new_config['chunk_overlap'])
            current_chunk_size = validated_config.get('chunk_size', self.config['chunk_size'])
            if 0 <= chunk_overlap <= current_chunk_size // 2:
                validated_config['chunk_overlap'] = chunk_overlap
        
        if 'temperature' in new_config:
            temperature = float(new_config['temperature'])
            if 0.0 <= temperature <= 2.0:
                validated_config['temperature'] = temperature
        
        if 'model_name' in new_config:
            model_name = str(new_config['model_name'])
            if model_name in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']:
                validated_config['model_name'] = model_name

        if validated_config:
            self.config.update(validated_config)
            
            if 'temperature' in validated_config or 'model_name' in validated_config:
                self._ensure_clients_initialized()
                self.llm = ChatOpenAI(
                    temperature=self.config['temperature'], 
                    model_name=self.config['model_name']
                )

            if 'chunk_size' in validated_config or 'chunk_overlap' in validated_config:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config['chunk_size'],
                    chunk_overlap=self.config['chunk_overlap'],
                    length_function=len,
                )

        return self.config.copy()

    def get_parameter_explanations(self) -> dict:
        return {
            'chunk_size': {
                'current': self.config['chunk_size'],
                'description': 'Size of text chunks (100-5000 chars)',
                'note': 'Larger = more context, smaller = more granular'
            },
            'chunk_overlap': {
                'current': self.config['chunk_overlap'],
                'description': 'Overlap between chunks',
                'note': 'Helps preserve context at boundaries'
            },
            'temperature': {
                'current': self.config['temperature'],
                'description': 'Response randomness (0.0-2.0)',
                'note': 'Lower = more consistent, higher = more creative'
            },
            'model_name': {
                'current': self.config['model_name'],
                'description': 'OpenAI model to use',
                'note': 'gpt-3.5-turbo is cheapest, gpt-4 is most accurate'
            }
        }

rag_service = RAGService()
