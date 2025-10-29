import os
import atexit
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from openai import OpenAI

def cleanup_multiprocessing():
    try:
        import multiprocessing
        for process in multiprocessing.active_children():
            process.terminate()
            process.join()
    except:
        pass

# Register cleanup function
atexit.register(cleanup_multiprocessing)

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
        """
        Process PDF file using docling to extract text and structure.
        """
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            
            pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pdf_format_option
                }
            )
            
            result = converter.convert(pdf_file_path)
            
            extracted_text = result.document.export_to_markdown()
            
            doc_structure = result.document
            
            page_texts = []
            page_boundaries = []
            char_count = 0
            
            if hasattr(doc_structure, 'pages') and doc_structure.pages:
                for page_num, page_obj in enumerate(doc_structure.pages, 1):
                    try:
                        page_text = ""
                        if hasattr(page_obj, 'export_to_markdown'):
                            page_text = page_obj.export_to_markdown()
                        elif hasattr(page_obj, 'text'):
                            page_text = page_obj.text
                        elif hasattr(page_obj, 'content'):
                            if hasattr(page_obj.content, 'export_to_markdown'):
                                page_text = page_obj.content.export_to_markdown()
                            else:
                                page_text = str(page_obj.content)
                        else:
                            page_text = str(page_obj)
                        
                        if page_text:
                            start_char = char_count
                            char_count += len(page_text)
                            page_texts.append(page_text)
                            page_boundaries.append((start_char, char_count))
                    except Exception as e:
                        print(f"Warning: Could not extract page {page_num}: {e}")
                        continue
            
            if not page_boundaries:
                lines = extracted_text.split('\n')
                lines_per_page = LINES_PER_PAGE_ESTIMATE
                num_pages = max(1, (len(lines) + lines_per_page - 1) // lines_per_page)
                
                for i in range(num_pages):
                    start_line = i * lines_per_page
                    end_line = min((i + 1) * lines_per_page, len(lines))
                    page_lines = lines[start_line:end_line]
                    page_text = '\n'.join(page_lines) + '\n'
                    
                    start_char = char_count
                    char_count += len(page_text)
                    page_boundaries.append((start_char, char_count))
            
            documents = self.text_splitter.create_documents([extracted_text])
            
            for i, doc in enumerate(documents):
                chunk_text = doc.page_content
                
                chunk_start_text = chunk_text[:50]
                char_offset = extracted_text.find(chunk_start_text)
                
                if char_offset >= 0:
                    current_page = 1
                    for page_idx, (start, end) in enumerate(page_boundaries, 1):
                        if start <= char_offset < end:
                            current_page = page_idx
                            break
                        elif char_offset < start:
                            break
                else:
                    lines = extracted_text.split('\n')
                    estimated_page = self._estimate_page_number(chunk_text, lines, i // len(page_boundaries) if page_boundaries else 1)
                    current_page = estimated_page
                
                section = self._detect_section(chunk_text)
                
                doc.metadata['page'] = current_page
                doc.metadata['section'] = section
                doc.metadata['chunk_id'] = i + 1
                doc.metadata['source'] = 'pdf'
            
            return documents, extracted_text
                
        except Exception as e:
            raise Exception(f"Failed to extract PDF with docling: {str(e)}")

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
