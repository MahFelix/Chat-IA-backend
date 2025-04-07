import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import uvicorn

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')  

# System Prompt como constante
SYSTEM_PROMPT = """
Você é **Aether**, um assistente técnico altamente especializado em desenvolvimento de software, sempre à frente nas tendências tecnológicas. Seu objetivo é fornecer respostas claras, precisas e otimizadas para desenvolvedores que buscam soluções rápidas e eficientes. Mantenha o foco em qualidade e organização.

### Diretrizes de Estilo:

1. **Formatação e Clareza**:
   - Use Markdown para organizar suas respostas:
     - **Códigos**: Utilize ```linguagem\nseu_codigo``` para blocos e `codigo` para trechos inline.
     - **Negrito**: **Texto** para enfatizar palavras ou frases.
     - **Itálico**: *Texto* para destacar de maneira sutil.
     - **Listas**: Utilize - ou 1. para listas ordenadas.
     - **Títulos**: # para título principal, ## para subtítulos.

2. **Blocos de Código**:
   - Especifique sempre a linguagem após os três crases (ex: ```javascript, ```python).
   - Preserve a indentação correta do código.
   - Exemplo:
     ```python
     print("Hello, Aether!")
     ```

3. **Seja Conciso, Mas Completo**:
   - Explique conceitos complexos de forma objetiva, usando parágrafos curtos.
   - Para respostas longas, divida em seções e subtítulos.
   - Ao listar etapas ou soluções, prefira números ou listas com marcadores.

4. **Exemplo de Resposta Ideal**:
   - Forneça uma solução técnica e passo a passo, com código bem estruturado, para uma dúvida comum de desenvolvimento. Por exemplo, ao explicar como configurar um Service Worker, você deve:
     1. Explicar o conceito e a importância do Service Worker.
     2. Apresentar um código de exemplo simples.
     3. Dividir a explicação em etapas fáceis de seguir, como instalação, configuração e ativação.

Lembre-se, Aether, sua missão é garantir que a resposta não apenas resolva o problema, mas também eduque, sempre mantendo o usuário no caminho do aprendizado contínuo. ✨
"""


class ChatRequest(BaseModel):
    message: str
    context: list = []

def prepare_messages(user_message: str, context: list = None):
    """Prepara o histórico de mensagens incluindo o system prompt"""
    messages = []
    
    # Adiciona system prompt como primeira mensagem
    messages.append({
        "role": "user",  # Gemini trata system prompt como user message
        "parts": [SYSTEM_PROMPT]
    })
    
    # Adiciona contexto histórico se existir
    if context:
        for msg in context:
            if msg.get("text"):
                role = "user" if msg.get("sender") == "user" else "model"
                messages.append({
                    "role": role,
                    "parts": [msg.get("text")]
                })
    
    # Adiciona a nova mensagem do usuário
    messages.append({
        "role": "user",
        "parts": [user_message]
    })
    
    return messages

def call_gemini_with_retry(messages: list, max_retries: int = 3, initial_delay: int = 30):
    """Chama a API Gemini com retry automático para rate limits"""
    for attempt in range(max_retries):
        try:
            # Para histórico de conversa
            if len(messages) > 2:  # Mais que system prompt + user message
                # Primeira mensagem é o system prompt
                chat = model.start_chat(history=messages[:-1])
                response = chat.send_message(messages[-1]["parts"][0])
            else:
                # Nova conversa (inclui system prompt)
                response = model.generate_content(messages)
            
            return response
        
        except Exception as e:
            if "429" in str(e):
                wait_time = initial_delay * (attempt + 1)
                logger.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Gemini API error: {str(e)}")
                raise
    
    raise Exception("Max retries reached for Gemini API")


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Service is healthy"}

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        logger.info(f"Received message: {request.message[:100]}...")  # Log parcial
        
        # Prepara todas as mensagens com system prompt
        messages = prepare_messages(
            user_message=request.message,
            context=request.context
        )
        
        # Chama Gemini com retry automático
        response = call_gemini_with_retry(messages)
        
        # Verifica se a resposta é válida
        if not response.text:
            raise ValueError("Empty response from Gemini API")
        
        logger.info(f"Generated reply: {response.text[:100]}...")  # Log parcial
        return {"reply": response.text}
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 5000)),
        log_level="debug"
    )