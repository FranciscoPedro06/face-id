from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
import uuid
import os
import tempfile
import logging
from supabase import create_client, Client

# -------------------------------
# CONFIGURAÇÃO (via variáveis de ambiente — nunca coloque chaves no código)
# -------------------------------
# SUPABASE_URL  -> URL do projeto Supabase
# SUPABASE_KEY  -> chave do Supabase. Fica SÓ no servidor (Railway etc.), nunca no repositório.
# API_KEY       -> (opcional) se definida, /cadastrar e /usuarios exigem o header X-API-Key
# ALLOWED_ORIGINS -> sites que podem chamar a API, separados por vírgula (padrão: *)
# MAX_UPLOAD_MB -> tamanho máximo da imagem enviada (padrão: 5)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Defina as variáveis de ambiente SUPABASE_URL e SUPABASE_KEY.")

API_KEY = os.environ.get("API_KEY", "")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "5")) * 1024 * 1024)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# FASTAPI
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="EduPass - Reconhecimento Facial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Credenciais (cookies) não são usadas por esta API; com "*" o navegador as recusaria de qualquer forma.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def exigir_api_key(x_api_key: str = Header(default="")):
    """Protege rotas sensíveis quando API_KEY estiver definida no servidor."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Não autorizado.")


async def salvar_imagem_temporaria(file: UploadFile) -> str:
    """Valida a imagem e grava num arquivo temporário seguro. Devolve o caminho."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie uma imagem válida.")

    conteudo = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(conteudo) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Imagem muito grande.")
    if not conteudo:
        raise HTTPException(status_code=400, detail="Imagem vazia.")

    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(conteudo)
    return temp_path


def remover(path: str):
    try:
        os.remove(path)
    except OSError:
        pass

# -----------------------------------------
# ROTAS
# -----------------------------------------

@app.get("/")
async def health():
    # Conta usuários direto do Supabase
    result = supabase.table("usuarios_face").select("id", count="exact").execute()
    total = result.count if result.count is not None else 0

    return {"status": "online", "usuarios_cadastrados": total}


@app.post("/cadastrar", dependencies=[Depends(exigir_api_key)])
async def cadastrar(nome: str = Form(..., max_length=120), file: UploadFile = File(...)):

    temp_path = await salvar_imagem_temporaria(file)

    try:
        # Modelo de embedding
        embedding_obj = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet512",
            detector_backend="ssd",
            enforce_detection=True
        )[0]

        embedding = embedding_obj["embedding"]

        user_id = str(uuid.uuid4())

        # Salvar no Supabase
        data = {
            "id": user_id,
            "nome": nome,
            "embedding": embedding,
            "embedding_size": len(embedding),
        }

        supabase.table("usuarios_face").insert(data).execute()

        return {
            "success": True,
            "mensagem": f"{nome} cadastrado com sucesso!",
            "id": user_id
        }

    except Exception:
        # O detalhe vai só para o log do servidor, nunca para quem chamou a API.
        logger.exception("Falha ao cadastrar rosto")
        return {"success": False, "error": "Não foi possível cadastrar. Verifique se há um rosto nítido na foto."}

    finally:
        remover(temp_path)


@app.post("/reconhecer")
async def reconhecer(file: UploadFile = File(...)):

    temp_path = await salvar_imagem_temporaria(file)

    try:
        # Buscar embeddings
        result = supabase.table("usuarios_face").select("id, nome, embedding").execute()
        usuarios = result.data

        if not usuarios:
            return {"success": False, "error": "Nenhum usuário cadastrado ainda."}

        try:
            embedding_obj = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=True
            )[0]

            embedding_atual = np.array(embedding_obj["embedding"])

        except Exception:
            return {"success": False, "error": "Não foi detectado rosto."}

    finally:
        remover(temp_path)

    menor_dist = float("inf")
    usuario_final = None

    # Comparação com TODOS do Supabase
    for user in usuarios:
        emb = np.array(user["embedding"])

        dot = np.dot(embedding_atual, emb)
        normA = np.linalg.norm(embedding_atual)
        normB = np.linalg.norm(emb)
        dist = 1 - (dot / (normA * normB))

        if dist < menor_dist:
            menor_dist = dist
            usuario_final = user

    THRESHOLD = 0.35

    if menor_dist < THRESHOLD:
        confianca = (1 - menor_dist) * 100
        return {
            "success": True,
            "mensagem": "Reconhecido",
            # Nunca devolver o embedding (dado biométrico) — só o necessário.
            "usuario": {"id": usuario_final["id"], "nome": usuario_final["nome"]},
            "distancia": round(float(menor_dist), 4),
            "confianca": round(float(confianca), 2)
        }

    return {
        "success": False,
        "mensagem": "Não reconhecido",
        "distancia": round(float(menor_dist), 4)
    }


@app.get("/usuarios", dependencies=[Depends(exigir_api_key)])
async def usuarios():
    result = supabase.table("usuarios_face").select("id, nome, data_cadastro").execute()

    return {"usuarios": result.data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
