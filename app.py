from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
import uuid
import os
from datetime import datetime
from supabase import create_client, Client
import logging

# -------------------------------
# CONFIGURAÇÃO SUPABASE
# -------------------------------
SUPABASE_URL = "https://ywwwcnolqehepqukbrdp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl3d3djbm9scWVoZXBxdWticmRwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1MTU2NzksImV4cCI6MjA3OTA5MTY3OX0.OXty613JpAwHxt1oFldArwDWhEMZrd8EO5SI0MhPFkI"   # ⚠️ coloque a service_role, não a anon!

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# FASTAPI
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="EduPass - Reconhecimento Facial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------
# ROTAS
# -----------------------------------------

@app.get("/")
async def health():
    # Conta usuários direto do Supabase
    result = supabase.table("usuarios_face").select("id", count="exact").execute()
    total = result.count if result.count is not None else 0

    return {"status": "online", "usuarios_cadastrados": total}


@app.post("/cadastrar")
async def cadastrar(nome: str = Form(...), file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        return {"success": False, "error": "Envie uma imagem válida."}

    # Cria arquivo temporário
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

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

        os.remove(temp_path)

        return {
            "success": True,
            "mensagem": f"{nome} cadastrado com sucesso!",
            "id": user_id
        }

    except Exception as e:
        os.remove(temp_path)
        return {"success": False, "error": str(e)}


@app.post("/reconhecer")
async def reconhecer(file: UploadFile = File(...)):

    # Buscar embeddings
    result = supabase.table("usuarios_face").select("*").execute()
    usuarios = result.data

    if not usuarios:
        return {"success": False, "error": "Nenhum usuário cadastrado ainda."}

    # Upload da imagem temporária
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        embedding_obj = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True
        )[0]

        embedding_atual = np.array(embedding_obj["embedding"])

    except Exception:
        os.remove(temp_path)
        return {"success": False, "error": "Não foi detectado rosto."}

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

    os.remove(temp_path)

    THRESHOLD = 0.35

    if menor_dist < THRESHOLD:
        confianca = (1 - menor_dist) * 100
        return {
            "success": True,
            "mensagem": "Reconhecido",
            "usuario": usuario_final,
            "distancia": round(menor_dist, 4),
            "confianca": round(confianca, 2)
        }

    return {
        "success": False,
        "mensagem": "Não reconhecido",
        "distancia": round(menor_dist, 4)
    }


@app.get("/usuarios")
async def usuarios():
    result = supabase.table("usuarios_face").select("id, nome, data_cadastro").execute()

    return {"usuarios": result.data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
