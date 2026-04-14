"""
══════════════════════════════════════════════════════════════
 BlindAR — Backend API (FastAPI + Supabase)
 Gateway de Segurança para IA via Óculos AR — Petrobras
 Grand Prix SENAI de Inovação 2026
══════════════════════════════════════════════════════════════
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client

load_dotenv()

# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")]

MATCH_THRESHOLD = 0.5  # Euclidean distance threshold for face matching
GEMINI_MODEL = "gemini-2.0-flash"
MAX_DESCRIPTOR_DIM = 128

# ──────────────────────────────────────────────────────────────
#  Supabase Client
# ──────────────────────────────────────────────────────────────
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ──────────────────────────────────────────────────────────────
#  FastAPI App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="BlindAR API",
    description="Gateway de segurança para IA — Protótipo Petrobras",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
#  Models (Pydantic)
# ──────────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    role: str = Field(default="Colaborador", max_length=100)
    face_descriptor: list[float] = Field(..., min_length=MAX_DESCRIPTOR_DIM, max_length=MAX_DESCRIPTOR_DIM)

class UserResponse(BaseModel):
    id: str
    name: str
    role: str
    registered_at: str
    token: str

class VerifyFace(BaseModel):
    face_descriptor: list[float] = Field(..., min_length=MAX_DESCRIPTOR_DIM, max_length=MAX_DESCRIPTOR_DIM)
    user_id: str

class VerifyResponse(BaseModel):
    match: bool
    similarity: float
    distance: float

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    security_level: int = Field(..., ge=1, le=5)
    user_name: str = Field(default="Desconhecido")
    user_role: str = Field(default="N/A")

class ChatResponse(BaseModel):
    response: str
    level: int
    filtered: bool
    mode: str

class SecurityEvent(BaseModel):
    event_type: str = Field(..., max_length=50)
    description: str = Field(..., max_length=500)
    severity: str = Field(default="info", pattern="^(info|warning|critical)$")
    user_id: Optional[str] = None
    metadata: Optional[dict] = None

class DashboardStats(BaseModel):
    total_queries: int
    total_alerts: int
    active_devices: int
    events: list[dict]


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Calculate euclidean distance between two face descriptors."""
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    return float(np.linalg.norm(arr_a - arr_b))


def generate_token(user_id: str) -> str:
    """Generate a simple session token (use JWT in production)."""
    raw = f"{user_id}:{JWT_SECRET}:{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_system_prompt(level: int, user_name: str, user_role: str) -> str:
    """Generate the Gemini system prompt based on security level."""
    return f"""Você é o assistente de IA corporativo da Petrobras, protegido pelo sistema de segurança BlindAR.
Você opera dentro de um dispositivo de realidade aumentada (óculos AR) usado por colaboradores em campo.

CONTEXTO DE SEGURANÇA ATUAL:
- Nível de segurança: {level} de 5
- Usuário autenticado: {user_name}
- Cargo: {user_role}

REGRAS DE RESPOSTA POR NÍVEL:

NÍVEL 1 (Verde — Sala segura + Rede interna):
- Responda com TODOS os dados disponíveis, incluindo valores exatos, nomes, IDs, contatos.
- Sem restrições. Forneça relatórios completos.

NÍVEL 2 (Azul — Área de trabalho + Rede interna):
- Forneça dados, mas SEM informações pessoais detalhadas (IDs, emails, telefones).
- Mencione que watermark de rastreabilidade foi aplicado.

NÍVEL 3 (Amarelo — Campo + Rede de terceiro):
- Forneça APENAS informações genéricas. NUNCA valores exatos.
- Em vez de "42.7 bar", diga "dentro dos parâmetros normais".
- Ao final, diga: "Para dados detalhados, conecte-se a uma rede segura."

NÍVEL 4 (Laranja — Área pública / Observador detectado):
- NÃO forneça NENHUM dado operacional.
- Responda apenas: "Esta informação requer um ambiente seguro."

NÍVEL 5 (Vermelho — Dispositivo comprometido):
- Responda APENAS: "❌ Acesso negado. Sessão bloqueada por razões de segurança."

CONTEXTO DA PETROBRAS:
- Produção: ~2.15 milhões de barris/dia
- Plataformas ativas: 47-52
- Áreas: Bacia de Santos, Bacia de Campos
- Equipamentos: compressores, bombas, turbogeradores, válvulas PSV
- Poços: P-47, P-76, P-82
- Faixa de pressão: 38-45 bar

Responda em português brasileiro. Seja técnico mas acessível. 3-8 linhas."""


# ──────────────────────────────────────────────────────────────
#  Routes — Health
# ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "BlindAR API",
        "version": "1.0.0",
        "status": "online",
        "supabase_connected": supabase is not None,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


# ──────────────────────────────────────────────────────────────
#  Routes — User Registration
# ──────────────────────────────────────────────────────────────
@app.post("/api/users/register", response_model=UserResponse)
async def register_user(user: UserRegister):
    """Register a new user with face descriptor."""
    now = datetime.now(timezone.utc).isoformat()

    if supabase:
        result = supabase.table("users").insert({
            "name": user.name,
            "role": user.role,
            "face_descriptor": user.face_descriptor,
            "registered_at": now,
        }).execute()
        user_id = result.data[0]["id"]
    else:
        user_id = hashlib.sha256(f"{user.name}:{now}".encode()).hexdigest()[:12]

    token = generate_token(user_id)

    # Log the registration event
    await log_event(SecurityEvent(
        event_type="user_registered",
        description=f"Usuário '{user.name}' cadastrado com biometria facial",
        severity="info",
        user_id=str(user_id),
    ))

    return UserResponse(
        id=str(user_id),
        name=user.name,
        role=user.role,
        registered_at=now,
        token=token,
    )


# ──────────────────────────────────────────────────────────────
#  Routes — Face Verification
# ──────────────────────────────────────────────────────────────
@app.post("/api/verify", response_model=VerifyResponse)
async def verify_face(data: VerifyFace):
    """Verify a face descriptor against the registered user."""
    if supabase:
        result = supabase.table("users").select("face_descriptor").eq("id", data.user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        stored_descriptor = result.data[0]["face_descriptor"]
    else:
        raise HTTPException(status_code=503, detail="Supabase não configurado — verificação local apenas")

    distance = euclidean_distance(data.face_descriptor, stored_descriptor)
    similarity = max(0.0, 1.0 - distance)
    is_match = distance < MATCH_THRESHOLD

    severity = "info" if is_match else "critical"
    await log_event(SecurityEvent(
        event_type="face_verification",
        description=f"Verificação facial: {'✓ Match' if is_match else '✗ Mismatch'} ({similarity*100:.0f}%)",
        severity=severity,
        user_id=data.user_id,
        metadata={"distance": distance, "similarity": similarity, "match": is_match},
    ))

    return VerifyResponse(match=is_match, similarity=similarity, distance=distance)


# ──────────────────────────────────────────────────────────────
#  Routes — AI Chat (Gemini Gateway)
# ──────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Gateway de IA — processa a pergunta aplicando o filtro de segurança
    por nível antes de enviar ao Gemini (ou responder localmente).
    """
    level = req.security_level

    # Level 5: block immediately
    if level == 5:
        return ChatResponse(
            response="❌ Acesso negado. Sessão bloqueada por razões de segurança.",
            level=5,
            filtered=True,
            mode="blocked",
        )

    # Level 4: deny operational data
    if level == 4:
        return ChatResponse(
            response="Esta informação requer um ambiente seguro. Mova-se para uma área autorizada.",
            level=4,
            filtered=True,
            mode="restricted",
        )

    # Levels 1-3: call Gemini or fallback
    if GEMINI_API_KEY:
        try:
            response_text = await call_gemini(req.question, level, req.user_name, req.user_role)
            mode = "gemini"
        except Exception as e:
            response_text = f"⚠️ Erro na API: {e}. Usando resposta local."
            mode = "local_fallback"
    else:
        response_text = get_local_response(req.question, level)
        mode = "local"

    filtered = level >= 3

    # Log the query
    await log_event(SecurityEvent(
        event_type="ai_query",
        description=f"Query nível {level}: {req.question[:80]}...",
        severity="info",
        metadata={"level": level, "mode": mode, "filtered": filtered},
    ))

    return ChatResponse(response=response_text, level=level, filtered=filtered, mode=mode)


async def call_gemini(question: str, level: int, user_name: str, user_role: str) -> str:
    """Call the Google Gemini API with the security-aware system prompt."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    body = {
        "system_instruction": {"parts": [{"text": get_system_prompt(level, user_name, user_role)}]},
        "contents": [{"parts": [{"text": question}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500},
    }

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(url, json=body)

    if res.status_code != 200:
        error_data = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
        msg = error_data.get("error", {}).get("message", f"HTTP {res.status_code}")
        raise RuntimeError(msg)

    data = res.json()
    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    if not text:
        raise RuntimeError("Resposta vazia da API")

    return text


def get_local_response(question: str, level: int) -> str:
    """Generate a local fallback response based on security level."""
    q = question.lower()

    if "pressão" in q or "poço" in q or "p-47" in q:
        if level == 1:
            return "A pressão atual do poço P-47 é de 42.7 bar, dentro da faixa operacional (38-45 bar). Última leitura: agora. Status: Normal."
        if level == 2:
            return "A pressão do poço P-47 é de 42.7 bar — dentro da faixa operacional. Status: Normal."
        return "A pressão do poço P-47 está dentro dos parâmetros normais. Para dados detalhados, conecte-se a uma rede segura."

    if "produção" in q:
        if level == 1:
            return "Produção Março/2026 — Petróleo: 2.15M bpd (↑ 3.2%). Gás natural: 580 mil m³/dia. Meta trimestral: 92.4% atingida."
        if level == 2:
            return "Resumo de produção — Petróleo: 2.15M bpd (↑ 3.2%). Meta: 92.4% atingida."
        return "A produção do mês está acima da meta estabelecida. Para dados detalhados, conecte-se a uma rede segura."

    if "responsável" in q or "quem" in q or "p-76" in q:
        if level == 1:
            return "Responsável P-76: Eng. Carlos Mendes (ID: 15832). Gerente de Plataforma. Contato: c.mendes@petrobras.com.br. Turno A (06-18h)."
        if level == 2:
            return "Responsável P-76: Carlos Mendes. Turno atual: A (06-18h)."
        return "O responsável está em turno ativo. Para detalhes, conecte-se a uma rede segura."

    # Generic
    if level == 1:
        return "Consulta processada com dados completos (Nível 1). Indicadores dentro das especificações operacionais."
    if level == 2:
        return "Consulta processada (Nível 2). Dados principais disponíveis com watermark de rastreabilidade."
    return "Informações gerais: indicadores dentro dos parâmetros. Para dados detalhados, conecte-se a uma rede segura."


# ──────────────────────────────────────────────────────────────
#  Routes — Security Events
# ──────────────────────────────────────────────────────────────
@app.post("/api/events")
async def create_event(event: SecurityEvent):
    """Log a security event."""
    await log_event(event)
    return {"status": "logged"}


@app.get("/api/events", response_model=list[dict])
async def get_events(limit: int = 50):
    """Retrieve recent security events."""
    if supabase:
        result = supabase.table("events").select("*").order("created_at", desc=True).limit(limit).execute()
        return result.data
    return []


async def log_event(event: SecurityEvent):
    """Persist a security event to Supabase."""
    if supabase:
        supabase.table("events").insert({
            "event_type": event.event_type,
            "description": event.description,
            "severity": event.severity,
            "user_id": event.user_id,
            "metadata": event.metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()


# ──────────────────────────────────────────────────────────────
#  Routes — Dashboard
# ──────────────────────────────────────────────────────────────
@app.get("/api/dashboard", response_model=DashboardStats)
async def dashboard():
    """Get aggregated dashboard statistics."""
    if not supabase:
        return DashboardStats(total_queries=0, total_alerts=0, active_devices=1, events=[])

    queries = supabase.table("events").select("id", count="exact").eq("event_type", "ai_query").execute()
    alerts = supabase.table("events").select("id", count="exact").in_("severity", ["warning", "critical"]).execute()
    recent = supabase.table("events").select("*").order("created_at", desc=True).limit(20).execute()

    return DashboardStats(
        total_queries=queries.count or 0,
        total_alerts=alerts.count or 0,
        active_devices=1,
        events=recent.data,
    )


# ──────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
