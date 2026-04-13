# BlindAR — Backend API

Gateway de segurança para IA via óculos AR (Petrobras).  
FastAPI + Supabase + Google Gemini.

## Setup Rápido

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edite .env com suas credenciais
python main.py
```

O servidor inicia em `http://localhost:8000`.  
Documentação interativa: `http://localhost:8000/docs`

## Supabase

1. Crie um projeto em [supabase.com](https://supabase.com)
2. Abra o **SQL Editor** e execute o conteúdo de `supabase_schema.sql`
3. Copie a **URL** e **anon key** do projeto para o `.env`

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Status da API |
| `GET` | `/health` | Health check |
| `POST` | `/api/users/register` | Cadastro facial |
| `POST` | `/api/verify` | Verificação de identidade |
| `POST` | `/api/chat` | Gateway IA (Gemini) |
| `POST` | `/api/events` | Registrar evento |
| `GET` | `/api/events` | Listar eventos |
| `GET` | `/api/dashboard` | Estatísticas |

## Arquitetura

```
Óculos AR (HoloLens 2)
    │
    ├── Camera → face-api.js → /api/verify (verificação contínua)
    ├── Chat   → /api/chat   → Gemini API (filtrado por nível)
    └── Logs   → /api/events → Supabase (Sentinel)
```
