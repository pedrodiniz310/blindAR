"""Quick check — what's in Supabase right now?"""
from dotenv import load_dotenv
import os
load_dotenv()
from supabase import create_client

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

print("=" * 50)
print("USERS")
print("=" * 50)
r = sb.table("users").select("id, name, role, registered_at").execute()
print(f"Total: {len(r.data)}")
for u in r.data:
    print(f"  - {u['name']} ({u['role']}) | {u['registered_at']}")

print()
print("=" * 50)
print("SESSIONS")
print("=" * 50)
r2 = sb.table("sessions").select("*").execute()
print(f"Total: {len(r2.data)}")
for s in r2.data:
    print(f"  - user_id: {s.get('user_id')} | level: {s.get('security_level')} | locked: {s.get('is_locked')}")

print()
print("=" * 50)
print("EVENTS (últimos 10)")
print("=" * 50)
r3 = sb.table("events").select("event_type, description, severity, created_at").order("created_at", desc=True).limit(10).execute()
print(f"Total no banco: (mostrando 10 mais recentes)")
for e in r3.data:
    print(f"  [{e['severity'].upper():8s}] {e['event_type']:20s} | {e['description'][:60]}")
