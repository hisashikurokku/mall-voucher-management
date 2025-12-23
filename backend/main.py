from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import httpx
from urllib.parse import quote_plus

app = FastAPI()

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "test" # replace with actual db name

client = AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]

def convert_to_json_safe(doc):
    if doc is None:
        return None

    new_doc = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            new_doc[k] = str(v)
        elif isinstance(v, datetime):
            new_doc[k] = v.isoformat()
        else:
            new_doc[k] = v
    return new_doc

async def send_session_to_web(
    plate_number: str,
    in_time: datetime,
    out_time: datetime | None = None,
):
    payload = {
        "plateNumber": plate_number,
        "inTime": in_time.isoformat(),
    }

    if out_time:
        payload["outTime"] = out_time.isoformat()

    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(
            "http://localhost:3000/api/device/session",
            json=payload,
        )
        r.raise_for_status()
        return r.json()

class PlateEvent(BaseModel):
    plate_number: str
    camera_id: str = "entrance_1"
    detected_at: datetime = datetime.utcnow()
    confidence: float | None = None

@app.post("/api/events/plate-detected")
async def plate_detected(event: PlateEvent):
    plate_number = event.plate_number
    detected_at = event.detected_at

    plate = await db.plates.find_one({"plate_number": plate_number})

    if plate is None:
        new_customer_id = await generate_next_customer_id()
        await db.plates.insert_one({
            "plate_number": plate_number,
            "customer_id": new_customer_id,
            "created_at": datetime.utcnow(),
        })

    last_visit = await db.visits.find_one(
        {"plate_number": plate_number},
        sort=[("detected_at", -1)],
    )

    if last_visit is None or last_visit.get("visit_type") == "exit":
        visit_type = "entry"
        in_time = detected_at
        out_time = None
    else:
        visit_type = "exit"
        in_time = last_visit.get("in_time")
        out_time = detected_at

    visit_doc = {
        "plate_number": plate_number,
        "visit_type": visit_type,
        "detected_at": detected_at,
        "in_time": in_time,
        "out_time": out_time,
    }

    result = await db.visits.insert_one(visit_doc)

    try:
        await send_session_to_web(
            plate_number=plate_number,
            in_time=in_time,
            out_time=out_time,
        )
        sent_to_web = True
    except Exception as e:
        print(f"[WARN] Failed to send session to web: {e}")
        sent_to_web = False

    return {
        "visit_id": str(result.inserted_id),
        "plate_number": plate_number,
        "visit_type": visit_type,
        "in_time": in_time.isoformat() if in_time else None,
        "out_time": out_time.isoformat() if out_time else None,
        "sent_to_web": sent_to_web,
    }

@app.get("/api/admin/visits")
async def admin_get_visits(
    plate_number: str | None = None,
    limit: int = 50
):
    query = {}
    if plate_number:
        query["plate_number"] = plate_number

    cursor = db.visits.find(query).sort("detected_at", -1).limit(limit)

    visits = []
    async for doc in cursor:
        visits.append(convert_to_json_safe(doc))

    return {"visits": visits}

async def generate_next_customer_id():
    doc = await db.plates.find({"customer_id": {"$exists": True}}).sort("customer_id", -1).limit(1).to_list(1)
    
    if not doc:
        return "001"

    last_id = doc[0].get("customer_id", "000")
    num = int(last_id)
    return str(num + 1).zfill(3)

class CustomerAssign(BaseModel):
    customer_id: str

@app.patch("/api/admin/plate/{plate_number}")
async def admin_assign_customer(plate_number: str, data: CustomerAssign):
    formatted_id = ''.join(ch for ch in data.customer_id if ch.isdigit()).zfill(3)

    result = await db.plates.update_one(
        {"plate_number": plate_number},
        {"$set": {"customer_id": formatted_id}}
    )

    if result.matched_count == 0:
        return {"error": "Plate not found"}

    return {
        "plate_number": plate_number,
        "customer_id": formatted_id,
        "message": "Customer assigned successfully"
    }
