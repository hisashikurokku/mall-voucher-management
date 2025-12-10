from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
import os

app = FastAPI()

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "test"

client = AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]

class PlateEvent(BaseModel):
    plate_number: str
    camera_id: str = "entrance_1"
    detected_at: datetime = datetime.utcnow()
    confidence: float | None = None

@app.post("/api/events/plate-detected")
async def plate_detected(event: PlateEvent):
    # Insert plate if not existed
    await db.plates.update_one(
        {"plate_number": event.plate_number},
        {"$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True
    )

    last_visit = await db.visits.find_one(
        {"plate_number": event.plate_number},
        sort=[("detected_at", -1)]
    )

    if last_visit is None:
        visit_type = "entry"
    elif last_visit.get("visit_type") == "entry":
        visit_type = "exit"
    else:
        visit_type = "entry"

    # Insert visit log
    visit_doc = event.model_dump()
    visit_doc["visit_type"] = visit_type
    result = await db.visits.insert_one(visit_doc)

    return {
        "inserted_id": str(result.inserted_id),
        "visit_type": visit_type,
        "message": f"{event.plate_number} marked as {visit_type}"
    }
