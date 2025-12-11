from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
import os
import uuid
from bson import ObjectId


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

class PlateEvent(BaseModel):
    plate_number: str
    camera_id: str = "entrance_1"
    detected_at: datetime = datetime.utcnow()
    confidence: float | None = None

@app.post("/api/events/plate-detected")
async def plate_detected(event: PlateEvent):
    plate = await db.plates.find_one({"plate_number": event.plate_number})

    if plate is None:
        new_customer_id = await generate_next_customer_id()
        await db.plates.insert_one({
            "plate_number": event.plate_number,
            "customer_id": new_customer_id,
            "created_at": datetime.utcnow()
        })
    else:
        new_customer_id = plate.get("customer_id")

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

    # Build visit document
    visit_doc = event.model_dump()
    visit_doc["visit_type"] = visit_type

    # Insert visit log
    result = await db.visits.insert_one(visit_doc)

    # If it's an ENTRY -> attempt voucher creation
    created_voucher = None
    if visit_type == "entry":
        created_voucher = await generate_voucher(
            event.plate_number, event.detected_at
        )

    return {
        "visit_id": str(result.inserted_id),
        "visit_type": visit_type,
        "voucher_created": convert_to_json_safe(created_voucher),
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

@app.get("/api/admin/vouchers")
async def admin_get_vouchers(
    plate_number: str | None = None,
    status: str | None = None,
    limit: int = 50
):
    query = {}

    if plate_number:
        query["plate_number"] = plate_number

    if status:
        query["status"] = status  # active / used / expired

    cursor = db.vouchers.find(query).sort("created_at", -1).limit(limit)

    vouchers = []
    async for doc in cursor:
        vouchers.append(convert_to_json_safe(doc))

    return {"vouchers": vouchers}

async def generate_voucher(plate_number: str, detected_at: datetime):
    day_start = datetime(detected_at.year, detected_at.month, detected_at.day)
    day_end = day_start + timedelta(days=1)

    existing = await db.vouchers.find_one({
        "plate_number": plate_number,
        "created_at": {"$gte": day_start, "$lt": day_end}
    })

    if existing:
        return None

    voucher_code = f"{detected_at.strftime('%Y%m%d')}-{plate_number.replace('-', '').replace('.', '')}"

    voucher = {
        "plate_number": plate_number,
        "voucher_code": voucher_code,
        "discount_type": "percentage",
        "discount_value": 10,
        "created_at": detected_at,
        "valid_from": detected_at,
        "valid_until": day_end,
        "status": "active"
    }

    result = await db.vouchers.insert_one(voucher)
    voucher["_id"] = result.inserted_id
    return voucher 

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
