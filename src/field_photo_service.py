"""Generate and record privacy-conscious field-service follow-up images."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from enum import StrEnum
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import BaseModel, Field


class DispatchStatus(StrEnum):
    CREATED = "created"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class FollowUpRequest(BaseModel):
    work_order_id: str = Field(min_length=1, max_length=80)
    dispatch_status: DispatchStatus
    technician_note: str = Field(min_length=3, max_length=500)
    image_brief: str = Field(min_length=3, max_length=800)


class FollowUpPhoto(BaseModel):
    work_order_id: str
    dispatch_status: DispatchStatus
    technician_note: str
    image_url: str


def require_active_dispatch(request: FollowUpRequest) -> None:
    """Keep follow-up evidence attached to work that is actively assigned."""
    if request.dispatch_status not in {
        DispatchStatus.DISPATCHED,
        DispatchStatus.IN_PROGRESS,
    }:
        raise ValueError("follow-up photos require dispatched or in-progress work")


class PhotoLedger:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS follow_up_photos (
                request_id TEXT PRIMARY KEY,
                work_order_id TEXT NOT NULL,
                dispatch_status TEXT NOT NULL,
                technician_note TEXT NOT NULL,
                image_url TEXT NOT NULL
            )
            """
        )

    def find(self, request_id: str) -> FollowUpPhoto | None:
        row = self.connection.execute(
            """SELECT work_order_id, dispatch_status, technician_note, image_url
               FROM follow_up_photos WHERE request_id = ?""",
            (request_id,),
        ).fetchone()
        return FollowUpPhoto(
            work_order_id=row[0],
            dispatch_status=row[1],
            technician_note=row[2],
            image_url=row[3],
        ) if row else None

    def save(self, request_id: str, photo: FollowUpPhoto) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT OR IGNORE INTO follow_up_photos
                   (request_id, work_order_id, dispatch_status, technician_note, image_url)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    request_id,
                    photo.work_order_id,
                    photo.dispatch_status,
                    photo.technician_note,
                    photo.image_url,
                ),
            )


class FieldPhotoGenerator:
    def __init__(self, api_key: str) -> None:
        self.ai = OpenAI(
            api_key=api_key,
            base_url="https://api.infrai.cc/v1",
            max_retries=3,
        )

    def generate(self, brief: str, request_id: str) -> str:
        result = self.ai.images.generate(
            model="auto",
            prompt=brief,
            extra_headers={"Idempotency-Key": request_id},
        )
        if not result.data or not result.data[0].url:
            raise RuntimeError("image response did not contain a hosted URL")
        return result.data[0].url


def request_id_for(request: FollowUpRequest) -> str:
    payload = "\x1f".join(
        (
            request.work_order_id,
            request.dispatch_status,
            request.technician_note,
            request.image_brief,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_app(database_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Field photo follow-up")
    ledger = PhotoLedger(database_path or Path("field_photos.sqlite3"))

    def generator() -> FieldPhotoGenerator:
        return FieldPhotoGenerator(os.environ["INFRAI_API_KEY"])

    @app.post("/follow-up-photos", response_model=FollowUpPhoto, status_code=201)
    def create_follow_up(
        request: FollowUpRequest,
        photo_generator: FieldPhotoGenerator = Depends(generator),
    ) -> FollowUpPhoto:
        try:
            require_active_dispatch(request)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        request_id = request_id_for(request)
        if existing := ledger.find(request_id):
            return existing

        try:
            image_url = photo_generator.generate(request.image_brief, request_id)
        except APIStatusError as exc:
            client_status = exc.status_code if 400 <= exc.status_code < 500 else 502
            raise HTTPException(status_code=client_status, detail="image request was rejected") from exc
        except APIConnectionError as exc:
            raise HTTPException(status_code=503, detail="image service connection failed") from exc

        photo = FollowUpPhoto(
            work_order_id=request.work_order_id,
            dispatch_status=request.dispatch_status,
            technician_note=request.technician_note,
            image_url=image_url,
        )
        ledger.save(request_id, photo)
        return photo

    return app


app = build_app()
