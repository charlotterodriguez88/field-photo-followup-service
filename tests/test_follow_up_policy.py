from pathlib import Path

from fastapi.testclient import TestClient

from field_photo_service import FieldPhotoGenerator, build_app


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, brief: str, request_id: str) -> str:
        self.calls += 1
        return "https://images.example.test/visit-204.png"


def test_created_work_order_cannot_receive_follow_up_photo(
    tmp_path: Path, monkeypatch
) -> None:
    generator = RecordingGenerator()
    app = build_app(tmp_path / "photos.sqlite3")
    monkeypatch.setenv("INFRAI_API_KEY", "test-key")
    monkeypatch.setattr(
        FieldPhotoGenerator,
        "generate",
        lambda self, brief, request_id: generator.generate(brief, request_id),
    )

    response = TestClient(app).post(
        "/follow-up-photos",
        json={
            "work_order_id": "WO-204",
            "dispatch_status": "created",
            "technician_note": "Panel inspected at intake.",
            "image_brief": "A clear service-panel inspection photo without people.",
        },
    )

    assert response.status_code == 409
    assert generator.calls == 0


def test_dispatched_follow_up_is_generated_once_and_reused(
    tmp_path: Path, monkeypatch
) -> None:
    generator = RecordingGenerator()
    app = build_app(tmp_path / "photos.sqlite3")
    monkeypatch.setenv("INFRAI_API_KEY", "test-key")
    monkeypatch.setattr(
        FieldPhotoGenerator,
        "generate",
        lambda self, brief, request_id: generator.generate(brief, request_id),
    )
    client = TestClient(app)
    payload = {
        "work_order_id": "WO-204",
        "dispatch_status": "dispatched",
        "technician_note": "Panel condition documented before service.",
        "image_brief": "A clear service-panel inspection photo without people.",
    }

    first = client.post("/follow-up-photos", json=payload)
    second = client.post("/follow-up-photos", json=payload)

    assert first.status_code == 201
    assert first.json() == {
        "work_order_id": "WO-204",
        "dispatch_status": "dispatched",
        "technician_note": "Panel condition documented before service.",
        "image_url": "https://images.example.test/visit-204.png",
    }
    assert second.json() == first.json()
    assert generator.calls == 1
