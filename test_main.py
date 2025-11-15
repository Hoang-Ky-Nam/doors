import json
import os
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select, text
from main import SpaceEvent, app, Space, hash_password
from jsonschema import validate

TEST_DB = "sqlite:///test_database.db"


@pytest.fixture(scope="session", autouse=True)
def test_db():
    # Set up test database
    engine = create_engine(TEST_DB, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    # Patch app's engine to use test engine
    app.dependency_overrides = {}
    app.state._engine = engine

    # Provide a get_session override
    def get_session_override():
        with Session(engine) as session:
            yield session
    app.dependency_overrides = {}
    app.dependency_overrides[getattr(__import__(
        "main"), "get_session")] = get_session_override

    yield engine

    # Teardown: drop all tables and remove test db file
    SQLModel.metadata.drop_all(engine)
    if os.path.exists("./test_database.db"):
        os.remove("./test_database.db")


@pytest.fixture(autouse=True)
def setup_space(test_db):
    # Create a test space before each test
    with Session(test_db) as session:
        # Get all spaces and delete them
        for space in session.exec(select(Space)).all():
            session.delete(space)
        for space_event in session.exec(select(SpaceEvent)).all():
            session.delete(space_event)
        session.commit()
        space = Space(
            name="TestSpace",
            logo="http://example.org/logo.png",
            url="http://example.org",
            address="123 Test St",
            lat="45.0",
            lon="90.0",
            contact_email="test@example.org",
            basic_auth_password=hash_password("testpass")
        )
        session.add(space)
        session.commit()


client = TestClient(app)


def test_open_space_event_auth_fail():
    response = client.post(
        "/space_events/1/open",
        auth=("TestSpace", "wrongpass")
    )
    assert response.status_code == 403


def test_open_space():
    response = client.post(
        "/space_events/1/open",
        auth=("TestSpace", "testpass")
    )
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "open"


def test_close_space():
    response = client.post(
        "/space_events/1/close",
        auth=("TestSpace", "testpass")
    )
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "closed"


def test_space_api():
    response = client.get("/space/TestSpace/space.json")
    assert response.status_code == 200
    data = response.json()
    assert data["space"] == "TestSpace"


def test_space_api_schema():
    # Post an event to have some data
    response = client.post(
        "/space_events/1/open",
        auth=("TestSpace", "testpass")
    )
    response = client.get("/space/TestSpace/space.json")
    assert response.status_code == 200
    data = response.json()
    # Read the schema file
    with open("schema/15.json") as f:
        schema = f.read()
    schema_json = json.loads(schema)
    try:
        validate(instance=data, schema=schema_json)
    except Exception as e:
        pytest.fail(f"JSON schema validation failed: {e}")


def test_private_space_api_auth_fail():
    # Change space to private
    with Session(app.state._engine) as session:
        space = session.exec(select(Space).where(
            Space.name == "TestSpace")).first()
        space.is_private = True
        session.add(space)
        session.commit()
    response = client.get("/space/TestSpace/space.json")
    assert response.status_code == 401


def test_private_space_api_auth_success():
    # Change space to private
    with Session(app.state._engine) as session:
        space = session.exec(select(Space).where(
            Space.name == "TestSpace")).first()
        space.is_private = True
        session.add(space)
        session.commit()
    response = client.get(
        "/space/TestSpace/space.json",
        auth=("TestSpace", "testpass")
    )
    assert response.status_code == 200
