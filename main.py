from typing import Annotated, Optional
from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlmodel import Field, Session, SQLModel, create_engine, select, DateTime
from enum import Enum
from sqlalchemy.orm import reconstructor
from sqlalchemy import Column, event
from datetime import datetime, timezone
from argon2 import PasswordHasher
import requests
import logging
from contextlib import asynccontextmanager
import asyncio
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read keealive interval from env
KEEPALIVE_INTERVAL = int(
    os.getenv("KEEPALIVE_INTERVAL", "1300"))  # default 1300 seconds


class SpaceEventState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class SpacePublic(SQLModel):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    logo: str = Field(default=None, nullable=False)
    url: str = Field(default=None, nullable=False)
    address: str = Field(default=None, nullable=True)
    lat: float = Field(default=None, nullable=True)
    lon: float = Field(default=None, nullable=True)
    contact_email: str = Field(default=None, nullable=False)
    # last_keepalive: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_keepalive: datetime = Field(sa_column=Column(
        DateTime(timezone=True)), default_factory=lambda: datetime.now(timezone.utc))


class Space(SpacePublic, table=True):
    basic_auth_password: str = Field()
    is_private: bool = Field(default=False, nullable=False)
    telegram_bot_token: str = Field(default=None, nullable=True)
    telegram_channel_id: str = Field(default=None, nullable=True)
    telegram_enabled: bool = Field(default=False, nullable=False)


@event.listens_for(Space, "load")
def receive_load(space, context):
    if space.last_keepalive and space.last_keepalive.tzinfo is None:
        space.last_keepalive = space.last_keepalive.replace(
            tzinfo=timezone.utc)


class SpaceEventPublic(SQLModel):
    id: int | None = Field(default=None, primary_key=True)
    space_id: int = Field(foreign_key="space.id")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), index=True)
    state: SpaceEventState = Field(
        sa_column_kwargs={"default": SpaceEventState.UNKNOWN})


class SpaceEvent(SpaceEventPublic, table=True):
    telegram_message_id: int | None = Field(default=None, nullable=True)


def hash_password(password: str) -> str:
    """Hash password using argon2id"""
    return PasswordHasher().hash(password)


def verify_password(hashed_password: str, password: str) -> bool:
    """Verify password using argon2id"""
    try:
        return PasswordHasher().verify(hashed_password, password)
    except:
        return False


def authenticate(credentials: HTTPBasicCredentials, session: Session, space: Space) -> bool:
    """Authenticate user using basic auth"""
    if not space:
        return False
    if not verify_password(space.basic_auth_password, credentials.password):
        return False
    if credentials.username != space.name:
        return False
    return True


def send_telegram_message(space, space_event, session):
    """Send Telegram message about space event"""
    if not space.telegram_enabled or not space.telegram_bot_token or not space.telegram_channel_id:
        return
    message = f"'{space.name}' door is {space_event.state.value}."
    url = f"https://api.telegram.org/bot{space.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": space.telegram_channel_id,
        "text": message
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        # Save the message ID to the event
        resp_json = response.json()
        if resp_json.get("ok"):
            message_id = resp_json["result"]["message_id"]
            space_event.telegram_message_id = message_id
            session.add(space_event)
            session.commit()
        logger.info(
            f"Telegram message sent successfully for space '{space.name}' for event '{space_event.state.value}'.")
    except requests.RequestException as e:
        logger.error(f"Failed to send Telegram message: {e}")


def delete_telegram_message(space, session):
    """Delete previous Telegram message about space event"""
    if not space.telegram_enabled or not space.telegram_bot_token or not space.telegram_channel_id:
        return
    # Get the latest event with telegram_message_id
    latest_event = session.exec(
        select(SpaceEvent)
        .where(SpaceEvent.space_id == space.id, SpaceEvent.telegram_message_id != None)
        .order_by(SpaceEvent.timestamp.desc())
    ).first()
    if not latest_event:
        return
    url = f"https://api.telegram.org/bot{space.telegram_bot_token}/deleteMessage"
    payload = {
        "chat_id": space.telegram_channel_id,
        "message_id": latest_event.telegram_message_id
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        logger.info(
            f"Telegram message deleted successfully for space '{space.name}'.")
    except requests.RequestException as e:
        logger.error(f"Failed to delete Telegram message: {e}")


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    task = None
    # Create a default space for testing
    with Session(engine) as session:
        space = session.exec(select(Space).where(Space.id == 1)).first()
        if not space:
            hashed_password = hash_password("dummy_password")
            default_space = Space(
                id=1,
                name="ModeemiDummySpace",
                basic_auth_password=hashed_password,
                logo="https://trey.fi/media/modeemi-logo-ttyy-1.png",
                url="https://modeemi.fi",
                address="Tietotalo, huone TA013, Korkeakoulunkatu 1, FIN-33720 Tampere, Finland",
                lat=61.449940,
                lon=23.857036,
                contact_email="modeemi@example.org"
            )
            session.add(default_space)
            session.commit()
            # Add an initial unknown event
            initial_event = SpaceEvent(
                space_id=1, state=SpaceEventState.UNKNOWN)
            session.add(initial_event)
            session.commit()
        task = asyncio.create_task(scheduled_task(session))
    yield
    # Cleanup: cancel task on shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def scheduled_task(session):
    while True:
        await asyncio.sleep(KEEPALIVE_INTERVAL)
        await check_keepalives(session)


async def check_keepalives(session):
    spaces = session.exec(select(Space)).all()
    logger.info(f"Stage 0. Keepalive checking.")
    for space in spaces:
        latest_event = session.exec(
            select(SpaceEvent).where(SpaceEvent.space_id ==
                                     space.id).order_by(SpaceEvent.timestamp.desc())
        ).first()
        logger.info(
            f"Stage 1. Keepalive checking for space '{space.name}' '{latest_event.state}'.")
        if latest_event.state != SpaceEventState.UNKNOWN:
            logger.info(
                f"Stage 2. Keepalive checking for space '{space.name}' .")
            aware_keepalive = space.last_keepalive.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - aware_keepalive
            if delta.total_seconds() > KEEPALIVE_INTERVAL:
                logger.warning(
                    f"Space '{space.name}' has not sent keepalive for {delta.total_seconds()/60:.1f} minutes.")
                unknown_event = SpaceEvent(
                    space_id=space.id, state=SpaceEventState.UNKNOWN)
                session.add(unknown_event)
                session.commit()
                logger.info(
                    f"Space '{space.name}' state set to UNKNOWN due to missing keepalive.")
                delete_telegram_message(space, session)
                send_telegram_message(space, unknown_event, session)
    logger.info(f"Stage 5. Keepalive check ended.")


SessionDep = Annotated[Session, Depends(get_session)]


app = FastAPI(lifespan=lifespan,
              docs_url=None,     # disables Swagger UI
              redoc_url=None,    # disables ReDoc
              openapi_url=None)


security = HTTPBasic(auto_error=False)

app.mount("/static", StaticFiles(directory="static"), name="static")
# app.mount("/site", StaticFiles(directory="site", html = True), name="site")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def main_page():
    return RedirectResponse(url="/status")


@app.get("/status")
def status(request: Request, session: SessionDep):
    spaces_from_db = session.exec(select(Space)).all()
    spaces_dict = {}
    spaces_counter = 1
    for space_idx in spaces_from_db:
        latest_event = session.exec(select(SpaceEvent).where(
            SpaceEvent.space_id == space_idx.id).order_by(SpaceEvent.timestamp.desc())).first()
        if latest_event.state == SpaceEventState.OPEN:
            current_state = "open"
        elif latest_event.state == SpaceEventState.CLOSED:
            current_state = "closed"
        else:
            current_state = "unknown"
        spaces_dict[spaces_counter] = {
            "id": space_idx.id, "name": space_idx.name, "state": current_state}
        spaces_counter = spaces_counter + 1
    return templates.TemplateResponse(
        request=request, name="index.html", context={"id": id, "spaces": spaces_dict}
    )


@app.get("/tech")
def tech(request: Request):
    return templates.TemplateResponse(request=request, name="tech.html")


@app.post("/space_events/{space_id}/open", response_model=SpaceEventPublic)
async def open_space(space_id: int, session: SessionDep, credentials: Annotated[HTTPBasicCredentials, Depends(security)], background_tasks: BackgroundTasks) -> SpaceEvent:
    space = session.get(Space, space_id)
    if not authenticate(credentials, session, space):
        raise HTTPException(
            status_code=403, detail="Forbidden")
    event = SpaceEvent(space_id=space_id, state=SpaceEventState.OPEN)
    session.add(event)
    session.commit()
    session.refresh(event)
    logger.info(f"Space '{space.name}' opened.")
    delete_telegram_message(space, session)
    background_tasks.add_task(send_telegram_message, space, event, session)
    return event


@app.post("/space_events/{space_id}/close", response_model=SpaceEventPublic)
async def close_space(space_id: int, session: SessionDep, credentials: Annotated[HTTPBasicCredentials, Depends(security)], background_tasks: BackgroundTasks) -> SpaceEvent:
    space = session.get(Space, space_id)
    if not authenticate(credentials, session, space):
        raise HTTPException(
            status_code=403, detail="Forbidden")
    event = SpaceEvent(space_id=space_id, state=SpaceEventState.CLOSED)
    session.add(event)
    session.commit()
    session.refresh(event)
    logger.info(f"Space '{space.name}' closed.")
    delete_telegram_message(space, session)
    background_tasks.add_task(send_telegram_message, space, event, session)
    return event


@app.post("/space_events/{space_id}/keepalive/open")
def keepalive_space_open(space_id: int, session: SessionDep, credentials: Annotated[HTTPBasicCredentials, Depends(security)], background_tasks: BackgroundTasks):
    space = session.get(Space, space_id)
    if not credentials or not authenticate(credentials, session, space):
        raise HTTPException(
            status_code=403, detail="Forbidden")
    space.last_keepalive = datetime.now(timezone.utc)
    session.add(space)
    session.commit()

    latest_event = session.exec(
        select(SpaceEvent).where(SpaceEvent.space_id ==
                                 space.id).order_by(SpaceEvent.timestamp.desc())
    ).first()
    if latest_event.state != SpaceEventState.OPEN:
        event = SpaceEvent(space_id=space_id, state=SpaceEventState.OPEN)
        session.add(event)
        session.commit()
        session.refresh(event)
        delete_telegram_message(space, session)
        background_tasks.add_task(send_telegram_message, space, event, session)
    logger.info(f"Received keepalive from space '{space.name}'. State open.")
    return {"message": "Keepalive received"}


@app.post("/space_events/{space_id}/keepalive/close")
def keepalive_space_close(space_id: int, session: SessionDep, credentials: Annotated[HTTPBasicCredentials, Depends(security)], background_tasks: BackgroundTasks):
    space = session.get(Space, space_id)
    if not credentials or not authenticate(credentials, session, space):
        raise HTTPException(
            status_code=403, detail="Forbidden")
    space.last_keepalive = datetime.now(timezone.utc)
    session.add(space)
    session.commit()
    latest_event = session.exec(
        select(SpaceEvent).where(SpaceEvent.space_id ==
                                 space.id).order_by(SpaceEvent.timestamp.desc())
    ).first()
    if latest_event.state != SpaceEventState.CLOSED:
        event = SpaceEvent(space_id=space_id, state=SpaceEventState.CLOSED)
        session.add(event)
        session.commit()
        session.refresh(event)
        delete_telegram_message(space, session)
        background_tasks.add_task(send_telegram_message, space, event, session)
    logger.info(f"Received keepalive from space '{space.name}'. State closed.")
    return {"message": "Keepalive received"}

# SpaceAPI response


@app.get("/space/{space_name}/space.json")
def space_api(space_name: str, session: SessionDep, credentials: Optional[HTTPBasicCredentials] = Depends(security)):
    space = session.exec(select(Space).where(Space.name == space_name)).first()
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    if space.is_private:
        if not credentials or not authenticate(credentials, session, space):
            raise HTTPException(
                status_code=401, detail="Unauthorized")
    latest_event = session.exec(
        select(SpaceEvent).where(SpaceEvent.space_id ==
                                 space.id).order_by(SpaceEvent.timestamp.desc())
    ).first()
    state = latest_event.state if latest_event else SpaceEventState.UNKNOWN
    space_json = {
        "api_compatibility": ["15"],
        "space": space.name,
        "logo": space.logo,
        "url": space.url,
        "state": {
            "open": state == SpaceEventState.OPEN,
        },
        "contact": {
            "email": space.contact_email
        }
    }
    # Add location if available
    if space.address or (space.lat is not None and space.lon is not None):
        space_json["location"] = {}
        if space.address:
            space_json["location"]["address"] = space.address
        if space.lat is not None and space.lon is not None:
            space_json["location"]["lat"] = space.lat
            space_json["location"]["lon"] = space.lon
    if latest_event is not None:
        space_json["state"]["lastchange"] = int(
            latest_event.timestamp.timestamp())
    return space_json
