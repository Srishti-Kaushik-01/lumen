"""Flask entry point and persistent research workspace."""

import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from flask import Flask, abort, current_app, jsonify, redirect, render_template, request, session, url_for
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash

from model import HuggingFaceEncoder, OpenAlexSource, ResearchAssistant, configured_writer


def load_env_file(path=Path(__file__).with_name(".env")):
    """Read KEY=value lines from code/.env for local use; real environment variables take precedence."""
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() and not name.lstrip().startswith("#") and _:
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


load_env_file()

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
try:
    from psycopg_pool import ConnectionPool
except ImportError:
    ConnectionPool = None


class RateLimiter:
    """Counts recent requests per key in memory; the app runs as one worker process, so one counter is enough."""

    def __init__(self):
        self.events = {}
        self.lock = Lock()

    MAX_KEYS = 10_000

    def under(self, key, limit, seconds):
        """Whether the key has fewer than `limit` events in the last `seconds`, without recording one."""
        now = time.monotonic()
        with self.lock:
            return sum(now - moment < seconds for moment in self.events.get(key, ())) < limit

    def allow(self, key, limit, seconds):
        now = time.monotonic()
        with self.lock:
            recent = [moment for moment in self.events.pop(key, ()) if now - moment < seconds]
            allowed = len(recent) < limit
            if allowed:
                recent.append(now)
            # Re-inserting keeps the dictionary in order of last use, so the oldest keys are dropped first.
            self.events[key] = recent[-100:]
            while len(self.events) > self.MAX_KEYS:
                self.events.pop(next(iter(self.events)))
            return allowed


class ResearchRepository:
    SCHEMA = (
        """CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL, name TEXT)""",
        """CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
            updated_at TEXT NOT NULL)""",
        """CREATE INDEX IF NOT EXISTS conversations_owner ON conversations(owner_id, updated_at)""",
        """CREATE TABLE IF NOT EXISTS turns (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL CHECK (seq > 0), request_id TEXT NOT NULL,
            question TEXT NOT NULL, resolved TEXT NOT NULL, answer TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(conversation_id, seq), UNIQUE(conversation_id, request_id))""",
        """CREATE TABLE IF NOT EXISTS identities (
            provider TEXT NOT NULL, subject TEXT NOT NULL,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, PRIMARY KEY (provider, subject))""",
        """CREATE TABLE IF NOT EXISTS sources (url TEXT PRIMARY KEY)""",
        """CREATE TABLE IF NOT EXISTS citations (
            turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK (position > 0), source_url TEXT NOT NULL REFERENCES sources(url),
            metadata TEXT NOT NULL, PRIMARY KEY (turn_id, position))""",
    )

    def __init__(self, location):
        self.location = str(location)
        self.postgres = self.location.startswith(("postgres://", "postgresql://"))
        if self.postgres and psycopg is None:
            raise RuntimeError("PostgreSQL requires a working Psycopg driver and libpq.")
        self.schema = os.getenv("DATABASE_SCHEMA", "lumen")
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", self.schema):
            raise RuntimeError("DATABASE_SCHEMA must be a simple lower-case name.")
        self.ready = False
        self.setup_lock = Lock()
        self.pool = None
        if self.postgres and ConnectionPool is not None:
            # A small pool, one connection per web server thread, avoids a new TLS handshake for every query.
            # prepare_threshold=None keeps statements unprepared, as Neon's transaction pooler requires.
            # Neon closes idle connections, so each one is checked before use and retired after four idle minutes.
            self.pool = ConnectionPool(self.location, min_size=1, max_size=4, open=True, timeout=10, max_idle=240,
                                       check=ConnectionPool.check_connection,
                                       kwargs={"row_factory": dict_row, "connect_timeout": 5,
                                               "prepare_threshold": None})

    @contextmanager
    def postgres_connection(self):
        if self.pool is not None:
            with self.pool.connection() as connection:
                yield connection
        else:
            with psycopg.connect(self.location, row_factory=dict_row, connect_timeout=5) as connection:
                yield connection

    @contextmanager
    def connect(self):
        if self.postgres:
            with self.postgres_connection() as connection:
                # Lumen keeps its tables in their own schema so a shared database's other tables are untouched.
                # SET LOCAL lasts for this transaction only, which is safe behind Neon's connection pooler.
                if not self.ready:
                    connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
                connection.execute(f'SET LOCAL search_path TO "{self.schema}"')
                yield connection
            return
        connection = sqlite3.connect(self.location, timeout=5)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def execute(self, connection, statement, values=()):
        return connection.execute(statement.replace("?", "%s") if self.postgres else statement, values)

    @contextmanager
    def transaction(self):
        with self.setup_lock:
            if not self.ready:
                with self.connect() as connection:
                    if self.postgres:
                        connection.execute("SELECT pg_advisory_xact_lock(640641)")
                    else:
                        connection.execute("BEGIN IMMEDIATE")
                    for statement in self.SCHEMA:
                        connection.execute(statement)
                    self.upgrade(connection)
                self.ready = True
        with self.connect() as connection:
            yield connection

    def upgrade(self, connection):
        """Add columns introduced after the first release to databases created earlier."""
        if self.postgres:
            connection.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT")
            connection.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS session_version INTEGER NOT NULL DEFAULT 0")
            return
        columns = [row[1] for row in connection.execute("PRAGMA table_info(users)")]
        if "name" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN name TEXT")
        if "session_version" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN session_version INTEGER NOT NULL DEFAULT 0")

    def session_version(self, owner_id):
        """The account's sign-in generation; raising it signs out every other browser."""
        with self.transaction() as connection:
            row = self.execute(connection, "SELECT session_version FROM users WHERE id = ?", (owner_id,)).fetchone()
            return row["session_version"] if row else None

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def ping(self):
        with self.transaction() as connection:
            connection.execute("SELECT 1").fetchone()

    def create_user(self, email, password_hash, name=""):
        user_id = uuid4().hex
        with self.transaction() as connection:
            result = self.execute(connection, """INSERT INTO users (id, email, password_hash, created_at, name)
                                  VALUES (?, ?, ?, ?, ?) ON CONFLICT(email) DO NOTHING""",
                                  (user_id, email, password_hash, self.now(), name))
            return user_id if result.rowcount else None

    def user_by_email(self, email):
        with self.transaction() as connection:
            return self.execute(connection, """SELECT id, email, password_hash, session_version
                                FROM users WHERE email = ?""", (email,)).fetchone()

    def user_for_identity(self, provider, subject, email, name=""):
        """Return (id, email, session_version) for an external identity, creating or linking it on first sign-in.

        Email registration does not prove that the person owns the address, but Google has verified it.
        When Google is first linked to an account that has a password, that password is therefore removed and
        every other session is ended, so someone who registered the address in advance cannot keep access.
        """
        with self.transaction() as connection:
            # "!" is not a valid Werkzeug hash, so an account created here has no usable password.
            self.execute(connection, """INSERT INTO users (id, email, password_hash, created_at, name)
                         VALUES (?, ?, '!', ?, ?) ON CONFLICT(email) DO NOTHING""", (uuid4().hex, email, self.now(), name))
            user = self.execute(connection, "SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            created = self.execute(connection, """INSERT INTO identities VALUES (?, ?, ?)
                         ON CONFLICT(provider, subject) DO NOTHING""", (provider, subject, user["id"])).rowcount
            linked = self.execute(connection, """SELECT u.id, u.email FROM identities i JOIN users u ON u.id = i.user_id
                    WHERE i.provider = ? AND i.subject = ?""", (provider, subject)).fetchone()
            if created:
                self.execute(connection, """UPDATE users SET password_hash = '!', session_version = session_version + 1
                             WHERE id = ? AND password_hash <> '!'""", (linked["id"],))
            if name:
                self.execute(connection, "UPDATE users SET name = ? WHERE id = ? AND (name IS NULL OR name = '')",
                             (name, linked["id"]))
            version = self.execute(connection, "SELECT session_version FROM users WHERE id = ?",
                                   (linked["id"],)).fetchone()["session_version"]
            return linked["id"], linked["email"], version

    def account(self, owner_id):
        """Profile details and usage counts for the account page."""
        with self.transaction() as connection:
            user = self.execute(connection, "SELECT email, name, password_hash, created_at FROM users WHERE id = ?",
                                (owner_id,)).fetchone()
            if user is None:
                return None
            counts = self.execute(connection, """SELECT
                    (SELECT COUNT(*) FROM conversations WHERE owner_id = ?) AS conversations,
                    (SELECT COUNT(*) FROM turns t JOIN conversations c ON c.id = t.conversation_id
                     WHERE c.owner_id = ?) AS questions,
                    (SELECT COUNT(DISTINCT ci.source_url) FROM citations ci JOIN turns t ON t.id = ci.turn_id
                     JOIN conversations c ON c.id = t.conversation_id WHERE c.owner_id = ?) AS sources""",
                    (owner_id, owner_id, owner_id)).fetchone()
            providers = [row["provider"] for row in self.execute(
                connection, "SELECT provider FROM identities WHERE user_id = ? ORDER BY provider", (owner_id,))]
        return {"email": user["email"], "name": user["name"] or "", "created_at": user["created_at"], "has_password": user["password_hash"] != "!",
                "providers": providers, "conversations": counts["conversations"], "questions": counts["questions"],
                "sources": counts["sources"]}

    def set_name(self, owner_id, name):
        with self.transaction() as connection:
            return self.execute(connection, "UPDATE users SET name = ? WHERE id = ?", (name, owner_id)).rowcount > 0

    def password_hash(self, owner_id):
        with self.transaction() as connection:
            row = self.execute(connection, "SELECT password_hash FROM users WHERE id = ?", (owner_id,)).fetchone()
            return row["password_hash"] if row else None

    def set_password(self, owner_id, password_hash):
        """Store a new password and end every other session; returns the new session version."""
        with self.transaction() as connection:
            self.execute(connection, """UPDATE users SET password_hash = ?, session_version = session_version + 1
                         WHERE id = ?""", (password_hash, owner_id))
            row = self.execute(connection, "SELECT session_version FROM users WHERE id = ?", (owner_id,)).fetchone()
            return row["session_version"] if row else None

    def end_sessions(self, owner_id):
        with self.transaction() as connection:
            self.execute(connection, "UPDATE users SET session_version = session_version + 1 WHERE id = ?", (owner_id,))

    def delete_user(self, owner_id):
        """Delete an account; foreign keys cascade to identities, conversations, turns, and citations."""
        with self.transaction() as connection:
            return self.execute(connection, "DELETE FROM users WHERE id = ?", (owner_id,)).rowcount > 0

    def rename_conversation(self, owner_id, conversation_id, title):
        with self.transaction() as connection:
            return self.execute(connection, "UPDATE conversations SET title = ? WHERE id = ? AND owner_id = ?",
                                (title, conversation_id, owner_id)).rowcount > 0

    def conversations(self, owner_id):
        with self.transaction() as connection:
            return [dict(row) for row in self.execute(connection, """SELECT id, title, updated_at
                    FROM conversations WHERE owner_id = ? ORDER BY updated_at DESC, id LIMIT 100""",
                    (owner_id,))]

    def create_conversation(self, owner_id):
        conversation = {"id": str(uuid4()), "title": "Untitled conversation", "updated_at": self.now()}
        with self.transaction() as connection:
            self.execute(connection, """INSERT INTO conversations (id, owner_id, title, updated_at)
                         VALUES (?, ?, ?, ?)""",
                         (conversation["id"], owner_id, conversation["title"], conversation["updated_at"]))
        return conversation

    def delete_conversation(self, owner_id, conversation_id):
        with self.transaction() as connection:
            return self.execute(connection, "DELETE FROM conversations WHERE id = ? AND owner_id = ?",
                                (conversation_id, owner_id)).rowcount > 0

    def get_conversation(self, owner_id, conversation_id, limit=None):
        with self.transaction() as connection:
            row = self.execute(connection, "SELECT * FROM conversations WHERE id = ? AND owner_id = ?",
                               (conversation_id, owner_id)).fetchone()
            if row is None:
                return None
            conversation = dict(row)
            records = self.execute(connection, """SELECT id, seq, question, resolved, answer, created_at
                    FROM turns WHERE conversation_id = ? AND seq <= ? ORDER BY seq DESC LIMIT ?""",
                    (conversation_id, row["revision"], limit or 100)).fetchall()
            citations = self.execute(connection, """SELECT c.turn_id, c.metadata FROM citations c
                    JOIN turns t ON t.id = c.turn_id WHERE t.conversation_id = ? AND t.seq BETWEEN ? AND ?
                    ORDER BY t.seq, c.position""",
                    (conversation_id, records[-1]["seq"] if records else 0, row["revision"])).fetchall()
        by_turn = {}
        for item in citations:
            by_turn.setdefault(item["turn_id"], []).append(json.loads(item["metadata"]))
        conversation["turns"] = []
        for record in reversed(records):
            turn = dict(record)
            turn["answer"] = json.loads(turn["answer"])
            turn["answer"]["sources"] = by_turn.get(turn["id"], [])
            conversation["turns"].append(turn)
        conversation.pop("owner_id")
        return conversation

    def has_request(self, owner_id, conversation_id, request_id):
        with self.transaction() as connection:
            return self.execute(connection, """SELECT 1 FROM turns t JOIN conversations c ON c.id = t.conversation_id
                    WHERE c.id = ? AND c.owner_id = ? AND t.request_id = ?""",
                    (conversation_id, owner_id, request_id)).fetchone() is not None

    def save_turn(self, owner_id, conversation, request_id, question, resolved, answer):
        conversation_id = conversation["id"]
        revision = conversation["revision"]
        title = question[:72] if revision == 0 else conversation["title"]
        turn_id, timestamp = uuid4().hex, self.now()
        with self.transaction() as connection:
            update = self.execute(connection, """UPDATE conversations SET revision = revision + 1, title = ?,
                    updated_at = ? WHERE id = ? AND owner_id = ? AND revision = ?""",
                    (title, timestamp, conversation_id, owner_id, revision))
            if not update.rowcount:
                return False
            content = {key: value for key, value in answer.items() if key != "sources"}
            self.execute(connection, "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (turn_id, conversation_id, revision + 1, request_id,
                          question, resolved, json.dumps(content), timestamp))
            for position, source in enumerate(answer["sources"], start=1):
                self.execute(connection, "INSERT INTO sources VALUES (?) ON CONFLICT(url) DO NOTHING", (source["url"],))
                self.execute(connection, "INSERT INTO citations VALUES (?, ?, ?, ?)",
                             (turn_id, position, source["url"], json.dumps(source)))
        return True


class GoogleIdentity:
    """Google sign-in with OpenID Connect: authorization code flow with PKCE, state, and nonce."""

    AUTHORIZE = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN = "https://oauth2.googleapis.com/token"
    ISSUERS = {"https://accounts.google.com", "accounts.google.com"}

    def __init__(self, client_id, client_secret, timeout=10):
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout

    @staticmethod
    def challenge(verifier):
        return urlsafe_b64encode(sha256(verifier.encode()).digest()).rstrip(b"=").decode()

    def authorization_url(self, redirect_uri, state, nonce, verifier):
        return self.AUTHORIZE + "?" + urlencode({
            "client_id": self.client_id, "redirect_uri": redirect_uri, "response_type": "code",
            "scope": "openid email profile", "state": state, "nonce": nonce, "prompt": "select_account",
            "code_challenge": self.challenge(verifier), "code_challenge_method": "S256",
        })

    def exchange(self, code, redirect_uri, verifier):
        body = urlencode({
            "code": code, "client_id": self.client_id, "client_secret": self.client_secret,
            "redirect_uri": redirect_uri, "grant_type": "authorization_code", "code_verifier": verifier,
        }).encode()
        token_request = Request(self.TOKEN, data=body, headers={
            "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
        with urlopen(token_request, timeout=self.timeout) as response:
            return json.loads(response.read(200_000))["id_token"]

    def claims(self, id_token, nonce):
        """Check the ID token and return (subject, email, name).

        The token comes directly from Google's token endpoint over verified HTTPS, so OpenID Connect
        Core 1.0 section 3.1.3.7 allows TLS server validation in place of checking the signature.
        """
        payload = id_token.split(".")[1]
        data = json.loads(urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        audience = data.get("aud")
        audiences = audience if isinstance(audience, list) else [audience]
        expires = data.get("exp")
        if (data.get("iss") not in self.ISSUERS or self.client_id not in audiences
                or not isinstance(nonce, str) or not hmac.compare_digest(str(data.get("nonce", "")), nonce)
                or not isinstance(expires, (int, float)) or expires < time.time()
                or data.get("email_verified") is not True
                or not isinstance(data.get("sub"), str) or not isinstance(data.get("email"), str)):
            raise ValueError("The Google sign-in response could not be verified.")
        name = data.get("name") if isinstance(data.get("name"), str) else ""
        return data["sub"], data["email"].strip().casefold(), name.strip()[:80]


class WorkspaceAPI:
    GOOGLE_FAILED = "/app#google-sign-in-failed"
    EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")

    def __init__(self, repository, assistant, google=None):
        self.repository = repository
        self.assistant = assistant
        self.google = google
        self.dummy_hash = generate_password_hash(secrets.token_hex(16))
        self.limiter = RateLimiter()

    def limit(self, name, key, count, seconds):
        if not current_app.config.get("RATE_LIMITS", True):
            return
        if not self.limiter.allow(f"{name}:{key}", count, seconds):
            abort(429, "Too many requests. Wait a minute and try again.")

    @staticmethod
    def start_session(user_id, email, version):
        session.clear()
        session.permanent = True
        session.update(user_id=user_id, email=email, version=version)

    def register(self, app):
        app.add_url_rule("/", view_func=self.landing)
        app.add_url_rule("/app", view_func=self.index)
        app.add_url_rule("/healthz", view_func=self.health)
        app.add_url_rule("/api/session", view_func=self.session_state)
        app.add_url_rule("/auth/google", view_func=self.google_start)
        app.add_url_rule("/auth/google/callback", view_func=self.google_callback)
        app.add_url_rule("/api/auth/<action>", view_func=self.authenticate, methods=["POST"])
        app.add_url_rule("/api/conversations", view_func=self.conversations, methods=["GET", "POST"])
        app.add_url_rule("/api/conversations/<uuid:identifier>", view_func=self.conversation,
                         methods=["GET", "PATCH", "DELETE"])
        app.add_url_rule("/api/account", view_func=self.account, methods=["GET", "PATCH", "DELETE"])
        app.add_url_rule("/api/account/password", view_func=self.change_password, methods=["POST"])
        app.add_url_rule("/api/conversations/<uuid:identifier>/turns", view_func=self.turn, methods=["POST"])
        app.before_request(self.protect_mutations)
        app.after_request(self.secure_response)
        app.register_error_handler(HTTPException, self.http_error)
        app.register_error_handler(sqlite3.Error, self.storage_error)
        if psycopg is not None:
            app.register_error_handler(psycopg.Error, self.storage_error)

    @staticmethod
    def protect_mutations():
        if request.path.startswith("/api/") and request.method in {"POST", "PATCH", "DELETE"}:
            supplied = request.headers.get("X-CSRF-Token", "")
            expected = session.get("csrf")
            if not expected or not supplied.isascii() or not hmac.compare_digest(supplied, expected):
                abort(403, "Session check failed. Refresh the page and try again.")

    @staticmethod
    def secure_response(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'nonce-" + getattr(request, "csp_nonce", "") +
            "'; style-src 'self' 'nonce-" + getattr(request, "csp_nonce", "") +
            "'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        if current_app.config["SESSION_COOKIE_SECURE"]:
            # Production is served over HTTPS; keep browsers from sending credentials over plain HTTP.
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    @staticmethod
    def http_error(error):
        return jsonify(error=error.description), error.code

    @staticmethod
    def storage_error(_error):
        return jsonify(error="Saved conversations are temporarily unavailable. Try again shortly."), 503

    @staticmethod
    def payload():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, "Send a valid JSON object.")
        return data

    def owner(self):
        if not session.get("user_id"):
            abort(401, "Sign in to access this workspace.")
        # A password change, a new Google link, or account deletion raises the version and ends this session.
        if self.repository.session_version(session["user_id"]) != session.get("version", 0):
            session.clear()
            abort(401, "Your session has ended. Sign in again.")
        return session["user_id"]

    @staticmethod
    def page(template):
        request.csp_nonce = secrets.token_urlsafe(18)
        return render_template(template, nonce=request.csp_nonce, signed_in=bool(session.get("user_id")))

    def landing(self):
        return self.page("landing.html")

    def index(self):
        return self.page("index.html")

    def health(self):
        self.repository.ping()
        return jsonify(status="ok")

    def session_state(self):
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return jsonify(email=session.get("email"), csrf=session["csrf"], google=self.google is not None)

    def google_start(self):
        if self.google is None:
            abort(404)
        pending = {"state": secrets.token_urlsafe(32), "nonce": secrets.token_urlsafe(32),
                   "verifier": secrets.token_urlsafe(48)}
        session["google"] = pending
        return redirect(self.google.authorization_url(url_for("google_callback", _external=True),
                                                      pending["state"], pending["nonce"], pending["verifier"]))

    def google_callback(self):
        if self.google is None:
            abort(404)
        pending = session.pop("google", None)
        state, code = request.args.get("state", ""), request.args.get("code", "")
        if not pending or not code or not state.isascii() or not hmac.compare_digest(state, pending["state"]):
            return redirect(self.GOOGLE_FAILED)
        try:
            token = self.google.exchange(code, url_for("google_callback", _external=True), pending["verifier"])
            subject, email, name = self.google.claims(token, pending["nonce"])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            current_app.logger.exception("Google sign-in failed")
            return redirect(self.GOOGLE_FAILED)
        user_id, email, version = self.repository.user_for_identity("google", subject, email, name)
        self.start_session(user_id, email, version)
        return redirect("/app")

    def authenticate(self, action):
        if action == "logout":
            # Signing out ends every session of the account, so a copied cookie stops working too.
            if session.get("user_id"):
                self.repository.end_sessions(session["user_id"])
            session.clear()
            return self.session_state()
        if action not in {"register", "login"}:
            abort(404)
        self.limit("auth-ip", request.remote_addr, 20, 60)
        data = self.payload()
        email, password = data.get("email"), data.get("password")
        if not isinstance(email, str) or not isinstance(password, str):
            abort(400, "Enter a valid email address and password.")
        email = email.strip().casefold()
        if not self.EMAIL.fullmatch(email) or len(email) > 254 or not 8 <= len(password) <= 128:
            abort(400, "Enter a valid email address and a password of 8–128 characters.")
        # Failed guesses at one account from one network address are slowed down. Only failures count,
        # and the key includes the address, so nobody can lock the owner out by guessing from elsewhere.
        failures = f"{email}|{request.remote_addr}"
        if not self.limiter.under(f"auth-fail:{failures}", 10, 300) and current_app.config.get("RATE_LIMITS", True):
            abort(429, "Too many failed attempts. Wait a few minutes and try again.")
        version = 0
        if action == "register":
            name = data.get("name", "")
            if not isinstance(name, str) or len(name.strip()) > 80:
                abort(400, "Enter a name of up to 80 characters.")
            user_id = self.repository.create_user(email, generate_password_hash(password), name.strip())
            if not user_id:
                abort(409, "An account with that email already exists. Try signing in.")
        else:
            user = self.repository.user_by_email(email)
            matched = check_password_hash(user["password_hash"] if user else self.dummy_hash, password)
            if not matched or user is None:
                self.limiter.allow(f"auth-fail:{failures}", 10**6, 300)  # record the failure
                abort(401, "The email or password is incorrect.")
            user_id, version = user["id"], user["session_version"]
        self.start_session(user_id, email, version)
        return self.session_state()

    def account(self):
        owner = self.owner()
        if request.method == "DELETE":
            if self.payload().get("confirm") != session.get("email"):
                abort(400, "Type your email address exactly to confirm.")
            self.repository.delete_user(owner)
            session.clear()
            return self.session_state()
        if request.method == "PATCH":
            name = self.payload().get("name")
            if not isinstance(name, str) or len(name.strip()) > 80:
                abort(400, "Enter a name of up to 80 characters.")
            self.repository.set_name(owner, name.strip())
        details = self.repository.account(owner)
        if details is None:
            session.clear()
            abort(401, "Sign in to access this workspace.")
        return jsonify(details)

    def change_password(self):
        owner = self.owner()
        data = self.payload()
        current, password = data.get("current", ""), data.get("password")
        if not isinstance(password, str) or not 8 <= len(password) <= 128:
            abort(400, "Choose a new password of 8–128 characters.")
        stored = self.repository.password_hash(owner)
        # Accounts created through Google have no password yet and may set one without a current password.
        if stored is None or (stored != "!" and not (isinstance(current, str) and check_password_hash(stored, current))):
            abort(400, "The current password is incorrect.")
        # Other browsers signed in to this account are signed out; this one stays signed in.
        session["version"] = self.repository.set_password(owner, generate_password_hash(password))
        return jsonify(status="updated")

    def conversations(self):
        owner = self.owner()
        if request.method == "POST":
            self.limit("conversation", owner, 30, 3600)
            return jsonify(self.repository.create_conversation(owner)), 201
        return jsonify(self.repository.conversations(owner))

    def conversation(self, identifier):
        owner = self.owner()
        conversation_id = str(identifier)
        if request.method == "DELETE":
            if not self.repository.delete_conversation(owner, conversation_id):
                abort(404, "Conversation not found.")
            return "", 204
        if request.method == "PATCH":
            title = self.payload().get("title")
            if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120:
                abort(400, "Enter a title of 1–120 characters.")
            if not self.repository.rename_conversation(owner, conversation_id, title.strip()):
                abort(404, "Conversation not found.")
            return jsonify(id=conversation_id, title=title.strip())
        result = self.repository.get_conversation(owner, conversation_id)
        if result is None:
            abort(404, "Conversation not found.")
        return jsonify(result)

    def turn(self, identifier):
        owner = self.owner()
        conversation_id = str(identifier)
        data = self.payload()
        question = data.get("question")
        if not isinstance(question, str) or not 3 <= len(question.strip()) <= 700:
            abort(400, "Enter a research question between 3 and 700 characters.")
        try:
            request_id = UUID(str(data.get("request_id"))).hex
        except ValueError:
            abort(400, "Use a valid submission identifier.")
        question = question.strip()
        conversation = self.repository.get_conversation(owner, conversation_id, limit=8)
        if conversation is None:
            abort(404, "Conversation not found.")
        if not self.repository.has_request(owner, conversation_id, request_id):
            if conversation["revision"] >= 100:
                abort(409, "This conversation has reached 100 questions. Start a new conversation to continue.")
            # Each answer uses the shared OpenAlex and language model allowances, so one account is capped.
            self.limit("turn", owner, 40, 3600)
            try:
                resolved, answer = self.assistant.answer(question, conversation["turns"])
            except Exception:
                current_app.logger.exception("Research inference failed")
                abort(503, "Research inference is temporarily unavailable. Try again shortly.")
            saved = self.repository.save_turn(owner, conversation, request_id, question, resolved, answer)
            if not saved and not self.repository.has_request(owner, conversation_id, request_id):
                abort(409, "The conversation changed during this search. Reload it and try again.")
        result = self.repository.get_conversation(owner, conversation_id)
        if result is None:
            abort(404, "Conversation not found.")
        return jsonify(result)


def preload(encoder):
    try:
        encoder.load()
    except Exception:  # The first research request will try again and report the failure to the user.
        logging.getLogger(__name__).exception("Could not preload the ranking model")


def create_app(config=None, repository=None, assistant=None, google=None):
    app = Flask(__name__)
    production = os.getenv("APP_ENV") == "production"
    if production:
        # The host's HTTPS proxy terminates TLS; trust its headers so OAuth redirect URLs use https.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY") or (None if production else secrets.token_hex(32)),
        DATABASE_URL=os.getenv("DATABASE_URL") or (None if production else
                     str(Path.home() / ".ignou-research.sqlite3")),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=production,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=14 * 24 * 3600,
        MAX_CONTENT_LENGTH=12_000,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if config:
        app.config.update(config)
    if not app.config["SECRET_KEY"] or not app.config["DATABASE_URL"]:
        raise RuntimeError("Configure SECRET_KEY and DATABASE_URL before starting production.")
    if production and not app.config["DATABASE_URL"].startswith(("postgres://", "postgresql://")):
        raise RuntimeError("Production requires PostgreSQL.")
    repository = repository or ResearchRepository(app.config["DATABASE_URL"])
    if assistant is None:
        external = None if os.getenv("LOCAL_ONLY") == "1" else OpenAlexSource(os.getenv("OPENALEX_API_KEY"))
        encoder = HuggingFaceEncoder()
        assistant = ResearchAssistant(encoder, external, configured_writer())
        # Load the ranking model while the server starts, so the first visitor does not wait for it.
        Thread(target=preload, args=(encoder,), daemon=True).start()
    if google is None and os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"):
        google = GoogleIdentity(os.getenv("GOOGLE_CLIENT_ID"), os.getenv("GOOGLE_CLIENT_SECRET"))
    WorkspaceAPI(repository, assistant, google).register(app)
    return app


app = create_app()

if __name__ == "__main__":
    app.run()
