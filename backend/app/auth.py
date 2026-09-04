"""
Authentification JWT — hachage de mots de passe, création/vérification de tokens.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db

settings = get_settings()

# ── Import de jose / passlib (déclaré dans requirements.txt) ─────
# Si absents, l'authentification doit échouer avec un message explicite
# plutôt qu'un AttributeError confus au moment de l'appel.
try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    _HAS_AUTH_DEPS = True
except ImportError as _e:  # pragma: no cover
    _HAS_AUTH_DEPS = False
    _AUTH_DEPS_ERR = _e


def _auth_unavailable():
    raise RuntimeError(
        "Dépendances d'authentification manquantes. Installez "
        "`python-jose[cryptography]` et `passlib[bcrypt]` (backend/requirements.txt)."
    )


# ── Hachage des mots de passe ─────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") if _HAS_AUTH_DEPS else None

# ── Schéma OAuth2 ─────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


def hash_password(password: str) -> str:
    if pwd_context is None:
        _auth_unavailable()
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    if pwd_context is None:
        _auth_unavailable()
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    if jwt is None:
        _auth_unavailable()
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    """Décode un JWT et retourne le payload. Lève JWTError si invalide."""
    if jwt is None:
        _auth_unavailable()
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])


# ── FastAPI Dependencies ──────────────────────────────────────────

def _get_user_from_token(token: str, db: Session):
    """Extrait l'utilisateur à partir du token JWT."""
    from app.models.user import User  # import retardé pour éviter circularité

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants invalides",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    """Dependency : exige un token valide."""
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token requis",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _get_user_from_token(token, db)


async def get_current_user_optional(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    """Dependency : token optionnel (retourne None si absent/invalide)."""
    if token is None:
        return None
    try:
        return _get_user_from_token(token, db)
    except HTTPException:
        return None
