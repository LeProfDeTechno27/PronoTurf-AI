"""
Pydantic schemas pour validation et sérialisation
"""

# User schemas
from .user import UserCreate, UserRead, UserUpdate, Token

# Hippodrome schemas
from .hippodrome import (
    HippodromeCreate,
    HippodromeUpdate,
    HippodromeResponse,
    HippodromeSimple,
    HippodromeList,
)

# Reunion schemas
from .reunion import (
    ReunionCreate,
    ReunionUpdate,
    ReunionResponse,
    ReunionSimple,
    ReunionWithHippodrome,
    ReunionList,
    ReunionDetailResponse,
)

# Course schemas
from .course import (
    CourseCreate,
    CourseUpdate,
    CourseResponse,
    CourseSimple,
    CourseWithReunion,
    CourseList,
    CourseDetailResponse,
    CourseFilter,
)

# Horse schemas
from .horse import (
    HorseCreate,
    HorseUpdate,
    HorseResponse,
    HorseSimple,
    HorseList,
    HorseDetailResponse,
)

# Jockey schemas
from .jockey import (
    JockeyCreate,
    JockeyUpdate,
    JockeyResponse,
    JockeySimple,
    JockeyList,
    JockeyDetailResponse,
)

# Trainer schemas
from .trainer import (
    TrainerCreate,
    TrainerUpdate,
    TrainerResponse,
    TrainerSimple,
    TrainerList,
    TrainerDetailResponse,
)

# Partant schemas
from .partant import (
    PartantCreate,
    PartantUpdate,
    PartantResponse,
    PartantSimple,
    PartantWithRelations,
    PartantList,
    PartantDetailResponse,
    PartantBatchCreate,
    PartantResultUpdate,
    PartantOddsUpdate,
)

__all__ = [
    # User
    "UserCreate",
    "UserRead",
    "UserUpdate",
    "Token",
    # Hippodrome
    "HippodromeCreate",
    "HippodromeUpdate",
    "HippodromeResponse",
    "HippodromeSimple",
    "HippodromeList",
    # Reunion
    "ReunionCreate",
    "ReunionUpdate",
    "ReunionResponse",
    "ReunionSimple",
    "ReunionWithHippodrome",
    "ReunionList",
    "ReunionDetailResponse",
    # Course
    "CourseCreate",
    "CourseUpdate",
    "CourseResponse",
    "CourseSimple",
    "CourseWithReunion",
    "CourseList",
    "CourseDetailResponse",
    "CourseFilter",
    # Horse
    "HorseCreate",
    "HorseUpdate",
    "HorseResponse",
    "HorseSimple",
    "HorseList",
    "HorseDetailResponse",
    # Jockey
    "JockeyCreate",
    "JockeyUpdate",
    "JockeyResponse",
    "JockeySimple",
    "JockeyList",
    "JockeyDetailResponse",
    # Trainer
    "TrainerCreate",
    "TrainerUpdate",
    "TrainerResponse",
    "TrainerSimple",
    "TrainerList",
    "TrainerDetailResponse",
    # Partant
    "PartantCreate",
    "PartantUpdate",
    "PartantResponse",
    "PartantSimple",
    "PartantWithRelations",
    "PartantList",
    "PartantDetailResponse",
    "PartantBatchCreate",
    "PartantResultUpdate",
    "PartantOddsUpdate",
]
