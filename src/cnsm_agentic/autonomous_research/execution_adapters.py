from __future__ import annotations
from pathlib import Path
from typing import Any, Protocol

class ExecutionAdapter(Protocol):
    family:str
    def supports(self,plan:dict[str,Any])->bool:...
    def execute(self,*,plan:dict[str,Any],preregistration:dict[str,Any],output_dir:Path)->dict[str,Any]:...

_ADAPTERS:list[ExecutionAdapter]=[]
def register_adapter(adapter:ExecutionAdapter)->None:_ADAPTERS.append(adapter)
def resolve_adapter(plan:dict[str,Any])->ExecutionAdapter|None:
    return next((a for a in _ADAPTERS if a.supports(plan)),None)
