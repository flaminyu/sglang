"""
RequestLog: Reads pre-recorded request logs for simulation.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, TextIO


@dataclass
class RequestLogEntry:
    rid: str
    program_id: str
    turn_index: int
    token_ids: List[int]
    input_len: int
    output_len: int
    arrival_time: float
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    extra_key: Optional[str] = None
    # Real tool execution time in seconds (for CDF calculation)
    actual_tool_duration: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequestLogEntry":
        return cls(
            rid=str(data.get("rid", data.get("request_id", ""))),
            program_id=str(data.get("program_id", data.get("conversation_id", ""))),
            turn_index=int(data.get("turn_index", 0)),
            token_ids=data.get("token_ids", data.get("input_ids", [])),
            input_len=int(data.get("input_len", data.get("input_length", len(data.get("token_ids", []))))),
            output_len=int(data.get("output_len", data.get("output_length", 0))),
            arrival_time=float(data.get("arrival_time", data.get("timestamp", 0.0))),
            is_tool_call=bool(data.get("is_tool_call", data.get("tool_call", False))),
            tool_name=data.get("tool_name"),
            extra_key=data.get("extra_key"),
            actual_tool_duration=data.get("actual_tool_duration"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rid": self.rid,
            "program_id": self.program_id,
            "turn_index": self.turn_index,
            "token_ids": self.token_ids,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "arrival_time": self.arrival_time,
            "is_tool_call": self.is_tool_call,
            "tool_name": self.tool_name,
            "extra_key": self.extra_key,
        }


class RequestLogReader:
    """Reads pre-recorded request logs from files."""
    
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self.entries: List[RequestLogEntry] = []
        self._index = 0
    
    def load(self, log_path: Optional[str] = None) -> int:
        if log_path:
            self.log_path = log_path
        
        if not self.log_path:
            raise ValueError("No log_path specified")
        
        self.entries = []
        
        with open(self.log_path, 'r') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                self._load_json_array(f)
            else:
                self._load_jsonl(f)
        
        self.entries.sort(key=lambda e: e.arrival_time)
        return len(self.entries)
    
    def _load_jsonl(self, f: TextIO) -> None:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entry = RequestLogEntry.from_dict(data)
                self.entries.append(entry)
            except Exception as e:
                print(f"Warning: Failed to parse line {line_num + 1}: {e}")
    
    def _load_json_array(self, f: TextIO) -> None:
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    try:
                        entry = RequestLogEntry.from_dict(item)
                        self.entries.append(entry)
                    except Exception as e:
                        print(f"Warning: Failed to parse entry: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file: {e}")
    
    def iter_requests(self) -> Iterator["RequestLogEntry"]:
        for entry in self.entries:
            yield entry
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self):
        return iter(self.entries)
    
    def __getitem__(self, index: int) -> RequestLogEntry:
        return self.entries[index]
    
    @staticmethod
    def extract_ttl(extra_key: Optional[str]) -> Optional[float]:
        if extra_key is None:
            return None
        match = re.search(r'__ttl=(\d+\.?\d*)', str(extra_key))
        if match:
            return float(match.group(1))
        return None


class RequestLogWriter:
    """Writes request logs to files."""
    
    def __init__(self, log_path: str, format: str = "jsonl"):
        self.log_path = log_path
        self.format = format
        self.entries: List[Dict[str, Any]] = []
    
    def add_entry(self, entry: RequestLogEntry) -> None:
        self.entries.append(entry.to_dict())
    
    def add_entry_dict(self, data: Dict[str, Any]) -> None:
        self.entries.append(data)
    
    def write(self) -> int:
        with open(self.log_path, 'w') as f:
            if self.format == "jsonl":
                for entry in self.entries:
                    f.write(json.dumps(entry) + "\n")
            else:
                json.dump(self.entries, f, indent=2)
        return len(self.entries)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write()
