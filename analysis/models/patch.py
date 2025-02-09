from __future__ import annotations
from pydantic import BaseModel


class Diff(BaseModel):
    """Represents a diff between two versions of a file."""
    before: str
    after: str

class Patch(BaseModel):
    """Represents a git patch."""
    files: dict[str, Diff]

    @staticmethod
    def from_str(patch: str) -> Patch:
        """Parse a git patch into a Patch object."""
        files = {}
        current_file = None
        current_before = []
        current_after = []
        
        try:
            lines = patch.split('\n')
            for line in lines:
                if line.startswith('diff --git'):
                    # Save previous file if exists
                    if current_file:
                        files[current_file] = Diff(
                            before='\n'.join(current_before),
                            after='\n'.join(current_after)
                        )
                    # Start new file
                    current_file = line.split()[-1].lstrip('b/')
                    current_before = []
                    current_after = []
                elif line.startswith('+++') or line.startswith('---'):
                    continue
                elif line.startswith('+'):
                    current_after.append(line[1:])
                elif line.startswith('-'):
                    current_before.append(line[1:])
                elif line.startswith(' '):
                    current_before.append(line[1:])
                    current_after.append(line[1:])
            
            # Save last file
            if current_file:
                files[current_file] = Diff(
                    before='\n'.join(current_before),
                    after='\n'.join(current_after)
                )
        except Exception:
            pass
        
        return files