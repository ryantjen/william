# tools.py
from __future__ import annotations
import io
import re
import contextlib
import traceback
from typing import List, Tuple, Dict, Any

def extract_code_from_markdown(text: str) -> str:
    """
    Extract Python code from markdown blocks.
    Handles ```python ... ``` or ``` ... ``` blocks.
    If no block found, returns trimmed text (might be raw code).
    """
    text = (text or "").strip()
    
    # Try ```python ... ``` first
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # No markdown block - assume raw code
    return text


def run_python(code: str) -> Tuple[str, List[Any]]:
    """
    Execute Python code in-process and return (text_output, figures).

    - Captures stdout/stderr into text_output
    - Captures matplotlib figures created during execution
    - Pre-injects numpy (np), pandas (pd), matplotlib (plt) for convenience
    """
    code = extract_code_from_markdown(code)
    
    stdout_buf = io.StringIO()
    figures = []

    # Prepare global namespace with common data science libraries
    exec_globals: Dict[str, Any] = {
        "__name__": "__main__",
    }
    
    # Pre-inject numpy and pandas
    try:
        import numpy as np
        exec_globals["np"] = np
        exec_globals["numpy"] = np
    except ImportError:
        pass
    
    try:
        import pandas as pd
        exec_globals["pd"] = pd
        exec_globals["pandas"] = pd
    except ImportError:
        pass

    try:
        # Import matplotlib lazily; safe if user doesn't plot
        import matplotlib.pyplot as plt  # type: ignore
        
        # Close any existing figures to start fresh
        plt.close('all')
        
        # Turn off interactive mode and disable show()
        plt.ioff()
        plt.show = lambda: None
        
        exec_globals["plt"] = plt
    except Exception:
        plt = None  # matplotlib not available

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
            # Execute user code
            exec(code, exec_globals, None)

            # Collect figures if matplotlib is available
            if plt is not None:
                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    figures.append(fig)

    except Exception:
        stdout_buf.write("\nException:\n")
        stdout_buf.write(traceback.format_exc())

    return stdout_buf.getvalue().strip(), figures
