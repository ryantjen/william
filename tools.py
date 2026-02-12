# tools.py
from __future__ import annotations
import io
import contextlib
import traceback
from typing import List, Tuple, Dict, Any

def run_python(code: str) -> Tuple[str, List[Any]]:
    """
    Execute Python code in-process and return (text_output, figures).

    - Captures stdout/stderr into text_output
    - Captures matplotlib figures created during execution
    """
    stdout_buf = io.StringIO()
    figures = []

    # Prepare a constrained-ish global namespace
    # (This is NOT secure against malicious code—fine for personal use.)
    exec_globals: Dict[str, Any] = {
        "__name__": "__main__",
    }

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
