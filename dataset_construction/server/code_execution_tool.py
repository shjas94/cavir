from __future__ import annotations

import base64
import json
import logging
import os
import uuid
import datetime
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set
from PIL import Image
from io import BytesIO
from mcp.server.fastmcp import FastMCP
from interpreters.subprocess_interpreter import SubprocessInterpreter

# --------------------------------------------------------------------------- #
#  Default import whitelist configuration
# --------------------------------------------------------------------------- #

DEFAULT_IMPORT_WHITELIST = [
    # Standard library
    'math', 'random', 're', 'collections', 'itertools', 'functools', 'operator',
    'typing', 'base64', 'logging', 'warnings',
    
    # Data science core libraries
    'numpy', 'pandas', 'pd', 'matplotlib', 'seaborn',
    'scipy', 'sklearn', 'plotly', 'dash'
    
    # Image processing
    'PIL', 'cv2', 'skimage', 'imageio',

    # Other common tools
    'tqdm'
]

# --------------------------------------------------------------------------- #
#  Import whitelist validation functions
# --------------------------------------------------------------------------- #

def _extract_imports_from_code(code: str) -> Set[str]:
    """
    Extract all imported module names from Python code.
    
    Args:
        code: Python source code
        
    Returns:
        Set of imported module names
    """
    imports = set()
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Get the top-level module name
                    module_name = name.name.split('.')[0]
                    imports.add(module_name)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Get the top-level module name
                    module_name = node.module.split('.')[0]
                    imports.add(module_name)
                    
    except SyntaxError:
        # If we can't parse the code, also try regex fallback
        pass
    
    # Regex fallback for cases where AST parsing fails
    import_patterns = [
        r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
    ]
    
    for pattern in import_patterns:
        matches = re.findall(pattern, code, re.MULTILINE)
        for match in matches:
            imports.add(match.split('.')[0])
    
    return imports


def _validate_imports(code: str, whitelist: Optional[List[str]] = None) -> tuple[bool, List[str], List[str]]:
    """
    Validate that all imports in code are in the whitelist.
    
    Args:
        code: Python source code to validate
        whitelist: List of allowed module names (defaults to DEFAULT_IMPORT_WHITELIST)
        
    Returns:
        Tuple of (is_valid, allowed_imports, forbidden_imports)
    """
    if whitelist is None:
        whitelist = DEFAULT_IMPORT_WHITELIST
    
    whitelist_set = set(whitelist)
    imports = _extract_imports_from_code(code)
    
    allowed_imports = []
    forbidden_imports = []
    
    for imp in imports:
        if imp in whitelist_set:
            allowed_imports.append(imp)
        else:
            forbidden_imports.append(imp)
    
    is_valid = len(forbidden_imports) == 0
    return is_valid, allowed_imports, forbidden_imports

# --------------------------------------------------------------------------- #
#  Logger setup
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('code_tool.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("code_exec")

# --------------------------------------------------------------------------- #
#  Simplified workspace management
# --------------------------------------------------------------------------- #

def _get_workspace_dir(task_cache_dir: Optional[str] = None) -> str:
    """
    Get or create the unified workspace directory for a task.
    
    📁 DIRECTORY STRUCTURE:
        ../sandbox  (flat structure - all files here)
    
    Args:
        task_cache_dir: Task-specific cache directory path. This is REQUIRED.
        
    Returns:
        str: Absolute path to workspace directory
        
    Raises:
        ValueError: If task_cache_dir is not provided.
    """
    if not task_cache_dir:
        raise ValueError("task_cache_dir must be provided to locate the workspace.")
    
    workspace_dir = Path(task_cache_dir) / "sandbox"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return str(workspace_dir)


def _log_execution(operation: str, details: dict, workspace_dir: str) -> None:
    """
    Log execution details for debugging and tracking.
    
    Args:
        operation: Type of operation performed
        details: Operation-specific details
        workspace_dir: Workspace directory path
    """
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": operation,
            "workspace": workspace_dir,
            "details": details
        }
        
        log_file = Path(workspace_dir).parent / "execution.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    except Exception as e:
        logger.error(f"Failed to log execution: {e}")
        
# --------------------------------------------------------------------------- #
#  Enhanced sandbox wrapper with unified workspace
# --------------------------------------------------------------------------- #

class _UnifiedWorkspaceSandbox:
    """
    Unified sandbox with flat workspace structure and import whitelist validation.
    
    🎯 KEY FEATURES:
        • All files in single workspace directory
        • Persistent environment across executions
        • Terminal command support
        • File overwrite behavior (no versioning)
        • Import whitelist validation for security
    
    🔒 IMPORT SECURITY:
        • Validates all import statements against whitelist
        • Supports torch, transformers, and other ML libraries
        • Blocks unauthorized imports for security
    """

    def __init__(
        self,
        workspace_dir: str,
        sandbox: Literal[
            "internal_python"
        ] = "subprocess",
        *,
        verbose: bool = False,
        unsafe_mode: bool = False,
        import_whitelist: Optional[list[str]] = None,
        require_confirm: bool = False,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.verbose = verbose
        
        # Set up import whitelist
        self.import_whitelist = import_whitelist or DEFAULT_IMPORT_WHITELIST.copy()
        self.unsafe_mode = unsafe_mode  # If True, skip whitelist validation
        self.require_confirm = require_confirm

        # Initialize the interpreter, ensuring it uses the correct workspace directory
        self.interpreter = self._initialize_interpreter(sandbox, self.workspace_dir)

    def _initialize_interpreter(self, sandbox_type: str, work_dir: str):
        """Initializes the correct interpreter."""
        
        # Default to SubprocessInterpreter
        return SubprocessInterpreter(
            require_confirm=self.require_confirm,
            print_stdout=self.verbose,
            print_stderr=self.verbose,
            workspace_dir=work_dir,
        )

    def execute_code(self, code: str, filename: str) -> str:
        """
        Execute Python code with import whitelist validation and save to specified filename.
        
        🔧 EXECUTION PROCESS:
            1. Validate imports against whitelist
            2. Change to workspace directory
            3. Write code to specified filename (OVERWRITES existing)
            4. Execute code in persistent environment
            5. Return execution result with validation info
        
        🔒 WHITELIST VALIDATION:
            • Checks all import statements in code
            • Allows: torch, transformers, numpy, pandas, matplotlib, etc.
            • Blocks: unauthorized system modules, network libraries not in whitelist
        
        Args:
            code: Python code to execute
            filename: Target filename (REQUIRED, will overwrite if exists)
            
        Returns:
            str: Execution output with import validation results and any error messages
        """
        # Ensure workspace directory exists
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        os.chdir(self.workspace_dir)
        
        try:
            # Validate imports if not in unsafe mode
            validation_result = ""
            if not self.unsafe_mode:
                is_valid, allowed_imports, forbidden_imports = _validate_imports(code, self.import_whitelist)
                
                if not is_valid:
                    error_msg = f"❌ IMPORT VALIDATION FAILED\n"
                    error_msg += f"Forbidden imports: {', '.join(forbidden_imports)}\n"
                    error_msg += f"Allowed imports in whitelist:\n"
                    for item in sorted(self.import_whitelist):
                        error_msg += f"  • {item}\n"
                    error_msg += f"\n💡 Contact admin to add other packages to whitelist."
                    return error_msg
                
                if allowed_imports:
                    validation_result = f"✅ IMPORTS VALIDATED: {', '.join(sorted(allowed_imports))}\n"
                    validation_result += f"🔒 Whitelist contains: {len(self.import_whitelist)} approved modules\n"
                    validation_result += "=" * 50 + "\n"
            
            # Write code to file (OVERWRITE if exists)
            code_file = Path(self.workspace_dir) / filename
            file_existed = code_file.exists()
            code_file.write_text(code, encoding="utf-8")
            logger.info(f"Code written to: {filename} ({'OVERWRITTEN' if file_existed else 'CREATED'})")
            
            # Execute the code
            execution_result = self.interpreter.run(code, code_type="python")
            
            # Combine validation and execution results
            full_result = validation_result + execution_result
            return full_result
            
        finally:
            os.chdir(original_cwd)

    def execute_terminal_command(self, command: str) -> str:
        """
        Execute a shell command in the workspace directory.
        
        Args:
            command: The shell command to execute.
            
        Returns:
            A string containing the stdout and stderr of the command.
        """
        if self.verbose:
            print(f"Executing terminal command: {command} in {self.workspace_dir}")

        log_details = {"command": command}
        _log_execution("execute_terminal_command", log_details, self.workspace_dir)
        
        # Ensure workspace directory exists before attempting to change to it
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        os.chdir(self.workspace_dir)
        
        try:
            # The interpreter is already configured with the correct working directory.
            # We pass 'bash' to indicate it's a shell command.
            result = self.interpreter.run(command, code_type="bash")

            # Process result based on its type
            if isinstance(result, tuple) and len(result) == 2:
                # Assuming (exit_code, logs)
                exit_code, logs = result
                if isinstance(logs, list):
                    output_lines = [log.content for log in logs]
                else: # Assuming logs is a string
                    output_lines = [str(logs)]

                if exit_code == 0:
                    status = "✅ Command executed successfully."
                else:
                    status = f"⚠️ Command finished with non-zero exit code: {exit_code}."
                
                return f"{status}\n\nSTDOUT/STDERR:\n{''.join(output_lines)}"

            # Handle simple string output for older interpreter versions
            elif isinstance(result, str):
                return f"✅ Command executed.\n\nOutput:\n{result}"
            
            # Handle other potential result formats
            else:
                return f"✅ Command executed.\n\nResult:\n{str(result)}"

        except Exception as e:
            return f"❌ Error executing terminal command: {str(e)}"
        finally:
            os.chdir(original_cwd)


# Global sandbox instances per task (for environment persistence)
_task_sandboxes: Dict[str, _UnifiedWorkspaceSandbox] = {}

def _get_or_create_sandbox(
    workspace_dir: str,
    sandbox: str,
    verbose: bool,
    unsafe_mode: bool,
    import_whitelist: Optional[List[str]] = None
) -> _UnifiedWorkspaceSandbox:
    """
    Get the existing sandbox for the workspace or create a new one.

    This function uses a global dictionary to cache sandbox instances based on
    the workspace directory. This ensures that the same sandbox is used for
all operations within the same task, preserving state.
    
    Args:
        workspace_dir: The absolute path to the workspace directory.
        sandbox: The type of sandbox to create.
        verbose: Whether to enable verbose logging.
        unsafe_mode: Whether to disable security checks (e.g., import validation).
        import_whitelist: A list of allowed Python modules.
        
    Returns:
        An instance of _UnifiedWorkspaceSandbox.
    """
    global _task_sandboxes
    if workspace_dir not in _task_sandboxes:
        if verbose:
            print(f"Creating new sandbox for workspace: {workspace_dir}")
        _task_sandboxes[workspace_dir] = _UnifiedWorkspaceSandbox(
            workspace_dir=workspace_dir,
            sandbox=sandbox,
            verbose=verbose,
            unsafe_mode=unsafe_mode,
            import_whitelist=import_whitelist,
        )
    return _task_sandboxes[workspace_dir]

# --------------------------------------------------------------------------- #
#  Enhanced tools with unified workspace
# --------------------------------------------------------------------------- #

def _normalize_filename(filename: str) -> str:
    """
    Sanitizes and normalizes a filename to prevent directory traversal.

    - Removes leading/trailing whitespace and quotes.
    - Replaces backslashes with forward slashes.
    - Removes any path components (e.g., '/', '..').
    - If the filename becomes empty, it defaults to a UUID-based name.
    
    Args:
        filename: The original filename provided by the user or agent.
        
    Returns:
        A safe, sanitized filename.
    """
    if not isinstance(filename, str):
        filename = str(filename)
        
    # Strip whitespace and quotes
    filename = filename.strip().strip('\'"')
    
    # Standardize path separators
    filename = filename.replace('\\', '/')
    
    # Remove any directory traversal components
    filename = os.path.basename(filename)
    
    # If the filename is empty after sanitization, create a default name
    if not filename:
        filename = f"file_{uuid.uuid4().hex[:8]}.txt"
        
    return filename

@mcp.tool()
async def run_code(
    filename: str,
    content: str,
    task_cache_dir: str | None = None,
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Write Python code to a `.py` file in the workspace and execute it.
    Use this tool when you want to calculate numerical values (calculate average, sum, min, or other complex calculations) or visualize data (drawing charts, plots) by writing and executing Python code.
    
    WARNING WITH IMAGE RESULT:
        When the result of code is images like charts or plots, please save the images to png files in the '/workspace/cavir/dataset_construction/temp_output' and return the full file path in the execution result. 
        The files created here will be visible to subsequent operations, so they can be accessed or manipulated by other tools as needed.
        Format of the result image must be PNG. You can use libraries like matplotlib, seaborn, or plotly to create and save the images. For example:
    
    Workflow Note: This is a one-shot write-and-execute tool. The code is saved
    as a `.py` file in the workspace and immediately run via
    `python {final_filename}` inside the persistent sandbox. To save code
    without executing it, use `write_workspace_file`. To run an existing
    script, use `execute_terminal_command`.

    🔧 EXECUTION PROCESS:
        1. Sanitize `filename` (strip path components, whitespace, quotes).
        2. Ensure the file ends with `.py` — any other extension is coerced
           (e.g. `solve_q1` → `solve_q1.py`, `foo.txt` → `foo.py`).
        3. Resolve filename collisions WITHOUT overwriting: if `{stem}.py`
           already exists, try `{stem}_1.py`, `{stem}_2.py`, … until an
           unused name is found. The chosen name is `final_filename`.
        4. Write `content` to `final_filename` in the workspace.
        5. Execute `python {final_filename}` in the persistent sandbox
           (same workspace as the other workspace tools).
        6. Return both the file-write log and the execution result.

    📝 INPUT:
        • filename: Base name for the script. Path components are stripped
          for safety; `.py` is appended if missing.
        • content: Python source code to write and execute (REQUIRED).
        • task_cache_dir: Task cache directory path (auto-injected).
        • verbose: Include detailed sandbox execution information.

    📤 OUTPUT:
        Returns a dict with two keys:
        • `file_write_log` (str):
            - On success: confirmation containing `final_filename`,
              workspace path, and content size.
            - On filename / write failure: an `❌ ...` error message.
        • `execution_result` (str):
            - On success: STDOUT / STDERR and exit-code status from
              `python {final_filename}`.
            - If the file write failed: empty string (no execution attempted).

    🔒 CONSTRAINTS:
        • Only `.py` files are produced; non-`.py` extensions are coerced.
        • NEVER overwrites existing files — auto-renames with `_1`, `_2`, …
        • Directory traversal in `filename` is stripped (basename only).
        • Runs in the same persistent workspace as the other tools, so any
          files created or state established here are visible to subsequent
          operations.
    
    Args:
        filename: Base name for the script (`.py` is added if missing)
        content: Python code to write into the file and execute
        task_cache_dir: Task cache directory path
        verbose: Whether to include verbose execution details

    Returns:
        Dict[str, str] with keys `file_write_log` and
        `execution_result`. On write failure, `execution_result` is `""`.

    Example Usage:
        run_code("solve_q1", "print(2 + 2)")
        # → writes solve_q1.py (or solve_q1_1.py if solve_q1.py already exists)
        #   and executes it, returning:
        #   {
        #       "file_write_log": "✅ File CREATED: solve_q1.py\\n📁 Workspace: ...\\n📊 Size: 13 characters",
        #       "execution_result": "✅ Command executed successfully.\\n\\nSTDOUT/STDERR:\\n4\\n",
        #   }
    """
    workspace_dir = _get_workspace_dir(task_cache_dir)
    try:
        normalized_filename = _normalize_filename(filename)
    except ValueError as e:
        return {"file_write_log": f"❌ Filename error: {e}", "execution_result": ""}

    # Ensure .py extension (per tool contract)
    base_path = Path(normalized_filename)
    if base_path.suffix.lower() != ".py":
        base_path = base_path.with_suffix(".py")

    stem = base_path.stem            # e.g. "solve_q1"
    suffix = base_path.suffix        # ".py"

    # Find a non-conflicting filename: {stem}.py, {stem}_1.py, {stem}_2.py, ...
    file_path = Path(workspace_dir) / base_path
    n = 1
    while file_path.exists():
        file_path = Path(workspace_dir) / f"{stem}_{n}{suffix}"
        n += 1

    final_filename = file_path.name
    file_existed = False  # by construction, we never overwrite now
    
    try:
        file_path.write_text(content, encoding="utf-8")
        
        # Log the operation
        _log_execution("write_workspace_file", {
            "filename": final_filename,
            "file_existed": file_existed,
            "content_length": len(content)
        }, workspace_dir)
        
        logger.info(f"{'Overwrote' if file_existed else 'Created'} file: {final_filename}")
        
        file_write_log = []
        file_write_log.append(f"✅ File {'RENAMED' if file_existed else 'CREATED'}: {final_filename}")
        file_write_log.append(f"📁 Workspace: {workspace_dir}")
        file_write_log.append(f"📊 Size: {len(content)} characters")
        
        file_write_log = "\n".join(file_write_log)
        
    except Exception as e:
        error_msg = f"Failed to write file {final_filename}: {e}"
        logger.error(error_msg)
        return {
            "file_write_log": f"❌ {error_msg}",
            "execution_result": ""
        }
    
    # Hardcoded to run the python code with my specific environment
    command = f"conda run -n research python {final_filename}"
    workspace_dir = _get_workspace_dir(task_cache_dir)
    # Get or create the sandbox for the workspace
    sandbox_instance = _get_or_create_sandbox(workspace_dir, "subprocess", verbose, False)
    # Execute the command
    execution_result = sandbox_instance.execute_terminal_command(command)
    
    # If resulting object is saved as a file in the workspace (chart or plot image)
    # Load the file content -> convert to the base64 string -> return the base64 string in the execution result
    
    if os.path.isfile(execution_result.strip()) and execution_result.strip().endswith(".png"):
        
        # TODO: move temp file to the result directory
        try:
            with open(execution_result.strip(), "rb") as f:
                file_content = f.read()
                base64_content = base64.b64encode(file_content).decode("utf-8")
                execution_result = f"data:image/png;base64,{base64_content}"
                execution_PIL = base64.b64decode(base64_content)
                execution_PIL = Image.open(BytesIO(execution_PIL))
                execution_PIL.save(f"{task_cache_dir}/{final_filename}_result.png")
                file_write_log += f"\nExecution result is a file: {task_cache_dir}/{final_filename}_result.png"
        except Exception as e:
            execution_result = f"❌ Failed to read or encode the result file: {e}"
            file_write_log += f"\n{execution_result}"
        
    return {
        "file_write_log": file_write_log,
        "execution_result": execution_result
    }

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")
