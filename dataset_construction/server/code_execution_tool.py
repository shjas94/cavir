from __future__ import annotations

import json
import logging
import os
import uuid
import datetime
import ast
import base64
import re
import shlex
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Set
import mcp.types as mcp_types
from mcp.server.mcpserver import MCPServer
from pydantic import Field
from interpreters.subprocess_interpreter import SubprocessInterpreter

# --------------------------------------------------------------------------- #
#  Default import whitelist configuration
# --------------------------------------------------------------------------- #

# `math_compute` is a pure numeric calculator. Every input number must already
# be a literal in the generated source.
MATH_IMPORT_WHITELIST = [
    'math', 'statistics', 'decimal', 'fractions', 'numpy', 'scipy', 'sympy', 'pandas', 'collections'
]

# `visualize_data` accepts the same numeric helpers plus matplotlib. No image,
# filesystem, network, or serialization modules are exposed.
VISUALIZATION_IMPORT_WHITELIST = [
    *MATH_IMPORT_WHITELIST, 'matplotlib', 'mpl_toolkits'
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


def _validate_imports(code: str, whitelist: List[str]) -> tuple[bool, List[str], List[str]]:
    """
    Validate that all imports in code are in the whitelist.
    
    Args:
        code: Python source code to validate
        whitelist: List of allowed module names
        
    Returns:
        Tuple of (is_valid, allowed_imports, forbidden_imports)
    """
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

# Absolute so the log lands in dataset_construction/logs (an output folder, not the code folder) whatever CWD the
# host spawned this stdio server from.
CODE_TOOL_LOG = os.getenv("CODE_TOOL_LOG", str(Path(__file__).resolve().parents[1] / "logs" / "code_tool.log"))
Path(CODE_TOOL_LOG).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CODE_TOOL_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  MCP server instance
# --------------------------------------------------------------------------- #

mcp = MCPServer("code_exec")

ScriptFilename = Annotated[
    str,
    Field(
        min_length=1,
        description=(
            "Descriptive basename for the generated Python script, for example "
            "'percentage_change.py'. Paths are stripped and collisions are auto-renamed."
        ),
    ),
]
MathSource = Annotated[
    str,
    Field(
        min_length=1,
        description=(
            "Self-contained Python source for a multi-step calculation. Put every "
            "observed input value directly in the code as a literal and print the result. "
            "Do not read files, images, arguments, interactive input, or network data."
        ),
    ),
]
VisualizationSource = Annotated[
    str,
    Field(
        min_length=1,
        description=(
            "Self-contained matplotlib.pyplot source using only observed literal values. "
            "Create exactly one figure; do not call savefig(), close(), open(), input(), "
            "or access files, images, arguments, or the network."
        ),
    ),
]
TaskCacheDirectory = Annotated[
    str | None,
    Field(description="Host-injected task workspace. Never provide this argument."),
]

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
    """One task's flat workspace: runs terminal commands in a subprocess whose cwd is the workspace.

    Generated source is checked against the import whitelist before it reaches here (``_validate_imports``).
    """

    def __init__(
        self,
        workspace_dir: str,
        *,
        verbose: bool = False,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.verbose = verbose

        # Initialize the interpreter, ensuring it uses the correct workspace directory
        self.interpreter = self._initialize_interpreter(self.workspace_dir)

    def _initialize_interpreter(self, work_dir: str):
        """Initializes the correct interpreter."""
        
        # Default to SubprocessInterpreter
        return SubprocessInterpreter(
            print_stdout=self.verbose,
            print_stderr=self.verbose,
            workspace_dir=work_dir,
        )

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

        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)

        # The interpreter already runs the subprocess with cwd=workspace_dir, so
        # this must NOT chdir: the CWD is process-global and this server handles
        # many tasks concurrently.
        try:
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

                output = f"{status}\n\nSTDOUT/STDERR:\n{''.join(output_lines)}"

            # Handle simple string output for older interpreter versions
            elif isinstance(result, str):
                # The interpreter folds the exit status into the text, so the
                # status header must be derived from it: labelling every string
                # result as a success hides failing scripts from the caller.
                failure = _detect_execution_error(result)
                status = f"⚠️ Command failed: {failure}." if failure else "✅ Command executed."
                output = f"{status}\n\nOutput:\n{result}"

            # Handle other potential result formats
            else:
                output = f"✅ Command executed.\n\nResult:\n{str(result)}"

        except Exception as e:
            output = f"❌ Error executing terminal command: {str(e)}"

        _log_execution(
            "execute_terminal_command",
            {"command": command, "output": output},
            self.workspace_dir,
        )
        return output


# Global sandbox instances per task (for environment persistence).
# Bounded as an LRU cache: without an eviction policy, one entry accumulates
# per unique workspace_dir for the entire lifetime of this MCP server
# process, which grows unboundedly across a long run (e.g. thousands of
# dataset samples) and never frees memory.
_MAX_SANDBOXES = int(os.getenv("MAX_TASK_SANDBOXES", "64"))
_task_sandboxes: "OrderedDict[str, _UnifiedWorkspaceSandbox]" = OrderedDict()

def _get_or_create_sandbox(
    workspace_dir: str,
    verbose: bool,
) -> _UnifiedWorkspaceSandbox:
    """
    Get the existing sandbox for the workspace or create a new one.

    This function uses a global LRU-bounded dictionary to cache sandbox
    instances based on the workspace directory. This ensures that the same
    sandbox is used for all operations within the same task, preserving
    state, while evicting the least-recently-used entries once the cache
    exceeds `_MAX_SANDBOXES` so memory usage doesn't grow without bound over
    a long-running process handling many distinct workspaces.
    
    Args:
        workspace_dir: The absolute path to the workspace directory.
        verbose: Whether to enable verbose logging.
        
    Returns:
        An instance of _UnifiedWorkspaceSandbox.
    """
    global _task_sandboxes
    if workspace_dir in _task_sandboxes:
        # Mark as most-recently-used.
        _task_sandboxes.move_to_end(workspace_dir)
        return _task_sandboxes[workspace_dir]

    if verbose:
        print(f"Creating new sandbox for workspace: {workspace_dir}")
    _task_sandboxes[workspace_dir] = _UnifiedWorkspaceSandbox(
        workspace_dir=workspace_dir,
        verbose=verbose,
    )

    # Evict least-recently-used sandboxes once over capacity.
    while len(_task_sandboxes) > _MAX_SANDBOXES:
        evicted_workspace_dir, _ = _task_sandboxes.popitem(last=False)
        if verbose:
            print(f"Evicting least-recently-used sandbox for workspace: {evicted_workspace_dir}")

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


# The interpreter never raises on a failing script: it reports the failure
# inside the returned text only, so these markers are the single source of
# truth for whether the executed code actually succeeded.
_EXIT_CODE_RE = re.compile(r"\(Execution failed with return code (-?\d+)\)")
_NONZERO_EXIT_RE = re.compile(r"Command finished with non-zero exit code:\s*(-?\d+)")
_TIMEOUT_RE = re.compile(r"Process timed out after \d+ seconds")


def _detect_execution_error(execution_result: str) -> str:
    """
    Classify a sandbox execution result as failed or succeeded.

    Args:
        execution_result: Raw or status-wrapped output of a sandbox run.

    Returns:
        A short human-readable failure description, or `""` when the run
        finished successfully.
    """
    if not execution_result:
        return ""

    if _TIMEOUT_RE.search(execution_result):
        return "execution timed out"

    match = _EXIT_CODE_RE.search(execution_result) or _NONZERO_EXIT_RE.search(execution_result)
    if match:
        return f"execution failed with return code {match.group(1)}"

    stripped = execution_result.lstrip()
    if stripped.startswith("❌"):
        first_line = stripped.splitlines()[0].lstrip("❌").strip()
        return first_line or "execution failed"

    return ""


# --------------------------------------------------------------------------- #
#  Generated-script contract
# --------------------------------------------------------------------------- #
# Both tools are transformations over values already observed by the model.
# Generated source may not acquire new information from files, arguments,
# images, the network, or interactive input.
_FORBIDDEN_CALLS = {
    "input": "input()",
    "open": "open()",
    "eval": "eval()",
    "exec": "exec()",
    "compile": "compile()",
    "__import__": "__import__()",
}
_FORBIDDEN_ATTRIBUTE_CALLS = {
    # Common numeric/data-library file readers.
    "load": "file loading",
    "loadtxt": "file loading",
    "genfromtxt": "file loading",
    "fromfile": "file loading",
    "memmap": "file loading",
    "read_csv": "file reading",
    "read_excel": "file reading",
    "read_feather": "file reading",
    "read_gbq": "network data reading",
    "read_hdf": "file reading",
    "read_json": "file reading",
    "read_orc": "file reading",
    "read_pickle": "file reading",
    "read_parquet": "file reading",
    "read_sas": "file reading",
    "read_spss": "file reading",
    "read_sql": "database or network data reading",
    "read_sql_query": "database or network data reading",
    "read_sql_table": "database or network data reading",
    "read_stata": "file reading",
    "read_table": "file reading",
    "read_fwf": "file reading",
    "read_html": "external data reading",
    "read_xml": "file reading",
    "read_clipboard": "external data reading",
    "read": "file reading",
    "imread": "image reading",
    "loadmat": "file loading",
    # The visualization tool owns its one output file. Model-generated source
    # must not write arbitrary additional artifacts.
    "save": "file writing",
    "savefig": "file writing",
    "savemat": "file writing",
    "imsave": "file writing",
    "savetxt": "file writing",
    "write": "file writing",
    "to_csv": "file writing",
    "to_excel": "file writing",
    "to_json": "file writing",
    "to_pickle": "file writing",
    "to_parquet": "file writing",
}

# A single arithmetic step is cheap enough to do in the reasoning, so `math_compute`
# is reserved for computations with several steps or an aggregation.
# Only these four operations count as one plain step. Power, floor-division and
# modulo remain valid math_compute work.
_ARITHMETIC_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
# Calls that merely format/pass through a value instead of computing something.
_PASSTHROUGH_CALLS = {"print", "round", "float", "int", "abs"}
# Nodes that prove the script does more than one plain arithmetic step.
_MULTI_STEP_NODES = (
    ast.For, ast.While, ast.comprehension, ast.Lambda, ast.FunctionDef,
    ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.IfExp, ast.Compare, ast.BoolOp,
)


def _is_trivial_script(code: str) -> bool:
    """
    Detect scripts that compute nothing (bare `pass`, or printing only literals).

    These are the "meaningless" code-tool calls we want to reject before paying
    for a sandbox round-trip: the model already knows the printed constant, so
    running it adds no information to the trajectory.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(node, ast.Expr):
            value = node.value
            if isinstance(value, ast.Constant):  # docstring / bare literal
                continue
            is_constant_print = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "print"
                and all(isinstance(arg, ast.Constant) for arg in value.args)
            )
            if is_constant_print:
                continue
        return False
    return True


def _is_single_step_arithmetic(code: str) -> bool:
    """
    Detect scripts whose whole computation is one plain arithmetic step.

    Such a step belongs in the model's own reasoning, not in a sandbox
    round-trip. Anything richer — several chained operations, an aggregation (`sum`, `max`, ...), a
    loop, a comparison or a library call — is genuinely `math_compute` territory.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    operations = 0
    for node in ast.walk(tree):
        if isinstance(node, _MULTI_STEP_NODES):
            return False
        if isinstance(node, ast.Call):
            # Any real function/method call (sum, np.mean, math.sqrt, ...) is
            # more than one plain step.
            if not (isinstance(node.func, ast.Name) and node.func.id in _PASSTHROUGH_CALLS):
                return False
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ARITHMETIC_OPS):
                return False
            operations += 1

    return operations <= 1


def _call_name(call: ast.Call) -> str | None:
    """Return the final identifier of a direct or attribute call."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _validate_common_script(
    code: str, import_whitelist: List[str]
) -> tuple[ast.Module | None, str]:
    """
    Enforce the self-contained source contract shared by both tools.

    Args:
        code: Python source the model produced.
        import_whitelist: Complete set of modules allowed for this tool.

    Returns:
        Parsed syntax tree plus an empty error, or ``None`` plus an actionable
        rejection message.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return None, f"❌ INVALID PYTHON: {exc.msg} at line {exc.lineno}."

    if _is_trivial_script(code):
        return None, (
            "❌ NO-OP CODE REJECTED: the script only prints constants or does nothing.\n"
            "💡 Use the tool only to calculate from or visualize numeric values "
            "already observed in the question or image."
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "__file__":
            return None, (
                "❌ EXTERNAL INPUT NOT ALLOWED: the code uses __file__.\n"
                "💡 Put every numeric input directly in the source as a literal."
            )
        if isinstance(node, ast.Attribute) and node.attr == "argv":
            return None, (
                "❌ EXTERNAL INPUT NOT ALLOWED: the code uses sys.argv.\n"
                "💡 Put every numeric input directly in the source as a literal."
            )
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        if call_name in _FORBIDDEN_CALLS:
            return None, (
                f"❌ EXTERNAL INPUT NOT ALLOWED: the code uses {_FORBIDDEN_CALLS[call_name]}.\n"
                "The script receives no arguments, files, images, network data, or interactive input.\n"
                "💡 Put every numeric input directly in the source as a literal."
            )
        if call_name in _FORBIDDEN_ATTRIBUTE_CALLS:
            return None, (
                f"❌ EXTERNAL ACCESS NOT ALLOWED: `{call_name}()` performs "
                f"{_FORBIDDEN_ATTRIBUTE_CALLS[call_name]}.\n"
                "💡 Use only numeric literals already observed. The tool manages its "
                "own script."
            )

    is_valid, _, forbidden_imports = _validate_imports(code, import_whitelist)
    if not is_valid:
        return None, (
            f"❌ FORBIDDEN IMPORT: {', '.join(sorted(forbidden_imports))}.\n"
            f"Allowed modules: {', '.join(sorted(import_whitelist))}.\n"
            "💡 Filesystem, network, image-loading and process modules are unavailable."
        )

    return tree, ""


def _validate_math_script(code: str) -> str:
    """Validate a self-contained, multi-step numeric computation."""
    tree, error = _validate_common_script(code, MATH_IMPORT_WHITELIST)
    if error:
        return error
    assert tree is not None

    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        for node in ast.walk(tree)
    ):
        return (
            "❌ RESULT NOT PRINTED: math_compute returns the script's stdout.\n"
            "💡 Print every result needed to answer the question, e.g. "
            "`print(round(result, 2))`."
        )

    if _is_single_step_arithmetic(code):
        return (
            "❌ SINGLE-STEP CALCULATION REJECTED: the script performs at most one "
            "arithmetic operation.\n"
            "💡 Do a single add, subtract, multiply, divide or round directly in your reasoning. "
            "Use math_compute only when the calculation needs several chained steps or an "
            "aggregation over many numbers (sum/mean/min/max, sorting, a loop)."
        )

    return ""


_PLOT_CALLS = {
    "plot", "scatter", "bar", "barh", "hist", "pie", "stem", "step",
    "fill", "fill_between", "stackplot", "errorbar", "boxplot", "violinplot",
    "contour", "contourf", "pcolormesh", "imshow", "quiver", "plot_surface",
    "plot_wireframe", "add_patch", "add_artist", "annotate", "text",
}


def _imports_matplotlib_pyplot(tree: ast.Module) -> bool:
    """Whether source explicitly imports pyplot through a supported form."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "matplotlib.pyplot" for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (
            node.module == "matplotlib.pyplot"
            or (
                node.module == "matplotlib"
                and any(alias.name == "pyplot" for alias in node.names)
            )
        ):
            return True
    return False


def _validate_visualization_script(code: str) -> str:
    """Validate code that creates one matplotlib visualization from literals."""
    tree, error = _validate_common_script(code, VISUALIZATION_IMPORT_WHITELIST)
    if error:
        return error
    assert tree is not None

    if not _imports_matplotlib_pyplot(tree):
        return (
            "❌ MATPLOTLIB PYPLOT REQUIRED: visualize_data saves pyplot's current figure.\n"
            "💡 Import `matplotlib.pyplot as plt`, create the plot from literal "
            "values, and do not call savefig(); the tool saves it automatically."
        )

    calls = {
        name for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _call_name(node))
    }
    if not calls.intersection(_PLOT_CALLS):
        return (
            "❌ NO VISUALIZATION CREATED: no matplotlib plotting call was found.\n"
            "💡 Create a plot with plot(), scatter(), bar(), hist(), add_patch(), "
            "or another matplotlib drawing method."
        )
    if "close" in calls:
        return (
            "❌ FIGURE CLOSED BEFORE RETURN: do not call close().\n"
            "💡 Leave the figure open; visualize_data saves and closes it after "
            "your code finishes."
        )
    return ""


def _allocate_artifact_paths(
    workspace_dir: str, filename: str, include_plot: bool
) -> tuple[Path, Path | None]:
    """Allocate non-conflicting script and optional plot paths."""
    normalized = _normalize_filename(filename)
    base_path = Path(normalized)
    if base_path.suffix.lower() != ".py":
        base_path = base_path.with_suffix(".py")

    stem = base_path.stem
    candidate_stem = stem
    index = 0
    while True:
        script_path = Path(workspace_dir) / f"{candidate_stem}.py"
        plot_path = (
            Path(workspace_dir) / f"{candidate_stem}_plot.png"
            if include_plot else None
        )
        if not script_path.exists() and (plot_path is None or not plot_path.exists()):
            return script_path, plot_path
        index += 1
        candidate_stem = f"{stem}_{index}"


def _execution_failure(error: str) -> Dict[str, str]:
    """Standard payload for a script rejected before execution."""
    return {
        "file_write_log": error,
        "execution_result": "",
        "execution_error": error.splitlines()[0].lstrip("❌").strip(),
        "script_path": "",
    }


def _write_and_execute(
    tool_name: str,
    filename: str,
    source: str,
    task_cache_dir: str | None,
    verbose: bool,
    *,
    plot: bool = False,
) -> tuple[Dict[str, str], bytes | None]:
    """Write a unique script, execute it, and optionally collect its PNG."""
    workspace_dir = _get_workspace_dir(task_cache_dir)
    try:
        script_path, plot_path = _allocate_artifact_paths(
            workspace_dir, filename, include_plot=plot
        )
    except ValueError as e:
        return _execution_failure(f"❌ Filename error: {e}"), None

    executable_source = source
    if plot:
        assert plot_path is not None
        executable_source = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            f"{source.rstrip()}\n\n"
            "# Added by visualize_data: persist the current figure.\n"
            "import matplotlib.pyplot as _cavir_plt\n"
            f"_cavir_plt.gcf().savefig({plot_path.name!r}, format='png', "
            "bbox_inches='tight')\n"
            "_cavir_plt.close(_cavir_plt.gcf())\n"
        )
    
    try:
        script_path.write_text(executable_source, encoding="utf-8")
        logger.info(f"Created file: {script_path.name}")
        log_lines = [
            f"✅ File CREATED: {script_path.name}",
            f"📁 Workspace: {workspace_dir}",
            f"📊 Source size: {len(source)} characters",
        ]
        if plot_path is not None:
            log_lines.append(f"🖼️ Plot: {plot_path.name}")
        file_write_log = "\n".join(log_lines)
    except Exception as e:
        error_msg = f"Failed to write file {script_path.name}: {e}"
        logger.error(error_msg)
        return _execution_failure(f"❌ {error_msg}"), None
    
    command = f"conda run -n research python {shlex.quote(script_path.name)}"
    sandbox_instance = _get_or_create_sandbox(workspace_dir, verbose)
    execution_result = sandbox_instance.execute_terminal_command(command)
    execution_error = _detect_execution_error(execution_result)

    plot_bytes = None
    if plot and not execution_error:
        assert plot_path is not None
        try:
            plot_bytes = plot_path.read_bytes()
            if not plot_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("output is not a valid PNG file")
        except Exception as exc:
            execution_error = f"plot output missing or invalid: {exc}"

    _log_execution(
        tool_name,
        {
            "filename": script_path.name,
            "path": str(script_path),
            "content_length": len(source),
            "plot_path": str(plot_path) if plot_path else "",
            "execution_result": execution_result,
            "execution_error": execution_error,
        },
        workspace_dir,
    )

    payload = {
        "file_write_log": file_write_log,
        "execution_result": execution_result,
        "execution_error": execution_error,
        "script_path": str(script_path),
    }
    if plot_path is not None:
        payload["plot_path"] = str(plot_path) if plot_bytes is not None else ""
        payload["plot_filename"] = plot_path.name
    return payload, plot_bytes


@mcp.tool()
async def math_compute(
    filename: ScriptFilename,
    content: MathSource,
    task_cache_dir: TaskCacheDirectory = None,
) -> Dict[str, str]:
    """
    Perform a self-contained multi-step calculation on numeric values already known.

    USE for percentages, multi-step arithmetic, statistics, equations, or
    geometry. DO NOT use for a single add, subtract, multiply, divide, or round;
    do that step in your reasoning instead. Power, modulo, floor division,
    library functions, and aggregations are valid here even when expressed in
    one statement.

    Put every source value directly in `content` as a literal and print the
    final result. Never inspect the input image or acquire data with files,
    input(), sys.argv, the network, or non-whitelisted imports. Returns the
    script execution record; the printed stdout is the calculation result.

    Example content:
    `old=120; new=150; print((new-old)/old*100)`
    """
    contract_error = _validate_math_script(content)
    if contract_error:
        logger.info(f"Rejected math_compute script: {contract_error.splitlines()[0]}")
        return _execution_failure(contract_error)

    payload, _ = _write_and_execute(
        "math_compute", filename, content, task_cache_dir, False
    )
    return payload


@mcp.tool()
async def visualize_data(
    filename: ScriptFilename,
    content: VisualizationSource,
    task_cache_dir: TaskCacheDirectory = None,
) -> mcp_types.CallToolResult:
    """
    Create one matplotlib plot from numeric values already known.

    USE only when a visual representation helps compare patterns, inspect a
    trend, plot coordinates, or sketch geometry. DO NOT use this tool to read
    or analyze the input image; use image tools for that.

    Put every plotted value directly in `content` as a literal. Import
    `matplotlib.pyplot as plt` and create exactly one figure. Do not call
    savefig(), close(), open(), input(), sys.argv, or any file/network/image
    reader. The tool automatically saves `<script_stem>_plot.png` and returns
    that PNG as an image.

    Example content:
    `import matplotlib.pyplot as plt; plt.plot([1,2,3], [4,7,5])`
    """
    contract_error = _validate_visualization_script(content)
    if contract_error:
        logger.info(f"Rejected visualize_data script: {contract_error.splitlines()[0]}")
        payload = _execution_failure(contract_error)
        return mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(
                    type="text", text=json.dumps(payload, ensure_ascii=False)
                )
            ]
        )

    payload, plot_bytes = _write_and_execute(
        "visualize_data", filename, content, task_cache_dir, False, plot=True
    )
    content_blocks: list[mcp_types.ContentBlock] = [
        mcp_types.TextContent(
            type="text", text=json.dumps(payload, ensure_ascii=False)
        )
    ]
    if plot_bytes is not None:
        content_blocks.append(
            mcp_types.ImageContent(
                type="image",
                data=base64.b64encode(plot_bytes).decode("ascii"),
                mimeType="image/png",
                _meta={
                    "plot_path": payload["plot_path"],
                    "plot_filename": payload["plot_filename"],
                },
            )
        )
    return mcp_types.CallToolResult(content=content_blocks)

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")
