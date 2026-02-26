# global_state.py
import threading
from typing import Callable, Optional

LOG_PATH = ""

START_FLAG = False
FIRST_MAIN = False


EXIT_FLAG = False
INIT_FLAG = False
INIT_LOCK = threading.Lock()

CONFIRM_HOOK: Optional[Callable[[str], bool]] = None

CODE_ENV = None
# WORKFLOW_FLAG = False
# AGENTCREATE_FLAS = False

# AGENT_CREATOR_AGENT_STATE = False
# META_AGENT_LAST_QUERY = ''
# WORKFLOW_CREATOR_AGENT_STATE = False
# WORKFLOW_AGENT_LAST_QUERY = ''

# container_name = 'auto_agent'
# port = 12345
# test_pull_name = 'autoagent_mirror'
# git_clone = True
# local_env = False
