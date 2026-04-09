MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LOCAL_PKG_DIR ?= $(MAKEFILE_DIR)/.a2d_pkg
COROBOT_WHL ?=
COROBOT_SITE_PACKAGES ?=
ROOT_PYTHON ?= 3.10
MCP_PYTHON ?= 3.12
BASIC_MEMORY_DIR := $(MAKEFILE_DIR)/src/mcp_server_demo/basic-memory
METASEARCH_DIR := $(MAKEFILE_DIR)/src/mcp_server_demo/metasearch-mcp
COROBOT_MCP_DIR := $(MAKEFILE_DIR)/src/mcp_server_demo/corobot_mcp_server
DATA_ANALYST_MCP_DIR := $(MAKEFILE_DIR)/src/mcp_server_demo/data_analyst_mcp_server
X2ROBOT_MCP_DIR := $(MAKEFILE_DIR)/src/mcp_server_demo/x2robot_mcp_server

# 默认仅注入 src；本地扩展目录存在时自动加入；可选追加授权用户的 site-packages
PYTHONPATH_VALUE := $(MAKEFILE_DIR)/src$(if $(wildcard $(LOCAL_PKG_DIR)),:$(LOCAL_PKG_DIR),)$(if $(COROBOT_SITE_PACKAGES),:$(COROBOT_SITE_PACKAGES),)
PYENV := PYTHONPATH=$(PYTHONPATH_VALUE)
LD_LIBRARY := LD_LIBRARY_PATH=/data/opencv45:$LD_LIBRARY_PATH
CURRENT_TIME := $(shell date +"%Y-%m-%d_%H-%M")
UV_RUN_ROOT := uv run --python $(ROOT_PYTHON)


init:
	mkdir -p ./applog/
	git submodule update --init --recursive || true
	uv python install $(ROOT_PYTHON) $(MCP_PYTHON)
	uv sync --frozen --python $(ROOT_PYTHON) --no-build-isolation || uv sync --python $(ROOT_PYTHON) --no-build-isolation
	uv pip install setuptools --python $(ROOT_PYTHON)
	$(UV_RUN_ROOT) --no-build-isolation pre-commit install || true
	@test -d "$(BASIC_MEMORY_DIR)/.git" && uv --directory "$(BASIC_MEMORY_DIR)" sync --frozen --python $(MCP_PYTHON) || echo "[skip] basic-memory submodule not cloned"
	@test -d "$(METASEARCH_DIR)/.git" && uv --directory "$(METASEARCH_DIR)" sync --frozen --python $(MCP_PYTHON) || echo "[skip] metasearch-mcp submodule not cloned"
	uv --directory "$(COROBOT_MCP_DIR)" sync --python $(ROOT_PYTHON)
	uv --directory "$(X2ROBOT_MCP_DIR)" sync --python $(ROOT_PYTHON) || echo "[skip] x2robot_mcp_server sync failed (optional)"
	uv --directory "$(DATA_ANALYST_MCP_DIR)" sync --python $(ROOT_PYTHON) || echo "[skip] data_analyst_mcp_server sync failed (optional)"

install_g01_whl:
	@test -n "$(COROBOT_WHL)" || (echo "请提供 COROBOT_WHL=/path/to/corobot-*.whl" && exit 1)
	@test -f "$(COROBOT_WHL)" || (echo "未找到 whl 文件: $(COROBOT_WHL)" && exit 1)
	mkdir -p "$(LOCAL_PKG_DIR)"
	uv pip install --target "$(LOCAL_PKG_DIR)" "$(COROBOT_WHL)"


test:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/agent_layer/llm_manager/openai_client/test_openai_client.py

run:
	$(MAKE) run_tui

run_a2d:
	$(LD_LIBRARY) $(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/cmd/olympus_img_cmd.py

run_gui:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/gradio_ui/gradio_ui.py

run_tui:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/tui/olympus_tui.py

test_img:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/agent_layer/llm_manager/openai_client/test_img.py

test_mm:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/agent_layer/memory_manager/test_memory_manager.py

test_a2d:
	$(LD_LIBRARY) $(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/machine_layer/test_dataloader_a2d.py

test_udp:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/interaction_layer/test_udp.py

test_a2d_img:
	$(LD_LIBRARY) $(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}tests/agent_demo/agent_layer/test_img_agent.py

# ---- Mock services (no hardware required) ----

run_mock_server:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/mcp_server_demo/corobot_mcp_server/mock_corobot_server.py

run_mock_gui:
	@echo "Starting MockCoRobotServer in background..."
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/mcp_server_demo/corobot_mcp_server/mock_corobot_server.py &
	@sleep 1
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/gradio_ui/gradio_ui.py

run_mock_tui:
	@echo "Starting MockCoRobotServer in background..."
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/mcp_server_demo/corobot_mcp_server/mock_corobot_server.py &
	@sleep 1
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/tui/olympus_tui.py

# ---- x2robot mock services ----

run_x2robot_mock_bridge:
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/mcp_server_demo/x2robot_mcp_server/mock_x2robot_bridge.py

run_x2robot_mock_gui:
	@echo "Starting MockX2RobotBridge in background..."
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/mcp_server_demo/x2robot_mcp_server/mock_x2robot_bridge.py &
	@sleep 1
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/gradio_ui/gradio_ui.py

run_x2robot_mock_tui:
	@echo "Starting MockX2RobotBridge in background..."
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/mcp_server_demo/x2robot_mcp_server/mock_x2robot_bridge.py &
	@sleep 1
	$(PYENV) $(UV_RUN_ROOT) python ${MAKEFILE_DIR}src/agent_demo/interaction_layer/tui/olympus_tui.py
