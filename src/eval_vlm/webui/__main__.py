"""eval_vlm Web UI 启动入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from .app import create_app
from .settings import Settings, set_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eval-vlm-webui",
        description="eval_vlm Web UI 服务: VLM 测试集可视化审查、配置与任务评测平台",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="绑定主机地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="绑定端口号 (默认: 8080)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="工作区目录 (默认: 全局配置中的 workspace 路径)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开发热重载模式",
    )

    args = parser.parse_args()

    settings = Settings(
        workspace_dir=args.workspace,
        host=args.host,
        port=args.port,
    )
    set_settings(settings)

    print(f"============================================================")
    print(f" eval_vlm Web UI 启动中...")
    print(f" 访问地址:  http://{args.host}:{args.port}")
    print(f" 工作区目录: {settings.workspace}")
    print(f" 状态目录:  {settings.state_dir}")
    print(f"============================================================")

    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
