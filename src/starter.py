"""Module for launching agents.

エージェントを起動するためのモジュール.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from utils.agent_utils import init_agent_from_packet

if TYPE_CHECKING:
    from agent.agent import Agent

from time import sleep

from aiwolf_nlp_common.client import Client
from aiwolf_nlp_common.packet import Request

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def create_client(config: dict[str, Any]) -> Client:
    """Create a client.

    クライアントの作成.

    Args:
        config (dict[str, Any]): Configuration dictionary / 設定辞書

    Returns:
        Client: Created client instance / 作成されたクライアントインスタンス
    """
    return Client(
        url=str(config["web_socket"]["url"]),
        token=(str(config["web_socket"]["token"]) if config["web_socket"]["token"] else None),
    )


def connect_to_server(client: Client, name: str) -> None:
    """Handle connection to the server.

    サーバーへの接続処理.

    Args:
        client (Client): Client instance / クライアントインスタンス
        name (str): Agent name / エージェント名
    """
    while True:
        try:
            client.connect()
            logger.info("エージェント %s がゲームサーバに接続しました", name)
            break
        except Exception as ex:  # noqa: BLE001
            logger.warning(
                "エージェント %s がゲームサーバに接続できませんでした",
                name,
            )
            logger.warning(ex)
            logger.info("再接続を試みます")
            sleep(15)


def handle_game_session(
    client: Client,
    config: dict[str, Any],
    name: str,
) -> None:
    """Handle game session.

    ゲームセッションの処理.

    Args:
        client (Client): Client instance / クライアントインスタンス
        config (dict[str, Any]): Configuration dictionary / 設定辞書
        name (str): Agent name / エージェント名
    """
    try:
        asyncio.run(handle_game_session_async(client, config, name))
    except Exception as ex:  # noqa: BLE001
        logger.error("ゲームセッション中にエラーが発生しました: %s", ex)
        raise


async def handle_game_session_async(
    client: Client,
    config: dict[str, Any],
    name: str,
) -> None:
    """Handle game session asynchronously.

    ゲームセッションの非同期処理.

    Args:
        client (Client): Client instance / クライアントインスタンス
        config (dict[str, Any]): Configuration dictionary / 設定辞書
        name (str): Agent name / エージェント名
    """
    agent: Agent | None = None
    talk_task: asyncio.Task | None = None
    whisper_task: asyncio.Task | None = None

    while True:
        # パケット受信（ブロッキング処理を非同期化）
        packet = await asyncio.to_thread(client.receive)

        if packet.request == Request.NAME:
            client.send(name)
            continue
        if packet.request == Request.INITIALIZE:
            agent = init_agent_from_packet(config, name, packet)
        if not agent:
            raise ValueError(agent, "エージェントが初期化されていません")
        agent.set_packet(packet)

        # グループチャット方式の処理
        match packet.request:
            case Request.TALK_PHASE_START:
                # トークフェーズ開始
                agent.in_talk_phase = True
                logger.info("トークフェーズが開始されました")
                # 非同期でトーク送信処理を開始
                talk_task = asyncio.create_task(agent.handle_talk_phase(client))

            case Request.TALK_PHASE_END:
                # トークフェーズ終了
                agent.in_talk_phase = False
                logger.info("トークフェーズが終了しました")
                # トーク送信タスクをキャンセル
                if talk_task and not talk_task.done():
                    talk_task.cancel()
                    try:
                        await talk_task
                    except asyncio.CancelledError:
                        logger.info("トーク送信タスクをキャンセルしました")

            case Request.WHISPER_PHASE_START:
                # 囁きフェーズ開始
                agent.in_whisper_phase = True
                logger.info("囁きフェーズが開始されました")
                # 非同期で囁き送信処理を開始
                whisper_task = asyncio.create_task(agent.handle_whisper_phase(client))

            case Request.WHISPER_PHASE_END:
                # 囁きフェーズ終了
                agent.in_whisper_phase = False
                logger.info("囁きフェーズが終了しました")
                # 囁き送信タスクをキャンセル
                if whisper_task and not whisper_task.done():
                    whisper_task.cancel()
                    try:
                        await whisper_task
                    except asyncio.CancelledError:
                        logger.info("囁き送信タスクをキャンセルしました")

            case Request.TALK_BROADCAST | Request.WHISPER_BROADCAST:
                # 新着トーク/囁き（set_packet内で処理済み）
                pass

            case _:
                # 従来のリクエスト処理
                req = await asyncio.to_thread(agent.action)
                agent.agent_logger.packet(agent.request, req)
                if req:
                    client.send(req)

        if packet.request == Request.FINISH:
            break


def connect(config: dict[str, Any], idx: int = 1) -> None:
    """Launch an agent.

    エージェントを起動する.

    Args:
        config (dict[str, Any]): Configuration dictionary / 設定辞書
        idx (int): Agent index (default: 1) / エージェントインデックス (デフォルト: 1)
    """
    name = str(config["agent"]["team"]) + str(idx)
    while True:
        try:
            client = create_client(config)
            connect_to_server(client, name)
            try:
                handle_game_session(client, config, name)
            finally:
                client.close()
                logger.info("エージェント %s とゲームサーバの接続を切断しました", name)
        except Exception as ex:  # noqa: BLE001
            logger.warning(
                "エージェント %s がエラーで終了しました",
                name,
            )
            logger.warning(ex)

        if not bool(config["web_socket"]["auto_reconnect"]):
            break
