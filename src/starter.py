"""Module for launching agents.

エージェントを起動するためのモジュール.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
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
    except Exception:
        logger.exception("ゲームセッション中にエラーが発生しました")
        raise


async def cancel_task(task: asyncio.Task[None] | None) -> None:
    """Cancel and await an asyncio task if it is still running.

    実行中のasyncioタスクをキャンセルして待機する.

    Args:
        task (asyncio.Task[None] | None): Task to cancel / キャンセル対象のタスク
    """
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def handle_game_session_async(  # noqa: C901
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
    talk_task: asyncio.Task[None] | None = None
    whisper_task: asyncio.Task[None] | None = None

    send_lock = threading.Lock()

    def send_with_lock(text: str) -> None:
        with send_lock:
            client.send(text)

    while True:
        packet = await asyncio.to_thread(client.receive)

        if packet.request == Request.NAME:
            send_with_lock(name)
            continue

        if packet.request == Request.INITIALIZE:
            agent = init_agent_from_packet(config, name, packet)
        if not agent:
            raise ValueError(agent, "エージェントが初期化されていません")
        agent.set_packet(packet)

        match packet.request:
            # グループチャット方式
            case Request.TALK_PHASE_START:
                await cancel_task(talk_task)
                agent.in_talk_phase = True
                talk_task = asyncio.create_task(agent.handle_talk_phase(send_with_lock))
            case Request.TALK_PHASE_END:
                agent.in_talk_phase = False
                await cancel_task(talk_task)
                talk_task = None
            case Request.WHISPER_PHASE_START:
                await cancel_task(whisper_task)
                agent.in_whisper_phase = True
                whisper_task = asyncio.create_task(agent.handle_whisper_phase(send_with_lock))
            case Request.WHISPER_PHASE_END:
                agent.in_whisper_phase = False
                await cancel_task(whisper_task)
                whisper_task = None
            case _:
                # 従来のリクエスト処理
                # @timeoutデコレータ内でスレッドを使用しているため、asyncio.to_threadは不要
                req = agent.action()
                agent.agent_logger.packet(agent.request, req)
                if req:
                    send_with_lock(req)

        if packet.request == Request.FINISH:
            agent.in_talk_phase = False
            agent.in_whisper_phase = False
            await cancel_task(talk_task)
            await cancel_task(whisper_task)
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
