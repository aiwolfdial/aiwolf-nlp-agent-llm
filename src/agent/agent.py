"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable
    from aiwolf_nlp_common.client import Client

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        # グループチャット方式用のフィールド
        self.in_talk_phase = False
        self.in_whisper_phase = False
        self.last_talk_time = 0.0
        self.last_whisper_time = 0.0
        self.talk_interval_min = 1.0  # 最小発言間隔（秒）
        self.talk_interval_max = 3.0  # 最大発言間隔（秒）

        self.comments: list[str] = []
        with Path.open(
            Path(str(self.config["path"]["random_talk"])),
            encoding="utf-8",
        ) as f:
            self.comments = f.read().splitlines()

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)

        # 新着トークの処理（グループチャット方式用）
        if packet.new_talk:
            self.talk_history.append(packet.new_talk)
            self.on_talk_received(packet.new_talk)
        if packet.new_whisper:
            self.whisper_history.append(packet.new_whisper)
            self.on_whisper_received(packet.new_whisper)

        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def on_talk_received(self, talk: Talk) -> None:
        """Called when a new talk is received (freeform mode).

        新しいトークを受信した時に呼ばれる（グループチャット方式用）.

        Args:
            talk (Talk): Received talk / 受信したトーク
        """

    def on_whisper_received(self, whisper: Talk) -> None:
        """Called when a new whisper is received (freeform mode).

        新しい囁きを受信した時に呼ばれる（グループチャット方式用）.

        Args:
            whisper (Talk): Received whisper / 受信した囁き
        """

    async def handle_talk_phase(self, client: Client) -> None:
        """Handle talk phase in freeform mode.

        グループチャット方式でのトークフェーズ処理.

        Args:
            client (Client): Client instance / クライアントインスタンス
        """
        self.agent_logger.logger.info("トークフェーズの処理を開始します")

        while self.in_talk_phase:
            # 残り回数チェック
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                if self.in_talk_phase:
                    client.send("Over")
                    self.agent_logger.logger.info("残り回数が0のためOverを送信しました")
                break

            # トーク送信判断
            if self.should_send_talk():
                try:
                    # タイムアウト付きでトーク生成
                    text = await asyncio.wait_for(
                        self.generate_talk_async(),
                        timeout=10.0,
                    )

                    # フェーズ終了チェック（生成中にphase endを受信した可能性）
                    if not self.in_talk_phase:
                        self.agent_logger.logger.info("トーク生成後にフェーズ終了を検出しました")
                        break

                    if text and text.strip():
                        client.send(text)
                        self.last_talk_time = time.time()
                        self.agent_logger.logger.info("トークを送信しました: %s", text)

                        if text.strip() == "Over":
                            self.agent_logger.logger.info("Overを送信したため終了します")
                            break

                except asyncio.TimeoutError:
                    self.agent_logger.logger.warning("トーク生成がタイムアウトしました")
                except Exception as e:  # noqa: BLE001
                    self.agent_logger.logger.error("トーク生成中にエラーが発生しました: %s", e)

            # 次のチェックまで待機
            await asyncio.sleep(0.1)

        self.agent_logger.logger.info("トークフェーズの処理を終了しました")

    async def handle_whisper_phase(self, client: Client) -> None:
        """Handle whisper phase in freeform mode.

        グループチャット方式での囁きフェーズ処理.

        Args:
            client (Client): Client instance / クライアントインスタンス
        """
        self.agent_logger.logger.info("囁きフェーズの処理を開始します")

        while self.in_whisper_phase:
            # 残り回数チェック
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                if self.in_whisper_phase:
                    client.send("Over")
                    self.agent_logger.logger.info("残り回数が0のためOverを送信しました")
                break

            # 囁き送信判断
            if self.should_send_whisper():
                try:
                    # タイムアウト付きで囁き生成
                    text = await asyncio.wait_for(
                        self.generate_whisper_async(),
                        timeout=10.0,
                    )

                    # フェーズ終了チェック（生成中にphase endを受信した可能性）
                    if not self.in_whisper_phase:
                        self.agent_logger.logger.info("囁き生成後にフェーズ終了を検出しました")
                        break

                    if text and text.strip():
                        client.send(text)
                        self.last_whisper_time = time.time()
                        self.agent_logger.logger.info("囁きを送信しました: %s", text)

                        if text.strip() == "Over":
                            self.agent_logger.logger.info("Overを送信したため終了します")
                            break

                except asyncio.TimeoutError:
                    self.agent_logger.logger.warning("囁き生成がタイムアウトしました")
                except Exception as e:  # noqa: BLE001
                    self.agent_logger.logger.error("囁き生成中にエラーが発生しました: %s", e)

            # 次のチェックまで待機
            await asyncio.sleep(0.1)

        self.agent_logger.logger.info("囁きフェーズの処理を終了しました")

    def should_send_talk(self) -> bool:
        """Determine whether to send a talk.

        トークを送信するべきか判断する.

        Returns:
            bool: True if should send talk / トークを送信する場合True
        """
        # 最小間隔チェック
        elapsed = time.time() - self.last_talk_time
        if elapsed < self.talk_interval_min:
            return False

        # ランダムなタイミング（最大間隔を超えたら高確率で送信）
        if elapsed > self.talk_interval_max:
            return random.random() < 0.8  # noqa: S311

        # 一定確率で送信
        return random.random() < 0.2  # noqa: S311

    def should_send_whisper(self) -> bool:
        """Determine whether to send a whisper.

        囁きを送信するべきか判断する.

        Returns:
            bool: True if should send whisper / 囁きを送信する場合True
        """
        # トークと同じロジック
        elapsed = time.time() - self.last_whisper_time
        if elapsed < self.talk_interval_min:
            return False

        if elapsed > self.talk_interval_max:
            return random.random() < 0.8  # noqa: S311

        return random.random() < 0.2  # noqa: S311

    async def generate_talk_async(self) -> str:
        """Generate talk asynchronously.

        トークを非同期生成する.

        Returns:
            str: Generated talk / 生成されたトーク
        """
        # デフォルト実装：既存のtalk()を呼ぶ
        return await asyncio.to_thread(self.talk)

    async def generate_whisper_async(self) -> str:
        """Generate whisper asynchronously.

        囁きを非同期生成する.

        Returns:
            str: Generated whisper / 生成された囁き
        """
        # デフォルト実装：既存のwhisper()を呼ぶ
        return await asyncio.to_thread(self.whisper)

    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        return random.choice(self.comments)  # noqa: S311

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        return random.choice(self.comments)  # noqa: S311

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return random.choice(self.get_alive_agents())  # noqa: S311

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return random.choice(self.get_alive_agents())  # noqa: S311

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return random.choice(self.get_alive_agents())  # noqa: S311

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return random.choice(self.get_alive_agents())  # noqa: S311

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
