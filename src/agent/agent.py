"""エージェントの基底クラスを定義するモジュール."""

from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

from utils import agent_util

if TYPE_CHECKING:
    import configparser
    from collections.abc import Callable

    from utils.agent_logger import AgentLogger
    
    from google.genai import Client
    from google.genai.chats import Chat

import random
from threading import Thread

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk
from google import genai
from google.genai.types import Part, UserContent


class Agent:
    """エージェントの基底クラス."""

    def __init__(
        self,
        config: configparser.ConfigParser | None = None,
        name: str | None = None,
        logger: AgentLogger | None = None,  # noqa: ARG002
    ) -> None:
        """エージェントの初期化を行う."""
        self.config = config
        self.agent_name: str = name if name is not None else ""
        self.agent_logger: AgentLogger | None = None
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.idx: int = -1
        self.role: Role | None = None

        self.client: Client | None = None
        self.chat: Chat | None
        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0

    @staticmethod
    def timeout(func: Callable) -> Callable:
        """アクションタイムアウトを設定するデコレータ."""

        def _wrapper(self, *args, **kwargs) -> str:  # noqa: ANN001, ANN002, ANN003
            res = ""

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(self, *args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = Thread(target=execute_with_timeout, daemon=True)
            thread.start()

            timeout_value = (
                self.info.action_timeout
                if self.info is not None and hasattr(self.info, "action_timeout")
                else 0
            )
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
            else:
                thread.join()

            if isinstance(res, Exception):
                raise res

            return res

        return _wrapper

    @staticmethod
    def send_agent_index(func: Callable) -> Callable:
        """エージェントのインデックスをインデックス付き文字列のエージェント名に変換するデコレータ."""

        def _wrapper(self, *args, **kwargs) -> str:  # noqa: ANN001, ANN002, ANN003
            res = func(self, *args, **kwargs)
            if type(res) is not int:
                return res
            return agent_util.agent_idx_to_agent(idx=res)

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """パケット情報をセットする."""
        if packet is None:
            return
        self.request = packet.request
        if packet.info is not None:
            self.info = packet.info
        if packet.setting is not None:
            self.setting = packet.setting
        if packet.talk_history is not None:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history is not None:
            self.whisper_history.extend(packet.whisper_history)

        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
            if self.info is None:
                return
            if self.info.agent is None or self.info.role_map is None:
                return
            self.idx = agent_util.agent_name_to_idx(name=self.info.agent)
            self.role = self.info.role_map.get(self.info.agent)

    def get_alive_agents(self) -> list[str]:
        """生存しているエージェントのリストを取得する."""
        if self.info is None:
            return []
        if self.info.status_map is None:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    @timeout
    def name(self) -> str:
        """名前リクエストに対する応答を返す."""
        return self.agent_name

    def initialize(self) -> None:
        """ゲーム開始リクエストに対する初期化処理を行う."""
        if self.config is None or self.info is None:
            return
        self.client = genai.Client(api_key=self.config.get("gemini", "api_key"))
        self.chat = self.client.chats.create(model=self.config.get("gemini", "model"))
        self.chat.send_message(
            f"""
            あなたは人狼ゲームのエージェントです。
            あなたの名前は{self.info.agent}です。
            あなたの役職は{self.role}です。

            これからゲームを進行していきます。リクエストが来た際には、適切な応答を返してください。
            トークリクエストと囁きリクエストに対しては、ゲーム内で発言するべき内容のみを出力してください。
            他のリクエストに対しては、行動の対象となるエージェントの名前のみを出力してください。

            説明は以上です。「はい」と出力してください。
            """,
        )

    def daily_initialize(self) -> None:
        """昼開始リクエストに対する処理を行う."""

    @timeout
    def whisper(self) -> str:
        """囁きリクエストに対する応答を返す."""
        if self.chat is None:
            return ""
        message = f"""
        囁きリクエスト
        履歴:
        {"\n".join(
            [
                f"{w.agent}: {w.text}"
                for w in self.whisper_history[self.sent_whisper_count :]
            ]
        )}
        """
        sleep(10)
        self.sent_whisper_count = len(self.whisper_history)
        return self.chat.send_message(message).text or ""

    @timeout
    def talk(self) -> str:
        """トークリクエストに対する応答を返す."""
        if self.chat is None:
            return ""
        message = f"""
        トークリクエスト
        履歴:
        {"\n".join(
            [
                f"{t.agent}: {t.text}"
                for t in self.talk_history[self.sent_talk_count :]
            ]
        )}
        """
        sleep(10)
        self.sent_talk_count = len(self.talk_history)
        return self.chat.send_message(message).text or ""

    def daily_finish(self) -> None:
        """昼終了リクエストに対する処理を行う."""

    @timeout
    @send_agent_index
    def divine(self) -> int:
        """占いリクエストに対する応答を返す."""
        if self.chat is None:
            return agent_util.agent_name_to_idx(
                name=random.choice(self.get_alive_agents()),  # noqa: S311
            )
        message = f"""
        占いリクエスト
        対象:
        {"\n".join(self.get_alive_agents())}
        """
        sleep(10)
        return agent_util.agent_name_to_idx(
            self.chat.send_message(message).text
            or random.choice(  # noqa: S311
                self.get_alive_agents(),
            ),
        )

    @timeout
    @send_agent_index
    def guard(self) -> int:
        """護衛リクエストに対する応答を返す."""
        if self.chat is None:
            return agent_util.agent_name_to_idx(
                name=random.choice(self.get_alive_agents()),  # noqa: S311
            )
        message = f"""
        護衛リクエスト
        対象:
        {"\n".join(self.get_alive_agents())}
        """
        sleep(10)
        return agent_util.agent_name_to_idx(
            self.chat.send_message(message).text
            or random.choice(  # noqa: S311
                self.get_alive_agents(),
            ),
        )

    @timeout
    @send_agent_index
    def vote(self) -> int:
        """投票リクエストに対する応答を返す."""
        if self.chat is None:
            return agent_util.agent_name_to_idx(
                name=random.choice(self.get_alive_agents()),  # noqa: S311
            )
        message = f"""
        投票リクエスト
        対象:
        {"\n".join(self.get_alive_agents())}
        """
        sleep(10)
        return agent_util.agent_name_to_idx(
            self.chat.send_message(message).text
            or random.choice(  # noqa: S311
                self.get_alive_agents(),
            ),
        )

    @timeout
    @send_agent_index
    def attack(self) -> int:
        """襲撃リクエストに対する応答を返す."""
        if self.chat is None:
            return agent_util.agent_name_to_idx(
                name=random.choice(self.get_alive_agents()),  # noqa: S311
            )
        message = f"""
        襲撃リクエスト
        対象:
        {"\n".join(self.get_alive_agents())}
        """
        sleep(10)
        return agent_util.agent_name_to_idx(
            self.chat.send_message(message).text
            or random.choice(  # noqa: S311
                self.get_alive_agents(),
            ),
        )

    def finish(self) -> None:
        """ゲーム終了リクエストに対する処理を行う."""

    def action(self) -> str | None:  # noqa: C901, PLR0911
        """リクエストの種類に応じたアクションを実行する."""
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
        return None

    def transfer_state(self, prev_agent: Agent) -> None:
        """エージェントの状態を別のエージェントにコピーする."""
        for attr_name, attr_value in vars(prev_agent).items():
            if not attr_name.startswith("__"):
                setattr(self, attr_name, attr_value)
