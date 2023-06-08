from typing import Optional

from determined.common import api
from determined.common.api import bindings


class User:
    """
    A User object represents an individual account on a Determined installation.
    It can be obtained from ``client.list_users`` or ``client.get_user_by_name()``.

    Attributes:
        session: HTTP request session.
        user_id: (int) Unique ID for the user in the Determined database.
        admin: (Mutable, bool) Whether the user has admin privileges.
        remote: (Mutable, bool) When true, prevents password sign-on and requires user to
        sign-on using external IdP
        agent_uid: (Mutable, int) UID on the agent this user is linked to.
        agent_gid: (Mutable, int) GID on the agent this user is linked to.
        agent_user: (Mutable, str) Unix user on the agent this user is linked to.
        agent_group: (Mutable, str) Unix group on the agent this user is linked to.
        display_name: (Mutable, str) Human-friendly name of the user.

    Note:
        All attributes are cached by default.

        Mutable properties may be changed by methods that update these values either automatically
        (eg. `rename`, `change_display_name`) or explicitly with `reload()`.
    """

    def __init__(self, user_id: int, session: api.Session):
        self.user_id = user_id
        self._session = session

        self.username = None  # type: Optional[str]
        self.admin = None  # type: Optional[bool]
        self.active = None  # type: Optional[bool]
        self.remote = None  # type: Optional[bool]
        self.agent_uid = None  # type: Optional[int]
        self.agent_gid = None  # type: Optional[int]
        self.agent_user = None  # type: Optional[str]
        self.agent_group = None  # type: Optional[str]
        self.display_name = None  # type: Optional[str]

    def _hydrate(self, user: bindings.v1User) -> None:
        self.username = user.username
        self.admin = user.admin
        self.remote = user.remote or False
        self.active = user.active or True
        self.display_name = user.displayName
        if user.agentUserGroup is not None:
            self.agent_uid = user.agentUserGroup.agentUid
            self.agent_gid = user.agentUserGroup.agentGid
            self.agent_user = user.agentUserGroup.agentUser
            self.agent_group = user.agentUserGroup.agentGroup

    def reload(self) -> None:
        resp = bindings.get_GetUser(session=self._session, userId=self.user_id).user
        self._hydrate(resp)

    def rename(self, new_username: str) -> None:
        patch_user = bindings.v1PatchUser(username=new_username)
        bindings.patch_PatchUser(self._session, body=patch_user, userId=self.user_id)
        self.reload()

    def activate(self) -> None:
        patch_user = bindings.v1PatchUser(active=True)
        bindings.patch_PatchUser(self._session, body=patch_user, userId=self.user_id)
        self.reload()

    def deactivate(self) -> None:
        patch_user = bindings.v1PatchUser(active=False)
        bindings.patch_PatchUser(self._session, body=patch_user, userId=self.user_id)
        self.reload()

    def change_display_name(self, display_name: str) -> None:
        patch_user = bindings.v1PatchUser(displayName=display_name)
        bindings.patch_PatchUser(self._session, body=patch_user, userId=self.user_id)
        self.reload()

    def change_password(self, new_password: str) -> None:
        new_password = api.salt_and_hash(new_password)
        patch_user = bindings.v1PatchUser(password=new_password, isHashed=True)
        bindings.patch_PatchUser(self._session, body=patch_user, userId=self.user_id)

    def link_with_agent(
        self,
        agent_uid: Optional[int] = None,
        agent_gid: Optional[int] = None,
        agent_user: Optional[str] = None,
        agent_group: Optional[str] = None,
    ) -> None:
        v1agent_user_group = bindings.v1AgentUserGroup(
            agentGid=agent_gid,
            agentGroup=agent_group,
            agentUid=agent_uid,
            agentUser=agent_user,
        )
        patch_user = bindings.v1PatchUser(agentUserGroup=v1agent_user_group)
        bindings.patch_PatchUser(self._session, body=patch_user, userId=self.user_id)
        self.reload()

    @classmethod
    def _from_bindings(cls, user_bindings: bindings.v1User, session: api.Session) -> "User":
        assert user_bindings.id
        user = cls(session=session, user_id=user_bindings.id)
        user._hydrate(user_bindings)
        return user
