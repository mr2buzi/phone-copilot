from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IdentityFact:
    key: str
    value: str
    confidence: float = 1.0
    visibility: str = "safe"
    allowed_relationships: list[str] = field(default_factory=list)
    source: str = "manual"
    updated_at: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IdentityFact":
        return cls(
            key=str(payload.get("key") or ""),
            value=str(payload.get("value") or ""),
            confidence=float(payload.get("confidence", 1.0) or 0.0),
            visibility=str(payload.get("visibility") or "safe"),
            allowed_relationships=[str(item) for item in payload.get("allowed_relationships", []) if str(item).strip()],
            source=str(payload.get("source") or "manual"),
            updated_at=str(payload.get("updated_at")) if payload.get("updated_at") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Fictional example data. Replace with owner-approved facts before live use.
DEFAULT_IDENTITY_PACK: dict[str, Any] = {'owner_aliases': ['Alex', 'Owner'],
 'display_name': 'Alex (example profile)',
 'age_text': '19',
 'age_policy': {'default': '19',
                'if_other_person_says_older_than_19': 'match the conversational context naturally, '
                                                      'e.g. say same only if appropriate; '
                                                      'otherwise say 19 wby',
                'templates': ['19 wby', 'im 19', 'same']},
 'location_text': 'Northbridge x Sampleford',
 'home_location': 'Northbridge',
 'current_base': 'Sampleford for uni',
 'location_policy': {'safe_public': 'northbridge n sampleford',
                     'known_contacts': 'northbridge but sampleford for uni',
                     'unknown_contacts': 'vague city only, no exact area',
                     'never_disclose': ['exact address',
                                        'accommodation',
                                        'live location',
                                        'street',
                                        'building']},
 'study_text': 'computer science',
 'university_text': 'Sampleford University',
 'year_text': '',
 'uni_policy': {'known_contacts': 'comp sci at sampleford',
                'unknown_contacts': 'i study tech / computer science',
                'share_university_after': {'minimum_messages': 12,
                                           'minimum_thread_minutes': 10,
                                           'allowed_relationships': ['close_friend',
                                                                     'romantic_interest',
                                                                     'trusted_contact']},
                'if_age_context_conflicts': 'avoid year detail and just say comp sci'},
 'work_text': 'uni, software, projects, clients, gym, boxing',
 'what_do_you_do_style': 'say i study comp sci but have stuff on the side; make it sound '
                         'interesting/mysterious, not cringe, not fake-rich',
 'side_projects': [],
 'interests': ['coding',
               'gym',
               'boxing',
               'cars',
               'AI',
               'software',
               'business',
               'going out with friends'],
 'hobbies': ['reading', 'building projects', 'walking'],
 'daily_life': {'common_activities': ['chilling',
                                      'working',
                                      'coding',
                                      'uni',
                                      'gym',
                                      'boxing',
                                      'praying',
                                      'going out with friends',
                                      'in bed'],
                'wyd_templates': ['hmmm im chilling wby',
                                  'js in bed wby',
                                  'js working icl wby',
                                  'nothing crazy wby'],
                'what_been_up_to_templates': ['i just been working on this project its killing me',
                                              'been busy icl uni gym coding clients all of it',
                                              'js been working on stuff icl',
                                              'lowk had the craziest day between work gym and '
                                              'life'],
                'where_been_templates': ['been busy icl uni projects gym all of it',
                                         'just been buried in work and projects',
                                         'been around icl just had loads going on',
                                         'uni gym clients coding life all hit at once'],
                'doing_anything_nice_templates': ['nothing crazy icl just uni projects gym',
                                                  'not really just working on stuff',
                                                  'probably gym then coding icl',
                                                  'just got things to sort with work and projects'],
                'how_was_day_templates': ['ykw it was lowk good',
                                          'busy icl but calm',
                                          'packed icl gym coding uni all of it',
                                          'long but productive'],
                'why_tired_templates': ['boxing plus hella work icl',
                                        'gym boxing and clients draining me',
                                        'been coding and training all day im finished',
                                        'work life gym all hit at once icl']},
 'personality': {'style_summary': 'short, conversational replies; avoid unsupported personal '
                                  'claims'},
 'phrases_common': ['thanks', 'sounds good', 'see you then'],
 'phrases_to_avoid': ['fair just chilling too',
                      'same just chilling',
                      'same icl',
                      'what u saying then',
                      'fine then give me a topic',
                      'im trying icl give me a topic',
                      'calm'],
 'emoji_policy': {'allowed': ['😭', '☹️', '😒', '😂', '💀'],
                  'style': 'use sparingly, mostly no punctuation'},
 'punctuation_policy': {'default': 'minimal',
                        'avoid': ['formal punctuation',
                                  'customer-service tone',
                                  'over-polished sentences'],
                        'allow_caps_for_emphasis': True},
 'flirt_policy': {'allowed_relationships': ['romantic_interest', 'close_friend', 'trusted_contact'],
                  'blocked_relationships': ['family',
                                            'professional',
                                            'university',
                                            'unknown_if_too_early'],
                  'allowed_terms': ['my love',
                                    'baby',
                                    'beautiful',
                                    'behave',
                                    'dangerous thing to say icl'],
                  'review_only': True,
                  'explicit_sexual_autosend': False,
                  'safe_flirty_templates': {'i_miss_u': ['i miss u so much more baby',
                                                         'missed u more icl',
                                                         'gosh i missed u'],
                                            'im_horny': ['behave 😭',
                                                         'dangerous thing to say icl',
                                                         'u cant just say that and expect me to '
                                                         'act normal'],
                                            'say_it_back': ['i was getting there, missed u too',
                                                            'missed u too icl',
                                                            'course i missed u']},
                  'never_kill_mood_with': ['fair just chilling too',
                                           'same icl',
                                           "that's sweet icl alone",
                                           'what u saying then',
                                           'give me a topic']},
 'privacy_policy': {'safe_anyone': ['age',
                                    'general city pair Northbridge/Sampleford',
                                    'studies computer science',
                                    'likes gym/boxing/coding/cars',
                                    'has side projects'],
                    'close_friends_only': ['specific university',
                                           'specific project names',
                                           'client/work stress',
                                           'personal routine details'],
                    'never_disclose': ['exact address',
                                       'family drama',
                                       'private disputes',
                                       'financial details',
                                       'API keys',
                                       'precise live location',
                                       "other people's private information"],
                    'family_details': 'avoid completely unless user explicitly brings it up and '
                                      'relationship is trusted'},
 'direct_answer_templates': {'how_old_r_u': ['19 wby', 'im 19', 'same'],
                             'where_u_from': ['northbridge n sampleford wby',
                                              'northbridge but sampleford for uni'],
                             'what_do_u_study': ['comp sci wby', 'computer science'],
                             'what_do_u_do': ['comp sci but ive got stuff running on the side',
                                              'study comp sci but im building stuff outside uni',
                                              'uni plus projects icl'],
                             'wyd': ['hmmm im chilling wby', 'js in bed wby', 'js working wby'],
                             'what_u_been_up_to': ['i just been working on this project its '
                                                   'killing me',
                                                   'been busy icl uni gym coding clients all of '
                                                   'it'],
                             'where_u_been': ['been busy icl uni projects gym all of it',
                                              'just had loads going on icl'],
                             'doing_anything_nice': ['nothing crazy icl just uni projects gym',
                                                     'probably gym then coding icl'],
                             'how_was_ur_day': ['ykw it was lowk good', 'busy icl but calm'],
                             'why_tired': ['boxing plus hella work icl',
                                           'gym boxing and clients draining me'],
                             'u_good': ['good wby', 'yeah... why'],
                             'i_missed_u': ['i miss u so much more baby', 'missed u more icl'],
                             'im_bored': ['entertain me then dafuq', 'same icl do smth'],
                             'ur_dry': ['HOW WTF', 'u are not giving me anything to work with'],
                             'why_u_moving_weird': ['how wth', 'wdym weird'],
                             'are_u_ok': ['yeah... why', 'im good dw why']},
 'safe_facts': {'age': {'key': 'age',
                        'value': '19',
                        'confidence': 1.0,
                        'visibility': 'safe',
                        'allowed_relationships': ['unknown',
                                                  'close_friend',
                                                  'romantic_interest',
                                                  'trusted_contact',
                                                  'professional',
                                                  'university'],
                        'source': 'manual',
                        'updated_at': None},
                'general_location': {'key': 'general_location',
                                     'value': 'Northbridge x Sampleford',
                                     'confidence': 1.0,
                                     'visibility': 'safe',
                                     'allowed_relationships': ['unknown',
                                                               'close_friend',
                                                               'romantic_interest',
                                                               'trusted_contact'],
                                     'source': 'manual',
                                     'updated_at': None},
                'study': {'key': 'study',
                          'value': 'computer science',
                          'confidence': 1.0,
                          'visibility': 'safe',
                          'allowed_relationships': ['unknown',
                                                    'close_friend',
                                                    'romantic_interest',
                                                    'trusted_contact',
                                                    'professional',
                                                    'university'],
                          'source': 'manual',
                          'updated_at': None},
                'interests': {'key': 'interests',
                              'value': 'coding, gym, boxing, cars, AI, projects',
                              'confidence': 1.0,
                              'visibility': 'safe',
                              'allowed_relationships': ['unknown',
                                                        'close_friend',
                                                        'romantic_interest',
                                                        'trusted_contact'],
                              'source': 'manual',
                              'updated_at': None}},
 'private_facts': {'university': {'key': 'university',
                                  'value': 'Sampleford University',
                                  'confidence': 1.0,
                                  'visibility': 'private',
                                  'allowed_relationships': ['close_friend',
                                                            'romantic_interest',
                                                            'trusted_contact',
                                                            'university'],
                                  'source': 'manual',
                                  'updated_at': None}},
 'banned_disclosures': ['exact address',
                        'family drama',
                        'private disputes',
                        'financial details',
                        'API keys',
                        'precise live location',
                        "other people's private information"],
 'profile_kind': 'fictional_example'}


@dataclass
class IdentityPack:
    owner_aliases: list[str] = field(default_factory=list)
    display_name: str = ""
    age_text: str = ""
    location_text: str = ""
    study_text: str = ""
    university_text: str = ""
    work_text: str = ""
    interests: list[str] = field(default_factory=list)
    hobbies: list[str] = field(default_factory=list)
    gym_context: str | None = None
    career_context: str | None = None
    common_self_descriptions: list[str] = field(default_factory=list)
    safe_facts: dict[str, IdentityFact] = field(default_factory=dict)
    private_facts: dict[str, IdentityFact] = field(default_factory=dict)
    banned_disclosures: list[str] = field(default_factory=list)
    direct_answer_templates: dict[str, list[str]] = field(default_factory=dict)
    style_rules: dict[str, Any] = field(default_factory=dict)
    flirt_policy: dict[str, Any] = field(default_factory=dict)
    privacy_policy: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    loaded: bool = False
    last_loaded_at: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, path: Path | None = None, loaded: bool = True) -> "IdentityPack":
        safe_facts = {str(key): IdentityFact.from_dict(value) for key, value in dict(payload.get("safe_facts") or {}).items() if isinstance(value, dict)}
        private_facts = {str(key): IdentityFact.from_dict(value) for key, value in dict(payload.get("private_facts") or {}).items() if isinstance(value, dict)}
        return cls(
            owner_aliases=[str(item) for item in payload.get("owner_aliases", [])],
            display_name=str(payload.get("display_name") or ""),
            age_text=str(payload.get("age_text") or ""),
            location_text=str(payload.get("location_text") or ""),
            study_text=str(payload.get("study_text") or ""),
            university_text=str(payload.get("university_text") or ""),
            work_text=str(payload.get("work_text") or ""),
            interests=[str(item) for item in payload.get("interests", [])],
            hobbies=[str(item) for item in payload.get("hobbies", [])],
            gym_context=payload.get("gym_context"),
            career_context=payload.get("career_context"),
            common_self_descriptions=[str(item) for item in payload.get("common_self_descriptions", [])],
            safe_facts=safe_facts,
            private_facts=private_facts,
            banned_disclosures=[str(item) for item in payload.get("banned_disclosures", [])],
            direct_answer_templates={str(key): [str(item) for item in value] for key, value in dict(payload.get("direct_answer_templates") or {}).items() if isinstance(value, list)},
            style_rules=dict(payload.get("style_rules") or payload.get("personality") or {}),
            flirt_policy=dict(payload.get("flirt_policy") or {}),
            privacy_policy=dict(payload.get("privacy_policy") or {}),
            raw=payload,
            path=path,
            loaded=loaded,
            last_loaded_at=utc_now() if loaded else None,
        )

    @property
    def fact_count(self) -> int:
        return len(self.safe_facts) + len(self.private_facts)

    def safe_identity_summary(self) -> str:
        parts = [
            f"name {self.display_name}" if self.display_name else "",
            f"age {self.age_text}" if self.age_text else "",
            "from northbridge/sampleford" if self.location_text else "",
            f"studies {self.study_text}" if self.study_text else "",
            "likes " + ", ".join(self.interests[:5]) if self.interests else "",
        ]
        return "; ".join(part for part in parts if part)

    def disclosure_allowed(
        self,
        fact: IdentityFact,
        relationship_type: str,
        *,
        message_count: int = 0,
        thread_minutes: float | None = None,
    ) -> bool:
        relationship = relationship_type or "unknown"
        if fact.visibility == "sensitive":
            return False
        if fact.visibility == "safe":
            return not fact.allowed_relationships or relationship in fact.allowed_relationships
        if relationship not in {"close_friend", "romantic_interest", "trusted_contact"} or relationship not in set(fact.allowed_relationships or []):
            return False
        if fact.key == "university":
            policy = dict(self.raw.get("uni_policy") or {}).get("share_university_after", {})
            if isinstance(policy, dict):
                min_messages = int(policy.get("minimum_messages", 12) or 12)
                min_minutes = float(policy.get("minimum_thread_minutes", 10) or 10)
                allowed = {str(item) for item in policy.get("allowed_relationships", [])}
                if allowed and relationship not in allowed:
                    return False
                if message_count < min_messages:
                    return False
                if thread_minutes is not None and thread_minutes < min_minutes:
                    return False
        if fact.key == "project_names" and message_count < 6:
            return False
        return True

    def relevant_facts(
        self,
        kind: str,
        relationship_type: str,
        *,
        message_count: int = 0,
        thread_minutes: float | None = None,
    ) -> dict[str, str]:
        keys_by_kind = {
            "age": ["age"],
            "location": ["general_location"],
            "study": ["study", "university"],
            "work": ["study", "interests", "project_names"],
            "project": ["interests", "project_names"],
            "where_been": ["study", "interests"],
            "recent_activity": ["study", "interests"],
            "doing_anything_nice": ["study", "interests"],
        }
        facts: dict[str, str] = {}
        for key in keys_by_kind.get(kind, list(self.safe_facts)):
            fact = self.safe_facts.get(key) or self.private_facts.get(key)
            if fact and self.disclosure_allowed(fact, relationship_type, message_count=message_count, thread_minutes=thread_minutes):
                facts[key] = fact.value
        return facts

    def templates_for_scene(
        self,
        kind: str,
        relationship_type: str,
        *,
        flirt_allowed: bool = False,
        affection: bool = False,
        message_count: int = 0,
        thread_minutes: float | None = None,
    ) -> list[str]:
        university = self.private_facts.get("university")
        university_allowed = bool(
            university
            and self.disclosure_allowed(university, relationship_type, message_count=message_count, thread_minutes=thread_minutes)
        )
        templates = {
            "age": self.direct_answer_templates.get("how_old_r_u", ["19 wby", "im 19"]),
            "location": self.direct_answer_templates.get("where_u_from", ["northbridge n sampleford wby"]),
            "study": (["comp sci at sampleford", "sampleford uni icl"] if university_allowed else []) + self.direct_answer_templates.get("what_do_u_study", ["comp sci wby", "computer science"]),
            "work": self.direct_answer_templates.get("what_do_u_do", ["comp sci but ive got stuff running on the side"]),
            "project": ["this project im building is lowk killing me but it could be serious", "got a dev thing running on the side", "building software stuff icl"],
            "where_been": self.direct_answer_templates.get("where_u_been", ["been busy icl uni projects gym all of it"]),
            "recent_activity": self.direct_answer_templates.get("what_u_been_up_to", ["i just been working on this project its killing me"]),
            "doing_anything_nice": self.direct_answer_templates.get("doing_anything_nice", ["nothing crazy icl just uni projects gym"]),
        }.get(kind, [])
        if relationship_type == "unknown":
            templates = [self._conservative_template(kind, item) for item in templates]
        if affection and kind in {"where_been", "recent_activity"}:
            prefix = "missed u too icl, " if flirt_allowed else "that's sweet icl, "
            templates = [prefix + item[0].lower() + item[1:] if item else item for item in templates]
        return list(dict.fromkeys(item for item in templates if item))

    def _conservative_template(self, kind: str, template: str) -> str:
        if kind == "location":
            return "northbridge n sampleford wby"
        if kind == "study":
            return "computer science"
        if kind in {"work", "project"}:
            return "study comp sci but im building stuff outside uni"
        if kind in {"where_been", "recent_activity"}:
            return "been busy icl uni projects gym all of it"
        return template

    def to_safe_status(self) -> dict[str, Any]:
        return {
            "identity_pack_loaded": self.loaded,
            "identity_pack_path": str(self.path or ""),
            "identity_fact_count": self.fact_count,
            "safe_identity_summary": self.safe_identity_summary(),
            "identity_last_loaded_at": self.last_loaded_at,
        }


def load_identity_pack(path: Path) -> IdentityPack:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_IDENTITY_PACK, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = DEFAULT_IDENTITY_PACK
    return IdentityPack.from_dict(payload if isinstance(payload, dict) else DEFAULT_IDENTITY_PACK, path=path, loaded=True)
