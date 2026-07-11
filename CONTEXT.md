# DwarfStar Server

The local inference server coordinates one mutable model session while presenting stateless wire protocols to clients.

## Language

**Live Session**:
The one mutable model and KV state used for local inference at a time.
_Avoid_: Shared session, client session

**Session-State Transaction**:
One admitted request's controlled attempt to reuse or extend the Live Session, ending in exactly one recorded state disposition. It does not imply that already-streamed output can be retracted.
_Avoid_: Database transaction, response transaction

**Terminal Outcome**:
The single typed record of how an admitted request ended, including its primary reason and the resulting Live Session disposition.
_Avoid_: Finish reason, HTTP status
